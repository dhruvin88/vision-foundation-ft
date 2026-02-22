"""Training loop built on PyTorch Lightning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.decoders.base import BaseDecoder
from core.training.callbacks import EarlyStoppingCallback, MetricsLoggerCallback
from core.training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


def _compute_cdn_loss(
    cdn_logits: torch.Tensor,
    cdn_boxes_pred: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    G: int,
    M: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """CDN loss: pos queries vs GT (CE+L1+GIoU), neg queries vs zeros.

    Args:
        cdn_logits:    (B, G*2M, C)
        cdn_boxes_pred: (B, G*2M, 4)
        gt_labels:     (B, M_max) padded with -1
        gt_boxes:      (B, M_max, 4) padded
        G:             number of CDN groups
        M:             max GT per image used when building CDN
    """
    from torchvision.ops import generalized_box_iou

    B = cdn_logits.shape[0]
    total_loss = torch.zeros(1, device=device)[0]
    pos_count = 0

    def cx2xy(b_: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = b_.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1).clamp(0, 1)

    for b in range(B):
        valid = gt_labels[b] >= 0
        labels_b = gt_labels[b][valid][:M]
        boxes_b = gt_boxes[b][valid][:M]
        n_valid = labels_b.shape[0]

        for g in range(G):
            pos_start = g * 2 * M
            pos_end = pos_start + M
            neg_end = pos_start + 2 * M

            pos_logits = cdn_logits[b, pos_start:pos_end]    # (M, C)
            pos_boxes = cdn_boxes_pred[b, pos_start:pos_end]  # (M, 4)
            neg_logits = cdn_logits[b, pos_end:neg_end]       # (M, C)

            # Negative: BCE all-zeros
            neg_target = torch.zeros_like(neg_logits)
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, neg_target, reduction="mean"
            )
            total_loss = total_loss + neg_loss

            if n_valid > 0:
                n = n_valid
                # Positive: supervised against GT
                pos_cls_target = torch.zeros(n, num_classes, device=device)
                pos_cls_target[torch.arange(n, device=device), labels_b] = 1.0
                pos_cls_loss = F.binary_cross_entropy_with_logits(
                    pos_logits[:n], pos_cls_target, reduction="mean"
                )
                l1_loss = F.l1_loss(pos_boxes[:n], boxes_b)
                giou = generalized_box_iou(cx2xy(pos_boxes[:n]), cx2xy(boxes_b))
                giou_loss = (1 - giou.diag()).mean()
                total_loss = total_loss + pos_cls_loss + 5.0 * l1_loss + 2.0 * giou_loss
                pos_count += n

    return total_loss / max(pos_count, 1)


class DecoderLightningModule(pl.LightningModule):
    """PyTorch Lightning module wrapping a decoder for training.

    Handles the training/validation loop, loss computation, and metric tracking
    for classification, detection, and segmentation tasks.
    """

    def __init__(
        self,
        decoder: BaseDecoder,
        lr: float = 1e-3,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        weight_decay: float = 0.01,
        total_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.task = decoder.task
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.total_epochs = total_epochs

        # Ensure encoder is frozen and not part of optimizer
        self.decoder.encoder.freeze()

        # mAP metric for detection validation (IoU-based, proper COCO-style)
        if self.task == "detection":
            from torchmetrics.detection import MeanAveragePrecision
            self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def forward(self, features: dict[str, torch.Tensor]) -> Any:
        return self.decoder(features)

    def _detection_loss_single(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """DETR loss for one set of predictions: Hungarian matching + cls + L1 + GIoU.

        Args:
            pred_logits: (B, Q, C) — C = num_classes + 1
            pred_boxes:  (B, Q, 4) — cxcywh normalized
            target_labels: (B, M) — class indices, -1 for padding
            target_boxes:  (B, M, 4) — cxcywh normalized, padded
        """
        from scipy.optimize import linear_sum_assignment
        from torchvision.ops import generalized_box_iou

        B, Q, C = pred_logits.shape
        num_classes = C - 1

        matched_tgt_labels = []
        matched_tgt_boxes = []

        for b in range(B):
            valid = target_labels[b] >= 0
            gt_labels = target_labels[b][valid]
            gt_boxes = target_boxes[b][valid]
            num_gt = gt_labels.shape[0]

            tgt_labels = torch.full(
                (Q,), num_classes, dtype=torch.long, device=pred_logits.device
            )
            tgt_boxes = torch.zeros(
                (Q, 4), dtype=torch.float32, device=pred_boxes.device
            )

            if num_gt > 0:
                with torch.no_grad():
                    probs = pred_logits[b].softmax(-1)
                    cls_cost = -probs[:, gt_labels]              # (Q, num_gt)
                    bbox_cost = torch.cdist(pred_boxes[b], gt_boxes, p=1)
                    cost = cls_cost + 5.0 * bbox_cost

                row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())
                for r, c in zip(row_idx, col_idx):
                    tgt_labels[r] = gt_labels[c]
                    tgt_boxes[r] = gt_boxes[c]

            matched_tgt_labels.append(tgt_labels)
            matched_tgt_boxes.append(tgt_boxes)

        tgt_labels = torch.stack(matched_tgt_labels)  # (B, Q)
        tgt_boxes = torch.stack(matched_tgt_boxes)    # (B, Q, 4)

        # --- Classification loss (down-weight no-object class) ---
        eos_coef = 0.1
        weight = torch.ones(C, device=pred_logits.device)
        weight[num_classes] = eos_coef
        cls_loss = F.cross_entropy(pred_logits.reshape(-1, C), tgt_labels.reshape(-1), weight=weight)

        # --- Box losses only on matched (foreground) queries ---
        valid_mask = tgt_labels.reshape(-1) < num_classes
        if valid_mask.any():
            matched_pred = pred_boxes.reshape(-1, 4)[valid_mask]  # cxcywh
            matched_tgt = tgt_boxes.reshape(-1, 4)[valid_mask]

            # L1 loss
            l1_loss = F.l1_loss(matched_pred, matched_tgt)

            # GIoU loss — convert cxcywh → xyxy first
            def cxcywh_to_xyxy(boxes):
                cx, cy, w, h = boxes.unbind(-1)
                return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

            giou = generalized_box_iou(
                cxcywh_to_xyxy(matched_pred.clamp(0, 1)),
                cxcywh_to_xyxy(matched_tgt.clamp(0, 1)),
            )
            giou_loss = (1 - giou.diag()).mean()
        else:
            l1_loss = torch.tensor(0.0, device=pred_logits.device)
            giou_loss = torch.tensor(0.0, device=pred_logits.device)

        return cls_loss + 5.0 * l1_loss + 2.0 * giou_loss

    @staticmethod
    def _varifocal_loss(
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
        iou_scores: torch.Tensor,
        num_classes: int,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Varifocal classification loss for RT-DETR.

        Args:
            pred_logits:   (B, Q, C) raw logits (no background class)
            target_labels: (B, Q) matched class indices, -1 for unmatched
            iou_scores:    (B, Q) IoU of matched predictions (0 for unmatched)
            num_classes:   number of foreground classes
        """
        B, Q, C = pred_logits.shape
        device = pred_logits.device

        targets = torch.zeros(B, Q, C, device=device)
        valid_mask = target_labels >= 0
        if valid_mask.any():
            b_idx, q_idx = torch.where(valid_mask)
            targets[b_idx, q_idx, target_labels[valid_mask]] = iou_scores[valid_mask]

        with torch.no_grad():
            pred_sig = pred_logits.sigmoid()
            weight = alpha * pred_sig.pow(gamma) * (1 - targets) + targets
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, targets, weight=weight, reduction="sum"
        )
        pos_count = float(valid_mask.sum().clamp(min=1))
        return loss / pos_count

    def _detection_loss_rtdetr(
        self,
        predictions: dict,
        target_labels: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """RT-DETR detection loss: VFL + L1 + GIoU + encoder aux + CDN."""
        from scipy.optimize import linear_sum_assignment
        from torchvision.ops import generalized_box_iou

        pred_logits = predictions["pred_logits"]  # (B, Q, C)
        pred_boxes = predictions["pred_boxes"]    # (B, Q, 4)
        B, Q, C = pred_logits.shape
        num_classes = C
        device = pred_logits.device

        def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
            cx, cy, w, h = boxes.unbind(-1)
            return torch.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
            ).clamp(0, 1)

        # --- Hungarian matching on final layer ---
        matched_indices: list[tuple[list, list]] = []
        full_tgt_labels = torch.full((B, Q), -1, dtype=torch.long, device=device)
        full_tgt_boxes = torch.zeros(B, Q, 4, device=device)

        for b in range(B):
            valid = target_labels[b] >= 0
            gt_labels_b = target_labels[b][valid]
            gt_boxes_b = target_boxes[b][valid]
            num_gt = gt_labels_b.shape[0]

            if num_gt == 0:
                matched_indices.append(([], []))
                continue

            with torch.no_grad():
                scores = pred_logits[b].sigmoid()  # (Q, C)
                cls_cost = -scores[:, gt_labels_b]  # (Q, num_gt)
                bbox_cost = torch.cdist(pred_boxes[b], gt_boxes_b, p=1)
                pred_xyxy = cxcywh_to_xyxy(pred_boxes[b])
                gt_xyxy = cxcywh_to_xyxy(gt_boxes_b)
                giou = generalized_box_iou(pred_xyxy, gt_xyxy)
                cost = cls_cost + 5.0 * bbox_cost + 2.0 * (-giou)

            row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())
            matched_indices.append((row_idx.tolist(), col_idx.tolist()))
            for r, c in zip(row_idx, col_idx):
                full_tgt_labels[b, r] = gt_labels_b[c]
                full_tgt_boxes[b, r] = gt_boxes_b[c]

        # IoU scores for VFL targets
        with torch.no_grad():
            iou_scores = torch.zeros(B, Q, device=device)
            for b in range(B):
                rows, cols = matched_indices[b]
                if not rows:
                    continue
                valid = target_labels[b] >= 0
                gt_boxes_b = target_boxes[b][valid]
                pred_xyxy_b = cxcywh_to_xyxy(pred_boxes[b])
                gt_xyxy_b = cxcywh_to_xyxy(gt_boxes_b)
                giou_mat = generalized_box_iou(pred_xyxy_b, gt_xyxy_b)
                for r, c in zip(rows, cols):
                    iou_scores[b, r] = giou_mat[r, c].clamp(0)

        # VFL classification loss
        vfl = self._varifocal_loss(pred_logits, full_tgt_labels, iou_scores, num_classes)

        # L1 + GIoU on matched foreground queries
        valid_mask = full_tgt_labels >= 0
        if valid_mask.any():
            matched_pred = pred_boxes[valid_mask]
            matched_tgt = full_tgt_boxes[valid_mask]
            l1_loss = F.l1_loss(matched_pred, matched_tgt)
            giou_mat = generalized_box_iou(
                cxcywh_to_xyxy(matched_pred), cxcywh_to_xyxy(matched_tgt)
            )
            giou_loss = (1 - giou_mat.diag()).mean()
        else:
            l1_loss = torch.tensor(0.0, device=device)
            giou_loss = torch.tensor(0.0, device=device)

        total_loss = vfl + 5.0 * l1_loss + 2.0 * giou_loss

        # Auxiliary decoder layer losses (re-use same matching)
        for aux in predictions.get("aux_outputs", []):
            aux_vfl = self._varifocal_loss(
                aux["pred_logits"], full_tgt_labels, iou_scores, num_classes
            )
            if valid_mask.any():
                mp = aux["pred_boxes"][valid_mask]
                mt = full_tgt_boxes[valid_mask]
                aux_l1 = F.l1_loss(mp, mt)
                giou_aux = generalized_box_iou(cxcywh_to_xyxy(mp), cxcywh_to_xyxy(mt))
                aux_giou = (1 - giou_aux.diag()).mean()
            else:
                aux_l1 = torch.tensor(0.0, device=device)
                aux_giou = torch.tensor(0.0, device=device)
            total_loss = total_loss + aux_vfl + 5.0 * aux_l1 + 2.0 * aux_giou

        # Encoder auxiliary loss (all-background BCE, weight 0.5)
        enc_out = predictions.get("enc_outputs")
        if enc_out is not None:
            enc_logits = enc_out["pred_logits"]
            enc_target = torch.zeros_like(enc_logits)
            enc_loss = 0.5 * F.binary_cross_entropy_with_logits(
                enc_logits, enc_target, reduction="mean"
            )
            total_loss = total_loss + enc_loss

        # CDN loss
        cdn_outputs = predictions.get("cdn_outputs")
        if cdn_outputs is not None:
            cdn_meta = predictions["cdn_meta"]
            cdn_gt_labels = predictions["cdn_gt_labels"]
            cdn_gt_boxes = predictions["cdn_gt_boxes"]
            for cdn_out in cdn_outputs:
                cdn_loss = _compute_cdn_loss(
                    cdn_out["pred_logits"],
                    cdn_out["pred_boxes"],
                    cdn_gt_labels,
                    cdn_gt_boxes,
                    cdn_meta["G"],
                    cdn_meta["M"],
                    num_classes,
                    device,
                )
                total_loss = total_loss + cdn_loss

        return total_loss

    def _compute_loss(self, predictions: Any, batch: dict) -> torch.Tensor:
        """Compute task-specific loss."""
        if self.task == "classification":
            return F.cross_entropy(predictions, batch["label"])

        elif self.task == "detection":
            target_labels = batch["labels"]
            target_boxes = batch["boxes"]

            if "enc_outputs" in predictions:
                return self._detection_loss_rtdetr(predictions, target_labels, target_boxes)

            # DETRLite path — unchanged
            loss = self._detection_loss_single(
                predictions["pred_logits"],
                predictions["pred_boxes"],
                target_labels,
                target_boxes,
            )
            for aux in predictions.get("aux_outputs", []):
                loss = loss + self._detection_loss_single(
                    aux["pred_logits"],
                    aux["pred_boxes"],
                    target_labels,
                    target_boxes,
                )
            return loss

        elif self.task == "segmentation":
            return F.cross_entropy(predictions, batch["mask"])

        raise ValueError(f"Unknown task: {self.task}")

    def _extract_features(self, batch: dict) -> dict[str, torch.Tensor]:
        """Extract encoder features from batch images."""
        images = batch["image"]
        with torch.no_grad():
            features = self.decoder.encoder.forward_features(images)
        features["image"] = images  # thread raw pixels for CNN branch
        return features

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        if "features" in batch:
            features = batch["features"]
            if self.task == "detection" and "image" in batch:
                features["image"] = batch["image"]
        else:
            features = self._extract_features(batch)

        if self.task == "detection":
            features["gt_labels"] = batch["labels"]
            features["gt_boxes"] = batch["boxes"]

        predictions = self.decoder(features)
        loss = self._compute_loss(predictions, batch)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        if "features" in batch:
            features = batch["features"]
        else:
            features = self._extract_features(batch)

        predictions = self.decoder(features)
        loss = self._compute_loss(predictions, batch)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Task-specific metrics
        if self.task == "classification":
            preds = predictions.argmax(dim=-1)
            acc = (preds == batch["label"]).float().mean()
            self.log("val_acc", acc, prog_bar=True, sync_dist=True)

        elif self.task == "detection":
            if "pred_logits" in predictions and "pred_boxes" in predictions:
                pred_logits = predictions["pred_logits"]  # (B, Q, C)
                pred_boxes = predictions["pred_boxes"]    # (B, Q, 4) cxcywh
                B, Q, C = pred_logits.shape

                def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
                    cx, cy, w, h = boxes.unbind(-1)
                    return torch.stack(
                        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
                    ).clamp(0, 1)

                # Build torchmetrics-compatible per-image prediction/target dicts
                map_preds, map_targets = [], []
                for b in range(B):
                    if "enc_outputs" in predictions:
                        # RTDETRDecoder: sigmoid scores, no background class.
                        # sigmoid(0) = 0.5, so use 0.3 threshold (RT-DETR standard)
                        # to avoid flooding the metric with low-confidence noise.
                        scores, labels = pred_logits[b].sigmoid().max(-1)
                        score_thresh = 0.3
                    else:
                        # DETRLite: softmax with background at index num_classes
                        probs = pred_logits[b].softmax(-1)  # (Q, C)
                        scores, labels = probs[:, :num_classes].max(-1)
                        score_thresh = 0.05

                    # Filter out background queries (low fg score)
                    fg_mask = scores > score_thresh
                    map_preds.append({
                        "boxes": cxcywh_to_xyxy(pred_boxes[b][fg_mask]),
                        "scores": scores[fg_mask],
                        "labels": labels[fg_mask],
                    })

                    valid = batch["labels"][b] >= 0
                    map_targets.append({
                        "boxes": cxcywh_to_xyxy(batch["boxes"][b][valid]),
                        "labels": batch["labels"][b][valid],
                    })

                self.val_map.update(map_preds, map_targets)

    def on_validation_epoch_end(self) -> None:
        if self.task == "detection":
            map_metrics = self.val_map.compute()
            self.log("val_map50", map_metrics["map_50"], prog_bar=True, sync_dist=True)
            self.log("val_map", map_metrics["map"], prog_bar=True, sync_dist=True)
            self.val_map.reset()

    def configure_optimizers(self) -> dict:
        # Only optimize decoder parameters
        optimizer = torch.optim.AdamW(
            self.decoder.trainable_parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = build_scheduler(
            optimizer,
            scheduler_type=self.scheduler_type,
            total_epochs=self.total_epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class Trainer:
    """High-level trainer that wraps PyTorch Lightning for easy use.

    Args:
        decoder: The decoder/head to train (encoder must be attached and frozen).
        train_dataset: Training dataset.
        val_dataset: Validation dataset (if None, split from train_dataset).
        lr: Learning rate.
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        scheduler: LR scheduler type ('cosine', 'step', 'constant').
        warmup_epochs: Number of warmup epochs.
        augmentation: Augmentation preset ('none', 'light', 'heavy').
        early_stopping_patience: Stop after N epochs without improvement (0 to disable).
        checkpoint_dir: Directory to save checkpoints.
        num_workers: Number of data loading workers.
        accelerator: PyTorch Lightning accelerator ('auto', 'gpu', 'cpu').
        devices: Number of devices to use.
        val_ratio: Validation split ratio (used only if val_dataset is None).
    """

    def __init__(
        self,
        decoder: BaseDecoder,
        train_dataset,
        val_dataset=None,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        scheduler: str = "cosine",
        warmup_epochs: int = 5,
        augmentation: str = "light",
        early_stopping_patience: int = 10,
        checkpoint_dir: str | Path = "./checkpoints",
        num_workers: int = 4,
        accelerator: str = "auto",
        devices: int | str = "auto",
        val_ratio: float = 0.2,
    ) -> None:
        self.decoder = decoder
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Split dataset if no validation set provided
        if val_dataset is None:
            train_dataset, val_dataset = train_dataset.split(val_ratio)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Create Lightning module
        self.lightning_module = DecoderLightningModule(
            decoder=decoder,
            lr=lr,
            scheduler_type=scheduler,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
        )

        # Configure callbacks
        callbacks = [MetricsLoggerCallback()]

        if early_stopping_patience > 0:
            monitor = "val_acc" if decoder.task == "classification" else "val_loss"
            mode = "max" if decoder.task == "classification" else "min"
            callbacks.append(
                EarlyStoppingCallback(
                    monitor=monitor,
                    patience=early_stopping_patience,
                    mode=mode,
                )
            )

        # Checkpoint callback
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename="best-{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # Create Lightning trainer
        self.pl_trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=10,
            default_root_dir=str(self.checkpoint_dir),
        )

        self._num_workers = num_workers
        self._results: dict | None = None

    def fit(self) -> dict:
        """Run training and return results.

        Returns:
            Dictionary with training results including best metrics.
        """
        # Use custom collate for detection (variable-length boxes/labels)
        collate_fn = None
        if self.decoder.task == "detection":
            from core.data.dataset import FFTDataset

            collate_fn = FFTDataset.detection_collate_fn

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        logger.info(
            "Starting training: %d train samples, %d val samples, %d epochs",
            len(self.train_dataset),
            len(self.val_dataset),
            self.epochs,
        )
        logger.info(
            "Decoder: %s (%d trainable params)",
            type(self.decoder).__name__,
            self.decoder.num_trainable_params(),
        )

        self.pl_trainer.fit(
            self.lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        # Extract validation metrics from logged values
        logged_metrics = self.pl_trainer.logged_metrics

        self._results = {
            "best_model_path": self.pl_trainer.checkpoint_callback.best_model_path,
            "best_val_loss": float(self.pl_trainer.checkpoint_callback.best_model_score or 0),
            "epochs_trained": self.pl_trainer.current_epoch,
        }

        # Add task-specific validation metrics if available
        if "val_acc" in logged_metrics:
            self._results["val_acc"] = float(logged_metrics["val_acc"])
        if "val_map50" in logged_metrics:
            self._results["val_map50"] = float(logged_metrics["val_map50"])
        if "val_map" in logged_metrics:
            self._results["val_map"] = float(logged_metrics["val_map"])

        return self._results

    def save(self, path: str | Path) -> None:
        """Save decoder weights only (not the encoder).

        Args:
            path: Path to save the decoder weights (.pt file).
        """
        from core.export.weights import save_decoder_weights

        save_decoder_weights(self.decoder, path)

    def load(self, path: str | Path) -> None:
        """Load decoder weights.

        Args:
            path: Path to the saved decoder weights (.pt file).
        """
        from core.export.weights import load_decoder_weights

        load_decoder_weights(self.decoder, path)
