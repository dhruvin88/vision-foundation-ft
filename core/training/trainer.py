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

    def forward(self, features: dict[str, torch.Tensor]) -> Any:
        return self.decoder(features)

    def _compute_loss(self, predictions: Any, batch: dict) -> torch.Tensor:
        """Compute task-specific loss."""
        if self.task == "classification":
            return F.cross_entropy(predictions, batch["label"])

        elif self.task == "detection":
            # DETR-style detection loss with Hungarian matching
            from scipy.optimize import linear_sum_assignment

            pred_logits = predictions["pred_logits"]  # (B, Q, C) where C = num_classes + 1
            pred_boxes = predictions["pred_boxes"]  # (B, Q, 4) cxcywh normalized
            B, Q, C = pred_logits.shape
            num_classes = C - 1  # last class index is "no object"

            target_labels = batch["labels"]  # (B, M) with -1 padding
            target_boxes = batch["boxes"]  # (B, M, 4) cxcywh normalized

            # Hungarian matching per sample
            matched_tgt_labels = []
            matched_tgt_boxes = []

            for b in range(B):
                valid = target_labels[b] >= 0
                gt_labels = target_labels[b][valid]
                gt_boxes = target_boxes[b][valid]
                num_gt = gt_labels.shape[0]

                # Default: all queries predict "no object"
                tgt_labels = torch.full(
                    (Q,), num_classes, dtype=torch.long, device=pred_logits.device
                )
                tgt_boxes = torch.zeros(
                    (Q, 4), dtype=torch.float32, device=pred_boxes.device
                )

                if num_gt > 0:
                    # Cost matrix for Hungarian matching
                    with torch.no_grad():
                        probs = pred_logits[b].softmax(-1)  # (Q, C)
                        cls_cost = -probs[:, gt_labels]  # (Q, num_gt)
                        bbox_cost = torch.cdist(
                            pred_boxes[b], gt_boxes, p=1
                        )  # (Q, num_gt)
                        cost = cls_cost + 5.0 * bbox_cost

                    row_idx, col_idx = linear_sum_assignment(
                        cost.detach().cpu().numpy()
                    )
                    for r, c in zip(row_idx, col_idx):
                        tgt_labels[r] = gt_labels[c]
                        tgt_boxes[r] = gt_boxes[c]

                matched_tgt_labels.append(tgt_labels)
                matched_tgt_boxes.append(tgt_boxes)

            target_labels = torch.stack(matched_tgt_labels)  # (B, Q)
            target_boxes = torch.stack(matched_tgt_boxes)  # (B, Q, 4)

            # Classification loss with down-weighted no-object class
            eos_coef = 0.1
            weight = torch.ones(C, device=pred_logits.device)
            weight[num_classes] = eos_coef
            cls_loss = F.cross_entropy(
                pred_logits.reshape(-1, C),
                target_labels.reshape(-1),
                weight=weight,
            )

            # Bbox regression loss (L1) only on matched queries
            valid_mask = target_labels.reshape(-1) < num_classes
            if valid_mask.any():
                bbox_loss = F.l1_loss(
                    pred_boxes.reshape(-1, 4)[valid_mask],
                    target_boxes.reshape(-1, 4)[valid_mask],
                )
            else:
                bbox_loss = torch.tensor(0.0, device=pred_logits.device)

            return cls_loss + 5.0 * bbox_loss

        elif self.task == "segmentation":
            return F.cross_entropy(predictions, batch["mask"])

        raise ValueError(f"Unknown task: {self.task}")

    def _extract_features(self, batch: dict) -> dict[str, torch.Tensor]:
        """Extract encoder features from batch images."""
        images = batch["image"]
        with torch.no_grad():
            features = self.decoder.encoder.forward_features(images)
        return features

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        if "features" in batch:
            features = batch["features"]
        else:
            features = self._extract_features(batch)

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

        self._results = {
            "best_model_path": self.pl_trainer.checkpoint_callback.best_model_path,
            "best_val_loss": float(self.pl_trainer.checkpoint_callback.best_model_score or 0),
            "epochs_trained": self.pl_trainer.current_epoch,
        }

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
