"""VLM training: VLMLightningModule + VLMTrainer.

Two-stage training loop mirroring trainer.py patterns:
  Stage 1 — Alignment: only MLP projector is trained (LLM frozen).
  Stage 2 — Instruction tuning: projector + LLM LoRA adapters are trained.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from core.decoders.vlm import VLMDecoder
from core.training.callbacks import EarlyStoppingCallback, MetricsLoggerCallback
from core.training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


class VLMLightningModule(pl.LightningModule):
    """PyTorch Lightning module for VLM two-stage training.

    Args:
        decoder: VLMDecoder (encoder + projector + LLM).
        lr: Learning rate for the optimizer.
        scheduler_type: LR scheduler type ('cosine', 'step', 'constant').
        warmup_epochs: Number of linear warmup epochs.
        total_epochs: Total training epochs (used by scheduler).
        stage: Training stage (1 = alignment, 2 = instruction tuning).
        weight_decay: AdamW weight decay.
    """

    def __init__(
        self,
        decoder: VLMDecoder,
        lr: float = 1e-3,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 1,
        total_epochs: int = 10,
        stage: int = 1,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.stage = stage
        self.weight_decay = weight_decay

        # Freeze encoder base weights; re-enable LoRA params if present
        self.decoder.encoder.freeze()
        if getattr(self.decoder.encoder, "_lora_enabled", False):
            for name, param in self.decoder.encoder.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

    def _extract_features(self, batch: dict) -> dict[str, torch.Tensor]:
        ctx = (
            contextlib.nullcontext()
            if getattr(self.decoder.encoder, "_lora_enabled", False)
            else torch.no_grad()
        )
        with ctx:
            return self.decoder.encoder.forward_features(batch["image"])

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        features = self._extract_features(batch)
        output = self.decoder(
            features,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = output["loss"]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        features = self._extract_features(batch)
        output = self.decoder(
            features,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = output["loss"]
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Token accuracy: among answer-token positions, how many did we predict right?
        logits = output["logits"]           # (B, V+T, vocab)
        B = logits.shape[0]
        V = self.decoder.num_visual_tokens
        device = logits.device

        # Reconstruct full_labels (same layout as in VLMDecoder.forward)
        visual_labels = torch.full((B, V), -100, dtype=torch.long, device=device)
        full_labels = torch.cat([visual_labels, batch["labels"]], dim=1)  # (B, V+T)

        # Causal shift: logit at position j predicts label at position j+1
        shift_logits = logits[:, :-1].contiguous()      # (B, V+T-1, vocab)
        shift_labels = full_labels[:, 1:].contiguous()  # (B, V+T-1)

        mask = shift_labels != -100
        if mask.any():
            preds = shift_logits[mask].argmax(-1)
            acc = (preds == shift_labels[mask]).float().mean()
            self.log("val_token_acc", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> dict:
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class VLMTrainer:
    """High-level VLM trainer wrapping PyTorch Lightning.

    Args:
        decoder: VLMDecoder instance.
        train_dataset: Training dataset (PetsVQADataset or _PetsVQASubset).
        val_dataset: Validation dataset. If None, split from train_dataset.
        lr: Learning rate.
        epochs: Training epochs.
        batch_size: Batch size.
        stage: 1 (alignment) or 2 (instruction tuning).
        checkpoint_dir: Directory for checkpoints.
        num_workers: DataLoader worker processes.
        val_ratio: Validation fraction (used only when val_dataset is None).
        scheduler: LR scheduler type.
        warmup_epochs: Number of warmup epochs.
        early_stopping_patience: Epochs without improvement before stopping (0=off).
        accelerator: PyTorch Lightning accelerator ('auto', 'gpu', 'cpu').
        devices: Number of devices to use.
    """

    def __init__(
        self,
        decoder: VLMDecoder,
        train_dataset,
        val_dataset=None,
        lr: float = 1e-3,
        epochs: int = 3,
        batch_size: int = 4,
        stage: int = 1,
        checkpoint_dir: str | Path = "./checkpoints/vlm",
        num_workers: int = 0,
        val_ratio: float = 0.2,
        scheduler: str = "cosine",
        warmup_epochs: int = 1,
        early_stopping_patience: int = 5,
        accelerator: str = "auto",
        devices: int | str = "auto",
    ) -> None:
        self.decoder = decoder
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._num_workers = num_workers

        if val_dataset is None:
            train_dataset, val_dataset = train_dataset.split(val_ratio)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.lightning_module = VLMLightningModule(
            decoder=decoder,
            lr=lr,
            scheduler_type=scheduler,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            stage=stage,
        )

        callbacks = [MetricsLoggerCallback()]

        if early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    mode="min",
                )
            )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename=f"vlm-stage{stage}-{{epoch}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

        # Use bf16-mixed on GPU for memory efficiency (Phi-3.5 is 3.8B params)
        precision = "bf16-mixed" if torch.cuda.is_available() else 32

        self.pl_trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=5,
            default_root_dir=str(self.checkpoint_dir),
            precision=precision,
        )

    def fit(self) -> dict:
        """Run training and return results dict."""
        from core.data.vqa_dataset import PetsVQADataset

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            collate_fn=PetsVQADataset.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=PetsVQADataset.collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        logger.info(
            "VLM Stage %d: %d train, %d val, %d epochs, lr=%.2e, trainable=%d",
            self.lightning_module.stage,
            len(self.train_dataset),
            len(self.val_dataset),
            self.epochs,
            self.lightning_module.lr,
            self.decoder.num_trainable_params(),
        )

        self.pl_trainer.fit(
            self.lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        logged = self.pl_trainer.logged_metrics
        results = {
            "best_model_path": self.pl_trainer.checkpoint_callback.best_model_path,
            "best_val_loss": float(
                self.pl_trainer.checkpoint_callback.best_model_score or 0
            ),
            "epochs_trained": self.pl_trainer.current_epoch,
        }
        if "val_token_acc" in logged:
            results["val_token_acc"] = float(logged["val_token_acc"])

        return results
