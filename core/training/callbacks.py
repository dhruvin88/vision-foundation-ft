"""Training callbacks for early stopping, metrics logging, etc."""

from __future__ import annotations

import logging
import time
from typing import Any

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(pl.callbacks.EarlyStopping):
    """Early stopping callback with logging."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.001,
    ) -> None:
        super().__init__(
            monitor=monitor,
            patience=patience,
            mode=mode,
            min_delta=min_delta,
            verbose=True,
        )

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger.info(
            "Early stopping enabled: monitor=%s, patience=%d, mode=%s",
            self.monitor,
            self.patience,
            self.mode,
        )


class MetricsLoggerCallback(pl.Callback):
    """Callback that logs training metrics and tracks progress."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_start_time: float = 0
        self.train_start_time: float = 0
        self.history: list[dict[str, Any]] = []

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.train_start_time = time.time()
        logger.info("Training started")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch_time = time.time() - self.epoch_start_time
        metrics = {
            k: float(v) for k, v in trainer.callback_metrics.items() if isinstance(v, (int, float))
        }
        # Also handle tensor values
        for k, v in trainer.callback_metrics.items():
            if k not in metrics:
                try:
                    metrics[k] = float(v)
                except (TypeError, ValueError):
                    pass

        metrics["epoch"] = trainer.current_epoch
        metrics["epoch_time_s"] = epoch_time
        self.history.append(metrics)

        logger.info(
            "Epoch %d/%d (%.1fs): %s",
            trainer.current_epoch + 1,
            trainer.max_epochs,
            epoch_time,
            ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k not in ("epoch", "epoch_time_s")),
        )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        total_time = time.time() - self.train_start_time
        logger.info("Training completed in %.1f seconds", total_time)

    def get_history(self) -> list[dict[str, Any]]:
        """Return the full metrics history."""
        return self.history
