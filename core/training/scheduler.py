"""Learning rate scheduler utilities."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    """Cosine annealing scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class WarmupStepScheduler(LRScheduler):
    """Step decay scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        step_size: int = 30,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            effective_epoch = self.last_epoch - self.warmup_epochs
            decay = self.gamma ** (effective_epoch // self.step_size)
            return [base_lr * decay for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    total_epochs: int = 50,
    warmup_epochs: int = 5,
    **kwargs,
) -> LRScheduler:
    """Build a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: One of 'cosine', 'step', 'constant'.
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of warmup epochs.

    Returns:
        A PyTorch LR scheduler.
    """
    if scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_type == "step":
        return WarmupStepScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
