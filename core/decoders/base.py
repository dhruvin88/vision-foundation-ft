"""Abstract base decoder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn

from core.encoders.base import BaseEncoder

TaskType = Literal["classification", "detection", "segmentation"]


class BaseDecoder(ABC, nn.Module):
    """Abstract base class for all task-specific decoders/heads.

    Decoders are lightweight modules that sit on top of a frozen encoder.
    Only decoder parameters are trained.
    """

    task: TaskType

    def __init__(self, encoder: BaseEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self._embed_dim = encoder.embed_dim

    @abstractmethod
    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass on pre-extracted encoder features.

        Args:
            features: Dictionary of encoder features from encoder.forward_features().

        Returns:
            Task-specific predictions.
        """

    def predict(self, images: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run full inference: encode then decode.

        Args:
            images: Input images of shape (B, 3, H, W).

        Returns:
            Task-specific predictions.
        """
        with torch.no_grad():
            features = self.encoder.forward_features(images)
        return self.forward(features)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the decoder's trainable parameters (excludes frozen encoder)."""
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.trainable_parameters())
