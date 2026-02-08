"""Abstract base encoder interface for foundation models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """Abstract base class for all encoders.

    Encoders are always frozen -- no gradients flow through them.
    They extract feature maps from input images that decoders use for downstream tasks.
    """

    def __init__(self) -> None:
        super().__init__()
        self._frozen = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature tensor. Shape depends on the encoder:
            - CLS token: (B, embed_dim)
            - Patch tokens: (B, num_patches, embed_dim)
        """

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract multi-scale or detailed features.

        Returns a dict with keys like 'cls_token', 'patch_tokens',
        and optionally intermediate layer features for multi-scale heads.
        """

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Embedding dimension of the encoder output."""

    @property
    @abstractmethod
    def patch_size(self) -> int:
        """Patch size used by the vision transformer."""

    @property
    @abstractmethod
    def num_patches(self) -> int:
        """Number of patches for the default input size."""

    def freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self._frozen = True
        self.eval()

    def train(self, mode: bool = True) -> BaseEncoder:
        """Override train to keep encoder in eval mode when frozen."""
        if self._frozen:
            return super().train(False)
        return super().train(mode)
