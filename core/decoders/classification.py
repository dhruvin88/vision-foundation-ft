"""Classification decoder heads: LinearProbe, MLPHead, TransformerHead."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.decoders.base import BaseDecoder
from core.encoders.base import BaseEncoder


class LinearProbe(BaseDecoder):
    """Single linear layer classification head.

    Operates on the CLS token from the encoder. Simplest head,
    often surprisingly effective with DINOv2 features.
    """

    task = "classification"

    def __init__(self, encoder: BaseEncoder, num_classes: int) -> None:
        super().__init__(encoder, num_classes)
        self.head = nn.Linear(self._embed_dim, num_classes)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        cls_token = features["cls_token"]  # (B, embed_dim)
        return self.head(cls_token)  # (B, num_classes)


class MLPHead(BaseDecoder):
    """Two-layer MLP classification head with dropout.

    Provides slightly more capacity than a linear probe.
    """

    task = "classification"

    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(encoder, num_classes)
        self.head = nn.Sequential(
            nn.Linear(self._embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        cls_token = features["cls_token"]  # (B, embed_dim)
        return self.head(cls_token)  # (B, num_classes)


class TransformerHead(BaseDecoder):
    """Transformer-based classification head with learnable CLS queries.

    Uses cross-attention from learnable queries to encoder patch tokens,
    providing the most expressive classification head.
    """

    task = "classification"

    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(encoder, num_classes)

        self.cls_query = nn.Parameter(torch.randn(1, 1, self._embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self._embed_dim,
            nhead=num_heads,
            dim_feedforward=self._embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.classifier = nn.Linear(self._embed_dim, num_classes)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        patch_tokens = features["patch_tokens"]  # (B, N, D)
        B = patch_tokens.shape[0]

        # Expand learnable CLS query for the batch
        queries = self.cls_query.expand(B, -1, -1)  # (B, 1, D)

        # Cross-attend to patch tokens
        decoded = self.transformer_decoder(queries, patch_tokens)  # (B, 1, D)
        decoded = decoded.squeeze(1)  # (B, D)

        return self.classifier(decoded)  # (B, num_classes)
