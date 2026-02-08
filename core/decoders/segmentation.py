"""Segmentation decoder heads: LinearSegHead, UPerNetHead, MaskTransformerHead."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.decoders.base import BaseDecoder
from core.encoders.base import BaseEncoder


class LinearSegHead(BaseDecoder):
    """Per-patch linear classifier for semantic segmentation.

    The simplest segmentation head: applies a linear layer to each patch token
    and upsamples to the original image resolution.
    """

    task = "segmentation"

    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        output_size: int = 518,
    ) -> None:
        super().__init__(encoder, num_classes)
        self.output_size = output_size
        self.classifier = nn.Conv2d(self._embed_dim, num_classes, 1)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict per-pixel class logits.

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        spatial = features["spatial_features"]  # (B, D, H', W')
        logits = self.classifier(spatial)  # (B, num_classes, H', W')
        logits = F.interpolate(
            logits,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return logits


class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module for multi-scale context aggregation."""

    def __init__(self, in_channels: int, pool_sizes: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        for size in pool_sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1),
                    nn.ReLU(),
                )
            )
        self.bottleneck = nn.Conv2d(
            in_channels + (in_channels // len(pool_sizes)) * len(pool_sizes),
            in_channels,
            3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pooled = [x]
        for stage in self.stages:
            pooled.append(
                F.interpolate(
                    stage(x), size=(h, w), mode="bilinear", align_corners=False
                )
            )
        return self.bottleneck(torch.cat(pooled, dim=1))


class UPerNetHead(BaseDecoder):
    """UPerNet-style segmentation decoder.

    Multi-scale feature aggregation with Pyramid Pooling Module and FPN.
    A strong baseline decoder (~15M params) that produces high-quality
    segmentation maps.

    Args:
        encoder: Frozen encoder.
        num_classes: Number of segmentation classes.
        fpn_channels: Number of channels in the FPN.
        output_size: Output spatial resolution.
        intermediate_layers: Which encoder layers to use for multi-scale features.
    """

    task = "segmentation"

    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        fpn_channels: int = 256,
        output_size: int = 518,
        intermediate_layers: list[int] | None = None,
    ) -> None:
        super().__init__(encoder, num_classes)
        self.output_size = output_size
        self.fpn_channels = fpn_channels
        self._intermediate_layers = intermediate_layers or [3, 6, 9, 11]
        num_scales = len(self._intermediate_layers)

        # PPM on the last scale
        self.ppm = PyramidPoolingModule(self._embed_dim)

        # Lateral connections
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(self._embed_dim, fpn_channels, 1) for _ in range(num_scales)]
        )

        # FPN convolutions
        self.fpn_convs = nn.ModuleList(
            [nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1) for _ in range(num_scales)]
        )

        # Fusion and classification
        self.fusion = nn.Conv2d(fpn_channels * num_scales, fpn_channels, 1)
        self.classifier = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(fpn_channels, num_classes, 1),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict per-pixel class logits using UPerNet.

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        if "intermediate" in features:
            multi_scale = features["intermediate"]
        else:
            spatial = features["spatial_features"]
            multi_scale = [spatial] * len(self.lateral_convs)

        # Apply PPM to last scale
        multi_scale = list(multi_scale)
        multi_scale[-1] = self.ppm(multi_scale[-1])

        # Lateral connections
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, multi_scale)
        ]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # FPN convolutions
        fpn_out = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Upsample all to the largest scale and concatenate
        target_size = fpn_out[0].shape[-2:]
        upsampled = [
            F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            for feat in fpn_out
        ]
        fused = self.fusion(torch.cat(upsampled, dim=1))

        logits = self.classifier(fused)
        logits = F.interpolate(
            logits,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return logits


class MaskTransformerHead(BaseDecoder):
    """Mask Transformer segmentation head.

    Uses learnable class queries with cross-attention to patch features,
    then generates masks via dot-product between query embeddings and
    patch features. Most flexible segmentation head.

    Args:
        encoder: Frozen encoder.
        num_classes: Number of segmentation classes.
        num_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension for the transformer.
        output_size: Output spatial resolution.
    """

    task = "segmentation"

    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        num_layers: int = 2,
        num_heads: int = 8,
        hidden_dim: int = 256,
        output_size: int = 518,
    ) -> None:
        super().__init__(encoder, num_classes)
        self.output_size = output_size
        self.hidden_dim = hidden_dim

        # Project encoder features to hidden_dim
        self.input_proj = nn.Linear(self._embed_dim, hidden_dim)

        # Learnable class queries
        self.class_queries = nn.Parameter(torch.randn(1, num_classes, hidden_dim) * 0.02)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Mask embedding projection
        self.mask_embed = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict segmentation masks via dot-product attention.

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        patch_tokens = features["patch_tokens"]  # (B, N, D_enc)
        B, N, _ = patch_tokens.shape
        h = w = int(math.sqrt(N))

        # Project to hidden dim
        memory = self.input_proj(patch_tokens)  # (B, N, D)

        # Expand class queries
        queries = self.class_queries.expand(B, -1, -1)  # (B, num_classes, D)

        # Decode
        decoded = self.transformer_decoder(queries, memory)  # (B, num_classes, D)

        # Generate masks via dot product
        mask_embeds = self.mask_embed(decoded)  # (B, num_classes, D)
        masks = torch.bmm(mask_embeds, memory.transpose(1, 2))  # (B, num_classes, N)
        masks = masks.reshape(B, self.num_classes, h, w)

        # Upsample to output size
        masks = F.interpolate(
            masks,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return masks
