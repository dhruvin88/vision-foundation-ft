"""Detection decoder heads: DETRLite, FPNHead."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.decoders.base import BaseDecoder
from core.encoders.base import BaseEncoder


class DETRLiteDecoder(BaseDecoder):
    """Lightweight DETR-style object detection decoder.

    Uses learnable object queries with cross-attention to encoder patch features,
    followed by FFN heads for bounding box regression and classification.

    Args:
        encoder: Frozen encoder providing patch features.
        num_classes: Number of object classes (excluding background).
        num_queries: Number of learnable object queries (max detections per image).
        num_decoder_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension (projected from encoder embed_dim if different).
        dropout: Dropout rate.
    """

    task = "detection"

    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        num_queries: int = 100,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(encoder, num_classes)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Project encoder features to hidden_dim if needed
        self.input_proj = nn.Linear(self._embed_dim, hidden_dim)

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Positional encoding for spatial features
        self.pos_embed = nn.Parameter(
            torch.randn(1, encoder.num_patches, hidden_dim) * 0.02
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Prediction heads
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # (cx, cy, w, h) normalized
        )

    def _predict(self, decoded: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply prediction heads to a decoder output tensor."""
        return {
            "pred_logits": self.class_head(decoded),
            "pred_boxes": self.bbox_head(decoded).sigmoid(),
        }

    def forward(
        self, features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run detection on encoder features.

        Returns:
            Dictionary with:
            - 'pred_logits': (B, num_queries, num_classes + 1)
            - 'pred_boxes': (B, num_queries, 4) -- normalized (cx, cy, w, h)
            - 'aux_outputs': list of dicts with same keys for intermediate layers
        """
        patch_tokens = features["patch_tokens"]  # (B, N, D_enc)
        B = patch_tokens.shape[0]

        # Project to hidden dim and add positional encoding
        memory = self.input_proj(patch_tokens) + self.pos_embed  # (B, N, D)

        # Expand object queries for the batch
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        # Run decoder layer by layer to collect intermediate outputs
        x = queries
        aux_outputs = []
        for i, layer in enumerate(self.transformer_decoder.layers):
            x = layer(x, memory)
            if i < len(self.transformer_decoder.layers) - 1:
                aux_outputs.append(self._predict(x))

        # Final layer predictions
        out = self._predict(x)
        out["aux_outputs"] = aux_outputs
        return out


class FPNHead(BaseDecoder):
    """Feature Pyramid Network detection head.

    Builds a feature pyramid from intermediate encoder layers and applies
    anchor-based classification and regression heads at each level.

    Requires the encoder to provide intermediate layer features.

    Args:
        encoder: Frozen encoder with intermediate layer support.
        num_classes: Number of object classes.
        fpn_channels: Number of channels in the FPN.
        num_anchors: Number of anchors per spatial location.
        intermediate_layers: Which encoder layers to use for multi-scale features.
    """

    task = "detection"

    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        fpn_channels: int = 256,
        num_anchors: int = 9,
        intermediate_layers: list[int] | None = None,
    ) -> None:
        super().__init__(encoder, num_classes)
        self.fpn_channels = fpn_channels
        self.num_anchors = num_anchors
        self._intermediate_layers = intermediate_layers or [3, 6, 9, 11]
        num_scales = len(self._intermediate_layers)

        # Lateral connections (project each intermediate layer to fpn_channels)
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(self._embed_dim, fpn_channels, 1) for _ in range(num_scales)]
        )

        # Top-down pathway convolutions
        self.fpn_convs = nn.ModuleList(
            [nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1) for _ in range(num_scales)]
        )

        # Classification and regression heads (shared across scales)
        self.cls_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(fpn_channels, num_anchors * num_classes, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(fpn_channels, num_anchors * 4, 1),
        )

    def forward(
        self, features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run FPN detection on encoder features.

        Expects features['intermediate'] to contain multi-scale feature maps.
        Falls back to reshaping patch_tokens if intermediate features unavailable.

        Returns:
            Dictionary with per-level predictions:
            - 'cls_preds': list of (B, num_anchors * num_classes, H_i, W_i)
            - 'reg_preds': list of (B, num_anchors * 4, H_i, W_i)
        """
        if "intermediate" in features:
            multi_scale = features["intermediate"]
        else:
            # Fallback: use patch tokens at a single scale
            patch_tokens = features["patch_tokens"]
            B, N, D = patch_tokens.shape
            h = w = int(math.sqrt(N))
            spatial = patch_tokens.permute(0, 2, 1).reshape(B, D, h, w)
            multi_scale = [spatial] * len(self.lateral_convs)

        # Build FPN pyramid (top-down pathway)
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, multi_scale)
        ]

        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fpn_features = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Apply detection heads at each scale
        cls_preds = [self.cls_head(feat) for feat in fpn_features]
        reg_preds = [self.reg_head(feat) for feat in fpn_features]

        return {"cls_preds": cls_preds, "reg_preds": reg_preds}


class DETRMultiScaleDecoder(BaseDecoder):
    """Multi-scale DETR decoder using feature pyramid fusion.

    Combines FPN-style multi-scale features with DETR's query-based detection.
    Uses intermediate encoder layers to build a feature pyramid, fuses them,
    and feeds to a DETR decoder.

    Args:
        encoder: Frozen encoder with intermediate layer support.
        num_classes: Number of object classes (excluding background).
        num_queries: Number of learnable object queries.
        num_decoder_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension for decoder.
        fpn_channels: Channels for FPN feature maps.
        dropout: Dropout rate.
        intermediate_layers: Which encoder layers to use (default: [3, 6, 9, 11]).
    """

    task = "detection"

    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        num_queries: int = 100,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        hidden_dim: int = 256,
        fpn_channels: int = 256,
        dropout: float = 0.1,
        intermediate_layers: list[int] | None = None,
    ) -> None:
        super().__init__(encoder, num_classes)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.fpn_channels = fpn_channels
        self._intermediate_layers = intermediate_layers or [3, 6, 9, 11]
        num_scales = len(self._intermediate_layers)

        # FPN lateral connections (project each scale to fpn_channels)
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(self._embed_dim, fpn_channels, 1) for _ in range(num_scales)]
        )

        # FPN top-down pathway
        self.fpn_convs = nn.ModuleList(
            [
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
                for _ in range(num_scales)
            ]
        )

        # Aggregate multi-scale features into single representation
        # We'll concatenate all scales (after resizing to same size) and project
        self.feature_aggregation = nn.Sequential(
            nn.Conv2d(fpn_channels * num_scales, hidden_dim, 1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(),
        )

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Positional encoding (will be computed dynamically based on feature size)
        self.pos_embed_scale = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Prediction heads
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # (cx, cy, w, h) normalized
        )

    def _build_fpn(self, multi_scale_features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Build feature pyramid from multi-scale encoder features."""
        # Apply lateral connections
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, multi_scale_features)
        ]

        # Top-down fusion (high-res to low-res)
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Apply FPN convolutions
        fpn_features = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return fpn_features

    def _aggregate_multiscale(
        self, fpn_features: list[torch.Tensor]
    ) -> torch.Tensor:
        """Aggregate multi-scale FPN features into single representation."""
        # Resize all features to the size of the finest scale (first one)
        target_size = fpn_features[0].shape[-2:]
        resized_features = []
        for feat in fpn_features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            resized_features.append(feat)

        # Concatenate along channel dimension
        concat_features = torch.cat(resized_features, dim=1)  # (B, C*num_scales, H, W)

        # Project to hidden_dim
        aggregated = self.feature_aggregation(concat_features)  # (B, D, H, W)
        return aggregated

    def forward(
        self, features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run multi-scale DETR detection.

        Returns:
            Dictionary with:
            - 'pred_logits': (B, num_queries, num_classes + 1)
            - 'pred_boxes': (B, num_queries, 4) -- normalized (cx, cy, w, h)
        """
        # Get multi-scale features from encoder
        if "intermediate" in features:
            multi_scale = features["intermediate"]
        else:
            # Fallback: use patch tokens at single scale
            patch_tokens = features["patch_tokens"]
            B, N, D = patch_tokens.shape
            h = w = int(math.sqrt(N))
            spatial = patch_tokens.permute(0, 2, 1).reshape(B, D, h, w)
            multi_scale = [spatial] * len(self.lateral_convs)

        # Build FPN
        fpn_features = self._build_fpn(multi_scale)

        # Aggregate multi-scale features
        aggregated = self._aggregate_multiscale(fpn_features)  # (B, D, H, W)

        B, D, H, W = aggregated.shape
        # Flatten spatial dimensions for transformer
        memory = aggregated.flatten(2).permute(0, 2, 1)  # (B, H*W, D)

        # Add positional encoding
        pos_embed = self.pos_embed_scale.expand(B, H * W, -1)
        memory = memory + pos_embed

        # Expand object queries for batch
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        # Run decoder layer by layer to collect intermediate outputs
        x = queries
        aux_outputs = []
        for i, layer in enumerate(self.transformer_decoder.layers):
            x = layer(x, memory)
            if i < len(self.transformer_decoder.layers) - 1:
                aux_outputs.append({
                    "pred_logits": self.class_head(x),
                    "pred_boxes": self.bbox_head(x).sigmoid(),
                })

        # Final layer predictions
        out = {
            "pred_logits": self.class_head(x),
            "pred_boxes": self.bbox_head(x).sigmoid(),
            "aux_outputs": aux_outputs,
        }
        return out
