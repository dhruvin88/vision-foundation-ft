"""DINOv2 encoder loading and freezing.

Supports ViT-S/14, ViT-B/14, ViT-L/14, and ViT-G/14 variants.
All models are loaded from torch.hub (facebookresearch/dinov2).
"""

from __future__ import annotations

import contextlib
import logging
from typing import Literal

import torch
import torch.nn as nn

from core.encoders.base import BaseEncoder

logger = logging.getLogger(__name__)

DINOV2_VARIANTS = {
    "dinov2_vits14": {"embed_dim": 384, "patch_size": 14, "num_heads": 6},
    "dinov2_vitb14": {"embed_dim": 768, "patch_size": 14, "num_heads": 12},
    "dinov2_vitl14": {"embed_dim": 1024, "patch_size": 14, "num_heads": 16},
    "dinov2_vitg14": {"embed_dim": 1536, "patch_size": 14, "num_heads": 24},
}

# Variants with registers (improved attention maps)
DINOV2_REG_VARIANTS = {
    "dinov2_vits14_reg": {"embed_dim": 384, "patch_size": 14, "num_heads": 6},
    "dinov2_vitb14_reg": {"embed_dim": 768, "patch_size": 14, "num_heads": 12},
    "dinov2_vitl14_reg": {"embed_dim": 1024, "patch_size": 14, "num_heads": 16},
    "dinov2_vitg14_reg": {"embed_dim": 1536, "patch_size": 14, "num_heads": 24},
}

ALL_VARIANTS = {**DINOV2_VARIANTS, **DINOV2_REG_VARIANTS}

DINOv2ModelName = Literal[
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
    "dinov2_vitl14_reg",
    "dinov2_vitg14_reg",
]


class DINOv2Encoder(BaseEncoder):
    """DINOv2 vision transformer encoder.

    Loads a pretrained DINOv2 model and freezes all parameters.
    Provides CLS token and patch-level features for downstream tasks.

    Args:
        model_name: DINOv2 variant to load.
        input_size: Expected input image size (must be divisible by patch_size).
        intermediate_layers: List of layer indices to extract for multi-scale features.
    """

    def __init__(
        self,
        model_name: DINOv2ModelName = "dinov2_vitb14",
        input_size: int = 518,
        intermediate_layers: list[int] | None = None,
    ) -> None:
        super().__init__()

        if model_name not in ALL_VARIANTS:
            raise ValueError(
                f"Unknown model: {model_name}. Choose from: {list(ALL_VARIANTS.keys())}"
            )

        self._model_name = model_name
        self._input_size = input_size
        self._config = ALL_VARIANTS[model_name]
        self._intermediate_layers = intermediate_layers

        # Number of transformer blocks (needed for default intermediate layer indices)
        # ViT-S=12, ViT-B=12, ViT-L=24, ViT-G=40
        self._num_blocks = {"vits": 12, "vitb": 12, "vitl": 24, "vitg": 40}[
            model_name.split("_")[1].replace("14", "").replace("_reg", "")
        ]

        logger.info("Loading DINOv2 model: %s", model_name)
        self.model: nn.Module = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True
        )

        # Freeze immediately
        self.freeze()
        self._lora_enabled = False
        logger.info(
            "Loaded and froze %s (embed_dim=%d, patch_size=%d)",
            model_name,
            self.embed_dim,
            self.patch_size,
        )

    @property
    def embed_dim(self) -> int:
        return self._config["embed_dim"]

    @property
    def patch_size(self) -> int:
        return self._config["patch_size"]

    @property
    def num_patches(self) -> int:
        return (self._input_size // self.patch_size) ** 2

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def num_heads(self) -> int:
        return self._config["num_heads"]

    @property
    def intermediate_layers(self) -> list[int] | None:
        """Currently configured intermediate layer indices."""
        return self._intermediate_layers

    @intermediate_layers.setter
    def intermediate_layers(self, layers: list[int] | None) -> None:
        """Set intermediate layers to extract (can be configured after construction)."""
        if layers is not None:
            for idx in layers:
                if idx < 0 or idx >= self._num_blocks:
                    raise ValueError(
                        f"Layer index {idx} out of range for {self._model_name} "
                        f"(has {self._num_blocks} blocks, valid: 0-{self._num_blocks - 1})"
                    )
        self._intermediate_layers = layers

    def default_intermediate_layers(self) -> list[int]:
        """Return sensible default intermediate layer indices for multi-scale extraction.

        Picks 4 evenly-spaced layers from the transformer blocks.
        """
        n = self._num_blocks
        return [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]

    @property
    def grid_size(self) -> int:
        """Spatial grid size (height = width) of the patch tokens."""
        return self._input_size // self.patch_size

    @property
    def _fwd_ctx(self):
        """Return nullcontext when LoRA is active (grads needed), else no_grad."""
        return contextlib.nullcontext() if self._lora_enabled else torch.no_grad()

    def enable_lora(
        self,
        rank: int = 4,
        alpha: float = 4.0,
        target_modules: list[str] | None = None,
    ) -> int:
        """Inject LoRA adapters into attention layers and unfreeze them.

        Args:
            rank: Low-rank dimension.
            alpha: Scaling factor (scale = alpha / rank).
            target_modules: List of layer name suffixes to patch.
                Defaults to ["attn.qkv", "attn.proj"].

        Returns:
            Number of layers patched.
        """
        from core.encoders.lora import apply_lora

        if target_modules is None:
            target_modules = ["attn.qkv", "attn.proj"]
        n = apply_lora(self.model, rank, alpha, target_modules)
        self._lora_enabled = True
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        return n

    def lora_parameters(self):
        """Return list of trainable LoRA parameters."""
        return [p for n, p in self.model.named_parameters() if "lora_" in n]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features.

        Args:
            x: Input images of shape (B, 3, H, W). H and W should be
               divisible by patch_size (default 518).

        Returns:
            CLS token features of shape (B, embed_dim).
        """
        with self._fwd_ctx:
            return self.model(x)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract detailed features including CLS and patch tokens.

        Args:
            x: Input images of shape (B, 3, H, W).

        Returns:
            Dictionary with:
            - 'cls_token': (B, embed_dim) -- global image representation
            - 'patch_tokens': (B, num_patches, embed_dim) -- per-patch features
            - 'spatial_features': (B, embed_dim, H', W') -- reshaped patch features
            - 'intermediate' (optional): list of (B, num_patches, embed_dim) from
              intermediate layers if intermediate_layers was specified
        """
        with self._fwd_ctx:
            features = self.model.forward_features(x)

            # DINOv2 returns dict with 'x_norm_clstoken' and 'x_norm_patchtokens'
            cls_token = features["x_norm_clstoken"]  # (B, embed_dim)
            patch_tokens = features["x_norm_patchtokens"]  # (B, N, embed_dim)

            B, N, D = patch_tokens.shape
            h = w = int(N**0.5)
            spatial = patch_tokens.permute(0, 2, 1).reshape(B, D, h, w)

            result = {
                "cls_token": cls_token,
                "patch_tokens": patch_tokens,
                "spatial_features": spatial,
            }

            # Extract intermediate layer features if requested
            if self._intermediate_layers:
                intermediates = self.model.get_intermediate_layers(
                    x, n=self._intermediate_layers, reshape=True
                )
                result["intermediate"] = list(intermediates)

        return result

    def get_transform(self) -> object:
        """Get the recommended preprocessing transform for this encoder.

        Returns a torchvision transform that resizes and normalizes images
        according to DINOv2 requirements.
        """
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Resize(
                    (self._input_size, self._input_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
