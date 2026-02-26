"""DINOv3 encoder loading and freezing.

Supports ViT-S/16, ViT-B/16, and ViT-L/16 variants.
All models are loaded from HuggingFace (facebook/dinov3-*).
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn

from core.encoders.base import BaseEncoder

logger = logging.getLogger(__name__)

DINOV3_VARIANTS = {
    "dinov3_vits16": {
        "embed_dim": 384,
        "patch_size": 16,
        "num_heads": 6,
        "num_blocks": 12,
        "hf_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
    },
    "dinov3_vitb16": {
        "embed_dim": 768,
        "patch_size": 16,
        "num_heads": 12,
        "num_blocks": 12,
        "hf_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    },
    "dinov3_vitl16": {
        "embed_dim": 1024,
        "patch_size": 16,
        "num_heads": 16,
        "num_blocks": 24,
        "hf_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    },
}

DINOv3ModelName = Literal[
    "dinov3_vits16",
    "dinov3_vitb16",
    "dinov3_vitl16",
]


class DINOv3Encoder(BaseEncoder):
    """DINOv3 vision transformer encoder.

    Loads a pretrained DINOv3 model from HuggingFace and freezes all parameters.
    Provides CLS token and patch-level features for downstream tasks.

    Args:
        model_name: DINOv3 variant to load.
        input_size: Expected input image size (must be divisible by patch_size).
        intermediate_layers: List of layer indices to extract for multi-scale features.
    """

    def __init__(
        self,
        model_name: DINOv3ModelName = "dinov3_vitb16",
        input_size: int = 512,
        intermediate_layers: list[int] | None = None,
    ) -> None:
        super().__init__()

        if model_name not in DINOV3_VARIANTS:
            raise ValueError(
                f"Unknown model: {model_name}. Choose from: {list(DINOV3_VARIANTS.keys())}"
            )

        self._model_name = model_name
        self._input_size = input_size
        self._config = DINOV3_VARIANTS[model_name]
        self._intermediate_layers = intermediate_layers
        self._num_blocks = self._config["num_blocks"]

        logger.info("Loading DINOv3 model: %s", model_name)
        from transformers import AutoModel

        self.model: nn.Module = AutoModel.from_pretrained(self._config["hf_name"])

        # Freeze immediately
        self.freeze()
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
        return self._intermediate_layers

    @intermediate_layers.setter
    def intermediate_layers(self, layers: list[int] | None) -> None:
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
        return self._input_size // self.patch_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features.

        Args:
            x: Input images of shape (B, 3, H, W).

        Returns:
            CLS token features of shape (B, embed_dim).
        """
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state[:, 0]

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract detailed features including CLS and patch tokens.

        Args:
            x: Input images of shape (B, 3, H, W).

        Returns:
            Dictionary with:
            - 'cls_token': (B, embed_dim)
            - 'patch_tokens': (B, num_patches, embed_dim)
            - 'spatial_features': (B, embed_dim, H', W')
            - 'intermediate' (optional): list of (B, embed_dim, H', W')
        """
        output_hidden_states = bool(self._intermediate_layers)
        outputs = self.model(pixel_values=x, output_hidden_states=output_hidden_states)

        last_hidden = outputs.last_hidden_state
        num_reg = getattr(self.model.config, "num_register_tokens", 4)

        cls_token = last_hidden[:, 0]
        # Skip CLS + register tokens to get patch tokens
        patch_tokens = last_hidden[:, 1 + num_reg:]

        B, N, D = patch_tokens.shape
        h = w = int(N**0.5)
        spatial = patch_tokens.permute(0, 2, 1).reshape(B, D, h, w)

        result = {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "spatial_features": spatial,
        }

        if self._intermediate_layers and outputs.hidden_states is not None:
            intermediates = []
            for layer_idx in self._intermediate_layers:
                # hidden_states[0] is the embedding output, so layer i output is at index i+1
                hidden = outputs.hidden_states[layer_idx + 1]
                layer_patches = hidden[:, 1 + num_reg:]
                B_l, N_l, D_l = layer_patches.shape
                h_l = w_l = int(N_l**0.5)
                intermediates.append(
                    layer_patches.permute(0, 2, 1).reshape(B_l, D_l, h_l, w_l)
                )
            result["intermediate"] = intermediates

        return result

    def get_transform(self) -> object:
        """Get the recommended preprocessing transform for this encoder."""
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
