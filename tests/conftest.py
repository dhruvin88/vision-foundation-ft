"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from core.encoders.base import BaseEncoder


class MockEncoder(BaseEncoder):
    """A lightweight mock encoder for testing decoders without loading DINOv2."""

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 14,
        input_size: int = 518,
        intermediate_layers: list[int] | None = None,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._patch_size = patch_size
        self._input_size = input_size
        self._num_patches = (input_size // patch_size) ** 2
        self._intermediate_layers = intermediate_layers
        self._num_blocks = 12  # Mock ViT-B
        # Minimal parameters so state_dict isn't empty
        self.proj = nn.Linear(3, embed_dim)
        self.freeze()

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def intermediate_layers(self) -> list[int] | None:
        return self._intermediate_layers

    @intermediate_layers.setter
    def intermediate_layers(self, layers: list[int] | None) -> None:
        self._intermediate_layers = layers

    def default_intermediate_layers(self) -> list[int]:
        n = self._num_blocks
        return [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]

    @property
    def grid_size(self) -> int:
        return self._input_size // self._patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.randn(B, self._embed_dim)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B = x.shape[0]
        N = self._num_patches
        h = w = int(N**0.5)
        patch_tokens = torch.randn(B, N, self._embed_dim)
        result = {
            "cls_token": torch.randn(B, self._embed_dim),
            "patch_tokens": patch_tokens,
            "spatial_features": patch_tokens.permute(0, 2, 1).reshape(
                B, self._embed_dim, h, w
            ),
        }
        if self._intermediate_layers:
            result["intermediate"] = [
                torch.randn(B, self._embed_dim, h, w)
                for _ in self._intermediate_layers
            ]
        return result

    def get_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((self._input_size, self._input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


@pytest.fixture
def mock_encoder():
    """Create a mock encoder for testing."""
    return MockEncoder()


@pytest.fixture
def mock_encoder_v3():
    """Create a mock encoder with DINOv3 dimensions (patch_size=16, input_size=512)."""
    return MockEncoder(embed_dim=768, patch_size=16, input_size=512)


@pytest.fixture
def mock_features_v3():
    """Create mock encoder features with DINOv3 dimensions."""
    B, N, D = 2, 1024, 768  # 32*32 patches for 512/16
    h = w = 32
    patch_tokens = torch.randn(B, N, D)
    return {
        "cls_token": torch.randn(B, D),
        "patch_tokens": patch_tokens,
        "spatial_features": patch_tokens.permute(0, 2, 1).reshape(B, D, h, w),
    }


@pytest.fixture
def mock_features():
    """Create mock encoder features for testing decoders."""
    B, N, D = 2, 1369, 768  # 37*37 patches for 518/14
    h = w = 37
    patch_tokens = torch.randn(B, N, D)
    return {
        "cls_token": torch.randn(B, D),
        "patch_tokens": patch_tokens,
        "spatial_features": patch_tokens.permute(0, 2, 1).reshape(B, D, h, w),
    }


@pytest.fixture
def mock_features_rtdetr(mock_encoder):
    """Create mock encoder features for RTDETRDecoder (includes intermediate + image)."""
    B, D, h, w = 2, 768, 37, 37
    patch_tokens = torch.randn(B, h * w, D)
    return {
        "cls_token": torch.randn(B, D),
        "patch_tokens": patch_tokens,
        "spatial_features": patch_tokens.permute(0, 2, 1).reshape(B, D, h, w),
        "intermediate": [torch.randn(B, D, h, w) for _ in range(3)],
        "image": torch.randn(B, 3, 518, 518),
    }


@pytest.fixture
def mock_multiscale_features():
    """Create mock encoder features including intermediate layers for FPN/UPerNet."""
    B, N, D = 2, 1369, 768  # 37*37 patches for 518/14
    h = w = 37
    patch_tokens = torch.randn(B, N, D)
    return {
        "cls_token": torch.randn(B, D),
        "patch_tokens": patch_tokens,
        "spatial_features": patch_tokens.permute(0, 2, 1).reshape(B, D, h, w),
        "intermediate": [torch.randn(B, D, h, w) for _ in range(4)],
    }


@pytest.fixture
def tmp_dataset_dir():
    """Create a temporary directory with a simple classification dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create class folders with dummy images
        for class_name in ["cats", "dogs"]:
            class_dir = tmpdir / class_name
            class_dir.mkdir()
            for i in range(5):
                # Create small random images
                from PIL import Image
                import numpy as np

                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(class_dir / f"img_{i}.jpg")

        yield tmpdir


@pytest.fixture
def tmp_segmentation_dir():
    """Create a temporary directory with a segmentation dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        images_dir = tmpdir / "images"
        masks_dir = tmpdir / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()

        from PIL import Image
        import numpy as np

        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(images_dir / f"img_{i}.jpg")

            mask = Image.fromarray(
                np.random.randint(0, 3, (64, 64), dtype=np.uint8)
            )
            mask.save(masks_dir / f"img_{i}.png")

        yield tmpdir
