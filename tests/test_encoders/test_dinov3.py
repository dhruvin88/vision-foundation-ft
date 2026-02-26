"""Tests for DINOv3 encoder."""

import pytest
import torch

from core.encoders import ALL_ENCODER_VARIANTS, DEFAULT_ENCODER, create_encoder
from core.encoders.dinov3 import DINOv3Encoder, DINOV3_VARIANTS


def test_all_variants_listed():
    """All DINOv3 variants should be in the variants dict."""
    assert "dinov3_vits16" in DINOV3_VARIANTS
    assert "dinov3_vitb16" in DINOV3_VARIANTS
    assert "dinov3_vitl16" in DINOV3_VARIANTS


def test_variant_configs():
    """Check embed dimensions for each variant."""
    assert DINOV3_VARIANTS["dinov3_vits16"]["embed_dim"] == 384
    assert DINOV3_VARIANTS["dinov3_vitb16"]["embed_dim"] == 768
    assert DINOV3_VARIANTS["dinov3_vitl16"]["embed_dim"] == 1024


def test_variant_patch_size():
    """All DINOv3 variants should have patch_size=16."""
    for name, config in DINOV3_VARIANTS.items():
        assert config["patch_size"] == 16, f"{name} should have patch_size=16"


def test_invalid_model_name():
    """Loading an invalid model name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        DINOv3Encoder("invalid_model")


def test_variant_num_patches():
    """Check num_patches calculation for default input_size=512."""
    # 512 / 16 = 32 -> 32*32 = 1024 patches
    for config in DINOV3_VARIANTS.values():
        expected = (512 // config["patch_size"]) ** 2
        assert expected == 1024


def test_default_intermediate_layers_vitb():
    """Test default_intermediate_layers for vitb16 (12 blocks)."""
    # Can't instantiate without HF model, but can check config
    config = DINOV3_VARIANTS["dinov3_vitb16"]
    n = config["num_blocks"]
    expected = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
    assert expected == [2, 5, 8, 11]


def test_default_intermediate_layers_vitl():
    """Test default_intermediate_layers for vitl16 (24 blocks)."""
    config = DINOV3_VARIANTS["dinov3_vitl16"]
    n = config["num_blocks"]
    expected = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
    assert expected == [5, 11, 17, 23]


def test_mock_encoder_v3(mock_encoder_v3):
    """Test the DINOv3 mock encoder fixture."""
    assert mock_encoder_v3.patch_size == 16
    assert mock_encoder_v3.input_size == 512
    assert mock_encoder_v3.num_patches == 1024
    assert mock_encoder_v3.embed_dim == 768
    assert mock_encoder_v3.grid_size == 32


def test_mock_encoder_v3_forward_features(mock_encoder_v3):
    """Test forward_features on mock DINOv3 encoder."""
    x = torch.randn(1, 3, 512, 512)
    features = mock_encoder_v3.forward_features(x)
    assert features["cls_token"].shape == (1, 768)
    assert features["patch_tokens"].shape == (1, 1024, 768)
    assert features["spatial_features"].shape == (1, 768, 32, 32)


def test_mock_features_v3(mock_features_v3):
    """Test the DINOv3 mock features fixture."""
    assert mock_features_v3["cls_token"].shape == (2, 768)
    assert mock_features_v3["patch_tokens"].shape == (2, 1024, 768)
    assert mock_features_v3["spatial_features"].shape == (2, 768, 32, 32)


# --- Factory tests ---

def test_default_encoder():
    """Default encoder should be dinov3_vitb16."""
    assert DEFAULT_ENCODER == "dinov3_vitb16"


def test_all_encoder_variants_includes_both():
    """ALL_ENCODER_VARIANTS should include both DINOv2 and DINOv3."""
    assert "dinov2_vitb14" in ALL_ENCODER_VARIANTS
    assert "dinov3_vitb16" in ALL_ENCODER_VARIANTS


def test_create_encoder_invalid():
    """Factory should raise ValueError for unknown encoder."""
    with pytest.raises(ValueError, match="Unknown encoder"):
        create_encoder("invalid_encoder")
