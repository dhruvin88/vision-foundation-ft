"""Tests for DINOv2 encoder."""

import pytest
import torch

from core.encoders.dinov2 import DINOv2Encoder, ALL_VARIANTS, DINOV2_VARIANTS


def test_all_variants_listed():
    """All DINOv2 variants should be in the variants dict."""
    assert "dinov2_vits14" in ALL_VARIANTS
    assert "dinov2_vitb14" in ALL_VARIANTS
    assert "dinov2_vitl14" in ALL_VARIANTS
    assert "dinov2_vitg14" in ALL_VARIANTS


def test_variant_configs():
    """Check embed dimensions for each variant."""
    assert DINOV2_VARIANTS["dinov2_vits14"]["embed_dim"] == 384
    assert DINOV2_VARIANTS["dinov2_vitb14"]["embed_dim"] == 768
    assert DINOV2_VARIANTS["dinov2_vitl14"]["embed_dim"] == 1024
    assert DINOV2_VARIANTS["dinov2_vitg14"]["embed_dim"] == 1536


def test_invalid_model_name():
    """Loading an invalid model name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        DINOv2Encoder("invalid_model")


def test_mock_encoder_intermediate_layers(mock_encoder):
    """Test intermediate_layers setter on MockEncoder."""
    assert mock_encoder.intermediate_layers is None

    # Enable intermediate layers
    mock_encoder.intermediate_layers = [2, 5, 8, 11]
    assert mock_encoder.intermediate_layers == [2, 5, 8, 11]

    # forward_features should now include 'intermediate'
    x = torch.randn(1, 3, 518, 518)
    features = mock_encoder.forward_features(x)
    assert "intermediate" in features
    assert len(features["intermediate"]) == 4

    # Disable
    mock_encoder.intermediate_layers = None
    features = mock_encoder.forward_features(x)
    assert "intermediate" not in features


def test_default_intermediate_layers(mock_encoder):
    """Test default_intermediate_layers returns 4 evenly-spaced indices."""
    defaults = mock_encoder.default_intermediate_layers()
    assert len(defaults) == 4
    # MockEncoder has 12 blocks: should be [2, 5, 8, 11]
    assert defaults == [2, 5, 8, 11]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Skipping model loading test (no GPU / slow download)",
)
def test_encoder_loading():
    """Test loading the smallest DINOv2 variant."""
    encoder = DINOv2Encoder("dinov2_vits14")
    assert encoder.embed_dim == 384
    assert encoder.patch_size == 14
    assert encoder._frozen is True

    # All parameters should be frozen
    for param in encoder.parameters():
        assert not param.requires_grad
