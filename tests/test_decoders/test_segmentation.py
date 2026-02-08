"""Tests for segmentation decoder heads."""

import pytest
import torch

from core.decoders.segmentation import LinearSegHead, UPerNetHead, MaskTransformerHead


class TestLinearSegHead:
    """Tests for the LinearSegHead segmentation head."""

    def test_output_shape(self, mock_encoder, mock_features):
        head = LinearSegHead(mock_encoder, num_classes=5)
        out = head(mock_features)
        assert out.shape == (2, 5, 518, 518)

    def test_custom_output_size(self, mock_encoder, mock_features):
        head = LinearSegHead(mock_encoder, num_classes=5, output_size=256)
        out = head(mock_features)
        assert out.shape == (2, 5, 256, 256)

    def test_task_attribute(self, mock_encoder):
        head = LinearSegHead(mock_encoder, num_classes=3)
        assert head.task == "segmentation"

    def test_trainable_params(self, mock_encoder):
        head = LinearSegHead(mock_encoder, num_classes=5)
        # Conv2d(768, 5, 1): 768*5 + 5 = 3845
        assert head.num_trainable_params() == 768 * 5 + 5

    def test_num_classes(self, mock_encoder):
        head = LinearSegHead(mock_encoder, num_classes=21)
        assert head.num_classes == 21


class TestUPerNetHead:
    """Tests for the UPerNet segmentation head."""

    def test_output_shape(self, mock_encoder, mock_features):
        head = UPerNetHead(mock_encoder, num_classes=5)
        out = head(mock_features)
        assert out.shape == (2, 5, 518, 518)

    def test_custom_output_size(self, mock_encoder, mock_features):
        head = UPerNetHead(mock_encoder, num_classes=5, output_size=256)
        out = head(mock_features)
        assert out.shape == (2, 5, 256, 256)

    def test_task_attribute(self, mock_encoder):
        head = UPerNetHead(mock_encoder, num_classes=3)
        assert head.task == "segmentation"

    def test_trainable_params_greater_than_linear(self, mock_encoder):
        linear = LinearSegHead(mock_encoder, num_classes=5)
        upernet = UPerNetHead(mock_encoder, num_classes=5)
        assert upernet.num_trainable_params() > linear.num_trainable_params()

    def test_custom_fpn_channels(self, mock_encoder, mock_features):
        head = UPerNetHead(mock_encoder, num_classes=5, fpn_channels=128)
        out = head(mock_features)
        assert out.shape == (2, 5, 518, 518)

    def test_with_intermediate_features(self, mock_encoder, mock_multiscale_features):
        """UPerNet should use real intermediate features when available."""
        head = UPerNetHead(mock_encoder, num_classes=5)
        out = head(mock_multiscale_features)
        assert out.shape == (2, 5, 518, 518)


class TestMaskTransformerHead:
    """Tests for the MaskTransformer segmentation head."""

    def test_output_shape(self, mock_encoder, mock_features):
        head = MaskTransformerHead(mock_encoder, num_classes=5)
        out = head(mock_features)
        assert out.shape == (2, 5, 518, 518)

    def test_custom_output_size(self, mock_encoder, mock_features):
        head = MaskTransformerHead(mock_encoder, num_classes=5, output_size=256)
        out = head(mock_features)
        assert out.shape == (2, 5, 256, 256)

    def test_task_attribute(self, mock_encoder):
        head = MaskTransformerHead(mock_encoder, num_classes=3)
        assert head.task == "segmentation"

    def test_class_queries_learnable(self, mock_encoder):
        head = MaskTransformerHead(mock_encoder, num_classes=5)
        assert head.class_queries.requires_grad is True
        assert head.class_queries.shape == (1, 5, 256)

    def test_trainable_params_positive(self, mock_encoder):
        head = MaskTransformerHead(mock_encoder, num_classes=5)
        assert head.num_trainable_params() > 0

    def test_custom_hidden_dim(self, mock_encoder, mock_features):
        head = MaskTransformerHead(mock_encoder, num_classes=5, hidden_dim=128)
        out = head(mock_features)
        assert out.shape == (2, 5, 518, 518)
