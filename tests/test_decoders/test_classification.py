"""Tests for classification decoder heads."""

import pytest
import torch

from core.decoders.classification import LinearProbe, MLPHead, TransformerHead


class TestLinearProbe:
    """Tests for the LinearProbe classification head."""

    def test_output_shape(self, mock_encoder, mock_features):
        head = LinearProbe(mock_encoder, num_classes=10)
        out = head(mock_features)
        assert out.shape == (2, 10)

    def test_task_attribute(self, mock_encoder):
        head = LinearProbe(mock_encoder, num_classes=5)
        assert head.task == "classification"

    def test_trainable_params(self, mock_encoder):
        head = LinearProbe(mock_encoder, num_classes=10)
        # Linear layer: 768 * 10 + 10 = 7690
        assert head.num_trainable_params() == 768 * 10 + 10

    def test_num_classes_stored(self, mock_encoder):
        head = LinearProbe(mock_encoder, num_classes=7)
        assert head.num_classes == 7

    def test_predict(self, mock_encoder):
        head = LinearProbe(mock_encoder, num_classes=5)
        images = torch.randn(2, 3, 518, 518)
        out = head.predict(images)
        assert out.shape == (2, 5)


class TestMLPHead:
    """Tests for the MLPHead classification head."""

    def test_output_shape(self, mock_encoder, mock_features):
        head = MLPHead(mock_encoder, num_classes=10)
        out = head(mock_features)
        assert out.shape == (2, 10)

    def test_task_attribute(self, mock_encoder):
        head = MLPHead(mock_encoder, num_classes=5)
        assert head.task == "classification"

    def test_custom_hidden_dim(self, mock_encoder, mock_features):
        head = MLPHead(mock_encoder, num_classes=10, hidden_dim=256)
        out = head(mock_features)
        assert out.shape == (2, 10)

    def test_trainable_params_greater_than_linear(self, mock_encoder):
        linear = LinearProbe(mock_encoder, num_classes=10)
        mlp = MLPHead(mock_encoder, num_classes=10)
        assert mlp.num_trainable_params() > linear.num_trainable_params()

    def test_predict(self, mock_encoder):
        head = MLPHead(mock_encoder, num_classes=5)
        images = torch.randn(2, 3, 518, 518)
        out = head.predict(images)
        assert out.shape == (2, 5)


class TestTransformerHead:
    """Tests for the TransformerHead classification head."""

    def test_output_shape(self, mock_encoder, mock_features):
        head = TransformerHead(mock_encoder, num_classes=10)
        out = head(mock_features)
        assert out.shape == (2, 10)

    def test_task_attribute(self, mock_encoder):
        head = TransformerHead(mock_encoder, num_classes=5)
        assert head.task == "classification"

    def test_custom_layers_and_heads(self, mock_encoder, mock_features):
        head = TransformerHead(
            mock_encoder, num_classes=10, num_layers=4, num_heads=4
        )
        out = head(mock_features)
        assert out.shape == (2, 10)

    def test_trainable_params_greater_than_mlp(self, mock_encoder):
        mlp = MLPHead(mock_encoder, num_classes=10)
        transformer = TransformerHead(mock_encoder, num_classes=10)
        assert transformer.num_trainable_params() > mlp.num_trainable_params()

    def test_predict(self, mock_encoder):
        head = TransformerHead(mock_encoder, num_classes=5)
        images = torch.randn(2, 3, 518, 518)
        out = head.predict(images)
        assert out.shape == (2, 5)

    def test_cls_query_is_learnable(self, mock_encoder):
        head = TransformerHead(mock_encoder, num_classes=5)
        assert head.cls_query.requires_grad is True
