"""Tests for detection decoder heads."""

import pytest
import torch

from core.decoders.detection import DETRLiteDecoder, FPNHead


class TestDETRLiteDecoder:
    """Tests for the DETRLite detection head."""

    def test_output_keys(self, mock_encoder, mock_features):
        head = DETRLiteDecoder(mock_encoder, num_classes=10)
        out = head(mock_features)
        assert "pred_logits" in out
        assert "pred_boxes" in out

    def test_output_shapes(self, mock_encoder, mock_features):
        num_classes = 10
        num_queries = 100
        head = DETRLiteDecoder(mock_encoder, num_classes=num_classes, num_queries=num_queries)
        out = head(mock_features)
        assert out["pred_logits"].shape == (2, num_queries, num_classes + 1)
        assert out["pred_boxes"].shape == (2, num_queries, 4)

    def test_custom_num_queries(self, mock_encoder, mock_features):
        head = DETRLiteDecoder(mock_encoder, num_classes=5, num_queries=50)
        out = head(mock_features)
        assert out["pred_logits"].shape == (2, 50, 6)
        assert out["pred_boxes"].shape == (2, 50, 4)

    def test_task_attribute(self, mock_encoder):
        head = DETRLiteDecoder(mock_encoder, num_classes=5)
        assert head.task == "detection"

    def test_boxes_normalized(self, mock_encoder, mock_features):
        head = DETRLiteDecoder(mock_encoder, num_classes=5)
        out = head(mock_features)
        # Boxes go through sigmoid, so values should be in [0, 1]
        assert out["pred_boxes"].min() >= 0.0
        assert out["pred_boxes"].max() <= 1.0

    def test_trainable_params_positive(self, mock_encoder):
        head = DETRLiteDecoder(mock_encoder, num_classes=10)
        assert head.num_trainable_params() > 0


class TestFPNHead:
    """Tests for the FPN detection head."""

    def test_output_keys(self, mock_encoder, mock_features):
        head = FPNHead(mock_encoder, num_classes=10)
        out = head(mock_features)
        assert "cls_preds" in out
        assert "reg_preds" in out

    def test_output_is_multi_scale(self, mock_encoder, mock_features):
        head = FPNHead(mock_encoder, num_classes=10)
        out = head(mock_features)
        # Default 4 intermediate layers -> 4 scales
        assert len(out["cls_preds"]) == 4
        assert len(out["reg_preds"]) == 4

    def test_cls_pred_channels(self, mock_encoder, mock_features):
        num_classes = 10
        num_anchors = 9
        head = FPNHead(mock_encoder, num_classes=num_classes, num_anchors=num_anchors)
        out = head(mock_features)
        # Each cls pred should have num_anchors * num_classes channels
        for cls_pred in out["cls_preds"]:
            assert cls_pred.shape[1] == num_anchors * num_classes

    def test_reg_pred_channels(self, mock_encoder, mock_features):
        num_anchors = 9
        head = FPNHead(mock_encoder, num_classes=5, num_anchors=num_anchors)
        out = head(mock_features)
        # Each reg pred should have num_anchors * 4 channels
        for reg_pred in out["reg_preds"]:
            assert reg_pred.shape[1] == num_anchors * 4

    def test_task_attribute(self, mock_encoder):
        head = FPNHead(mock_encoder, num_classes=5)
        assert head.task == "detection"

    def test_custom_fpn_channels(self, mock_encoder, mock_features):
        head = FPNHead(mock_encoder, num_classes=5, fpn_channels=128)
        out = head(mock_features)
        assert "cls_preds" in out
        assert len(out["cls_preds"]) == 4

    def test_with_intermediate_features(self, mock_encoder, mock_multiscale_features):
        """FPN should use real intermediate features when available."""
        head = FPNHead(mock_encoder, num_classes=10)
        out = head(mock_multiscale_features)
        assert "cls_preds" in out
        assert len(out["cls_preds"]) == 4

    def test_trainable_params_positive(self, mock_encoder):
        head = FPNHead(mock_encoder, num_classes=10)
        assert head.num_trainable_params() > 0
