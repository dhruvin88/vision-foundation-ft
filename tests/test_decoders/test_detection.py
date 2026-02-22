"""Tests for detection decoder heads."""

import pytest
import torch

from core.decoders.detection import DETRLiteDecoder, FPNHead
from core.decoders.rtdetr import RTDETRDecoder


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


class TestRTDETRDecoder:
    """Tests for the RT-DETR detection head."""

    def test_output_keys_inference(self, mock_encoder, mock_features_rtdetr):
        head = RTDETRDecoder(mock_encoder, num_classes=10)
        head.eval()
        out = head(mock_features_rtdetr)
        assert "pred_logits" in out
        assert "pred_boxes" in out
        assert "aux_outputs" in out
        assert "enc_outputs" in out
        assert "cdn_outputs" not in out

    def test_output_shapes(self, mock_encoder, mock_features_rtdetr):
        num_classes = 10
        head = RTDETRDecoder(mock_encoder, num_classes=num_classes)
        head.eval()
        out = head(mock_features_rtdetr)
        assert out["pred_logits"].shape == (2, 100, num_classes)
        assert out["pred_boxes"].shape == (2, 100, 4)

    def test_num_queries_formula(self, mock_encoder):
        head1 = RTDETRDecoder(mock_encoder, num_classes=1)
        assert head1.num_queries == 30

        head40 = RTDETRDecoder(mock_encoder, num_classes=40)
        assert head40.num_queries == 300

    def test_boxes_normalized(self, mock_encoder, mock_features_rtdetr):
        head = RTDETRDecoder(mock_encoder, num_classes=5)
        head.eval()
        out = head(mock_features_rtdetr)
        assert out["pred_boxes"].min() >= 0.0
        assert out["pred_boxes"].max() <= 1.0

    def test_aux_outputs_count(self, mock_encoder, mock_features_rtdetr):
        head = RTDETRDecoder(mock_encoder, num_classes=5, num_decoder_layers=4)
        head.eval()
        out = head(mock_features_rtdetr)
        assert len(out["aux_outputs"]) == 4

    def test_enc_outputs_shape(self, mock_encoder, mock_features_rtdetr):
        num_classes = 7
        head = RTDETRDecoder(mock_encoder, num_classes=num_classes)
        head.eval()
        out = head(mock_features_rtdetr)
        enc_logits = out["enc_outputs"]["pred_logits"]
        B = 2
        # N = 37*37 + 18*18 + 9*9 = 1369 + 324 + 81 = 1774
        expected_n = 37 * 37 + (37 // 2) ** 2 + (37 // 4) ** 2
        assert enc_logits.shape == (B, expected_n, num_classes)

    def test_cdn_present_in_training(self, mock_encoder, mock_features_rtdetr):
        head = RTDETRDecoder(mock_encoder, num_classes=10)
        head.train()
        features = dict(mock_features_rtdetr)
        features["gt_labels"] = torch.tensor([[0, 1, -1], [2, -1, -1]])
        features["gt_boxes"] = torch.tensor([
            [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]],
            [[0.3, 0.3, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ])
        out = head(features)
        assert "cdn_outputs" in out
        assert "cdn_meta" in out

    def test_cdn_absent_in_eval(self, mock_encoder, mock_features_rtdetr):
        head = RTDETRDecoder(mock_encoder, num_classes=10)
        head.eval()
        features = dict(mock_features_rtdetr)
        features["gt_labels"] = torch.tensor([[0, 1, -1], [2, -1, -1]])
        features["gt_boxes"] = torch.zeros(2, 3, 4)
        out = head(features)
        assert "cdn_outputs" not in out

    def test_task_attribute(self, mock_encoder):
        head = RTDETRDecoder(mock_encoder, num_classes=5)
        assert head.task == "detection"

    def test_trainable_params_positive(self, mock_encoder):
        head = RTDETRDecoder(mock_encoder, num_classes=10)
        assert head.num_trainable_params() > 0

    def test_intermediate_layers_set_on_encoder(self, mock_encoder):
        RTDETRDecoder(mock_encoder, num_classes=5)
        # ViT-B has 12 blocks: [12//4-1, 12//2-1, 3*12//4-1] = [2, 5, 8]
        assert mock_encoder.intermediate_layers == [2, 5, 8]

    def test_no_image_fallback(self, mock_encoder, mock_features_rtdetr):
        head = RTDETRDecoder(mock_encoder, num_classes=5)
        head.eval()
        features = {k: v for k, v in mock_features_rtdetr.items() if k != "image"}
        out = head(features)
        assert "pred_logits" in out
        assert "pred_boxes" in out


class TestRTDETRLoss:
    """Tests for RTDETRDecoder training loss integration."""

    def test_loss_finite_and_nonneg(self, mock_encoder):
        from core.training.trainer import DecoderLightningModule

        head = RTDETRDecoder(mock_encoder, num_classes=5)
        module = DecoderLightningModule(head, lr=1e-3)

        batch = {
            "image": torch.randn(2, 3, 518, 518),
            "labels": torch.tensor([[0, 1, -1], [2, -1, -1]]),
            "boxes": torch.tensor([
                [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]],
                [[0.3, 0.3, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ]),
        }
        loss = module.training_step(batch, 0)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_sdk_default_is_rtdetr(self, mock_encoder):
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        import sdk as fft
        head = fft.DetectionHead(mock_encoder, num_classes=5)
        assert isinstance(head, RTDETRDecoder)

    def test_detr_lite_still_works_via_sdk(self, mock_encoder, mock_features):
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        import sdk as fft
        head = fft.DetectionHead(mock_encoder, num_classes=5, head_type="detr_lite")
        head.eval()
        out = head(mock_features)
        assert "pred_logits" in out
        assert "pred_boxes" in out
