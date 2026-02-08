"""Tests for the Trainer and DecoderLightningModule."""

import pytest
import torch

from core.data.dataset import FFTDataset
from core.decoders.classification import LinearProbe
from core.training.trainer import DecoderLightningModule


class TestDatasetSplit:
    """Tests for FFTDataset.split()."""

    def test_split_ratios(self, tmp_dataset_dir):
        dataset = FFTDataset.from_folder(tmp_dataset_dir, task="classification")
        total = len(dataset)
        train_ds, val_ds = dataset.split(val_ratio=0.2, seed=42)
        assert len(train_ds) + len(val_ds) == total
        assert len(val_ds) == int(total * 0.2)

    def test_split_preserves_task(self, tmp_dataset_dir):
        dataset = FFTDataset.from_folder(tmp_dataset_dir, task="classification")
        train_ds, val_ds = dataset.split(val_ratio=0.2)
        assert train_ds.task == "classification"
        assert val_ds.task == "classification"

    def test_split_preserves_class_names(self, tmp_dataset_dir):
        dataset = FFTDataset.from_folder(tmp_dataset_dir, task="classification")
        train_ds, val_ds = dataset.split(val_ratio=0.2)
        assert train_ds.class_names == dataset.class_names
        assert val_ds.class_names == dataset.class_names

    def test_split_deterministic(self, tmp_dataset_dir):
        dataset = FFTDataset.from_folder(tmp_dataset_dir, task="classification")
        train1, val1 = dataset.split(val_ratio=0.2, seed=42)
        train2, val2 = dataset.split(val_ratio=0.2, seed=42)
        assert [s["image_path"] for s in train1.samples] == [
            s["image_path"] for s in train2.samples
        ]

    def test_split_different_seeds(self, tmp_dataset_dir):
        dataset = FFTDataset.from_folder(tmp_dataset_dir, task="classification")
        train1, _ = dataset.split(val_ratio=0.2, seed=42)
        train2, _ = dataset.split(val_ratio=0.2, seed=99)
        paths1 = [s["image_path"] for s in train1.samples]
        paths2 = [s["image_path"] for s in train2.samples]
        # Different seeds should (almost certainly) produce different orders
        assert paths1 != paths2


class TestDatasetLoading:
    """Tests for FFTDataset loading."""

    def test_from_folder_classification(self, tmp_dataset_dir):
        dataset = FFTDataset.from_folder(tmp_dataset_dir, task="classification")
        assert len(dataset) == 10  # 5 cats + 5 dogs
        assert dataset.task == "classification"
        assert set(dataset.class_names) == {"cats", "dogs"}

    def test_from_folder_segmentation(self, tmp_segmentation_dir):
        dataset = FFTDataset.from_folder(tmp_segmentation_dir, task="segmentation")
        assert len(dataset) == 5
        assert dataset.task == "segmentation"

    def test_get_stats(self, tmp_dataset_dir):
        dataset = FFTDataset.from_folder(tmp_dataset_dir, task="classification")
        stats = dataset.get_stats()
        assert stats["num_samples"] == 10
        assert stats["task"] == "classification"
        assert stats["num_classes"] == 2

    def test_getitem_classification(self, tmp_dataset_dir):
        dataset = FFTDataset.from_folder(tmp_dataset_dir, task="classification")
        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert isinstance(sample["label"], int)

    def test_invalid_task_raises(self, tmp_dataset_dir):
        with pytest.raises(ValueError, match="Unknown task"):
            FFTDataset.from_folder(tmp_dataset_dir, task="invalid")


class TestDecoderLightningModule:
    """Tests for the DecoderLightningModule."""

    def test_configure_optimizers(self, mock_encoder):
        decoder = LinearProbe(mock_encoder, num_classes=5)
        module = DecoderLightningModule(decoder, lr=1e-3)
        opt_config = module.configure_optimizers()
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config

    def test_encoder_frozen(self, mock_encoder):
        decoder = LinearProbe(mock_encoder, num_classes=5)
        module = DecoderLightningModule(decoder)
        for param in module.decoder.encoder.parameters():
            assert not param.requires_grad

    def test_task_attribute(self, mock_encoder):
        decoder = LinearProbe(mock_encoder, num_classes=5)
        module = DecoderLightningModule(decoder)
        assert module.task == "classification"

    def test_forward(self, mock_encoder, mock_features):
        decoder = LinearProbe(mock_encoder, num_classes=5)
        module = DecoderLightningModule(decoder)
        out = module(mock_features)
        assert out.shape == (2, 5)
