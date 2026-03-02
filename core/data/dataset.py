"""Unified dataset class supporting classification, detection, and segmentation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

TaskType = Literal["classification", "detection", "segmentation"]


class FFTDataset(Dataset):
    """Unified dataset for classification, detection, and segmentation.

    Supports loading from:
    - Folder structure (classification: one subfolder per class)
    - COCO JSON annotations
    - Internal format (list of dicts)

    Args:
        samples: List of sample dictionaries.
        task: One of 'classification', 'detection', 'segmentation'.
        transform: Optional image transform (applied before returning).
        target_transform: Optional target transform.
    """

    def __init__(
        self,
        samples: list[dict],
        task: TaskType,
        transform=None,
        target_transform=None,
        class_names: list[str] | None = None,
    ) -> None:
        self.samples = samples
        self.task = task
        self.transform = transform
        self.target_transform = target_transform
        self.class_names = class_names or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        original_size = image.size  # (W, H)

        result = {"image_path": sample["image_path"], "original_size": original_size}

        if self.task == "classification":
            label = sample["label"]
            if self.transform:
                image = self.transform(image)
            result["image"] = image
            result["label"] = label

        elif self.task == "detection":
            boxes = torch.tensor(sample.get("boxes", []), dtype=torch.float32)
            labels = torch.tensor(sample.get("labels", []), dtype=torch.long)

            if self.transform:
                image = self.transform(image)

            # Normalize boxes to [0, 1] and convert from xyxy to cxcywh
            if len(boxes) > 0:
                W, H = original_size  # original PIL image size (W, H)
                boxes[:, 0] /= W  # x1
                boxes[:, 1] /= H  # y1
                boxes[:, 2] /= W  # x2
                boxes[:, 3] /= H  # y2
                # Convert xyxy -> cxcywh
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                cy = (boxes[:, 1] + boxes[:, 3]) / 2
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                boxes = torch.stack([cx, cy, w, h], dim=1)

            result["image"] = image
            result["boxes"] = boxes  # (N, 4) in cxcywh normalized [0,1]
            result["labels"] = labels  # (N,)

        elif self.task == "segmentation":
            mask_path = sample.get("mask_path")
            if mask_path:
                mask = np.array(Image.open(mask_path))
            else:
                mask = np.zeros(
                    (image.size[1], image.size[0]), dtype=np.int64
                )

            if self.transform:
                image = self.transform(image)

            result["image"] = image
            result["mask"] = torch.from_numpy(mask).long()

        return result

    @classmethod
    def from_folder(
        cls,
        root: str | Path,
        task: TaskType = "classification",
        transform=None,
        **kwargs,
    ) -> FFTDataset:
        """Load dataset from a folder structure.

        For classification: expects root/class_name/image.jpg structure.
        For detection/segmentation: expects root/images/ and root/annotations/.
        """
        root = Path(root)

        if task == "classification":
            return cls._from_classification_folder(root, transform, **kwargs)
        elif task == "detection":
            return cls._from_detection_folder(root, transform, **kwargs)
        elif task == "segmentation":
            return cls._from_segmentation_folder(root, transform, **kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")

    @classmethod
    def _from_classification_folder(
        cls, root: Path, transform=None, **kwargs
    ) -> FFTDataset:
        """Load classification dataset from folder structure (one subfolder per class)."""
        samples = []
        class_names = sorted(
            [d.name for d in root.iterdir() if d.is_dir()]
        )
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = root / class_name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}:
                    samples.append(
                        {"image_path": str(img_path), "label": class_to_idx[class_name]}
                    )

        logger.info(
            "Loaded classification dataset: %d images, %d classes",
            len(samples),
            len(class_names),
        )
        return cls(
            samples,
            task="classification",
            transform=transform,
            class_names=class_names,
            **kwargs,
        )

    @classmethod
    def _from_detection_folder(
        cls, root: Path, transform=None, **kwargs
    ) -> FFTDataset:
        """Load detection dataset. Expects COCO-format annotations.json in root."""
        from core.data.formats import load_coco

        ann_file = root / "annotations.json"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Expected annotations.json at {ann_file}. "
                "Use load_coco(), load_voc(), or load_yolo() for other formats."
            )
        return load_coco(ann_file, root / "images", task="detection", transform=transform, **kwargs)

    @classmethod
    def _from_segmentation_folder(
        cls, root: Path, transform=None, **kwargs
    ) -> FFTDataset:
        """Load segmentation dataset. Expects images/ and masks/ subdirectories."""
        images_dir = root / "images"
        masks_dir = root / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(
                f"Expected 'images/' and 'masks/' subdirectories in {root}"
            )

        samples = []
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}:
                mask_path = masks_dir / img_path.with_suffix(".png").name
                if mask_path.exists():
                    samples.append(
                        {"image_path": str(img_path), "mask_path": str(mask_path)}
                    )

        logger.info("Loaded segmentation dataset: %d images", len(samples))
        return cls(
            samples, task="segmentation", transform=transform, **kwargs
        )

    def split(
        self, val_ratio: float = 0.2, seed: int = 42
    ) -> tuple[FFTDataset, FFTDataset]:
        """Split dataset into train and validation sets.

        Args:
            val_ratio: Fraction of data for validation.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.samples))
        val_size = int(len(indices) * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_samples = [self.samples[i] for i in train_indices]
        val_samples = [self.samples[i] for i in val_indices]

        train_ds = FFTDataset(
            train_samples,
            self.task,
            self.transform,
            self.target_transform,
            self.class_names,
        )
        val_ds = FFTDataset(
            val_samples,
            self.task,
            self.transform,
            self.target_transform,
            self.class_names,
        )
        return train_ds, val_ds

    @staticmethod
    def detection_collate_fn(batch: list[dict]) -> dict:
        """Custom collate for detection that pads boxes/labels to the same length.

        Pads labels with -1 (ignored in cross_entropy) and boxes with 0.
        """
        images = torch.stack([item["image"] for item in batch])
        max_objects = max(item["labels"].shape[0] for item in batch)
        # Ensure at least 1 slot to avoid empty tensors
        max_objects = max(max_objects, 1)

        B = len(batch)
        padded_labels = torch.full((B, max_objects), -1, dtype=torch.long)
        padded_boxes = torch.zeros((B, max_objects, 4), dtype=torch.float32)

        for i, item in enumerate(batch):
            n = item["labels"].shape[0]
            if n > 0:
                padded_labels[i, :n] = item["labels"]
                padded_boxes[i, :n] = item["boxes"]

        return {
            "image": images,
            "labels": padded_labels,
            "boxes": padded_boxes,
        }

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        stats = {
            "num_samples": len(self.samples),
            "task": self.task,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
        }

        if self.task == "classification":
            from collections import Counter

            label_counts = Counter(s["label"] for s in self.samples)
            stats["class_distribution"] = {
                self.class_names[k] if k < len(self.class_names) else str(k): v
                for k, v in sorted(label_counts.items())
            }

        return stats
