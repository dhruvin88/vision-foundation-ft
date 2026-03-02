"""Albumentations-based augmentation presets for all task types."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

AugmentationPreset = Literal["none", "light", "heavy"]


def get_augmentation_pipeline(
    preset: AugmentationPreset = "light",
    task: str = "classification",
    image_size: int = 518,
) -> A.Compose:
    """Get an augmentation pipeline for the given preset and task.

    Args:
        preset: Augmentation intensity -- 'none', 'light', or 'heavy'.
        task: Task type -- affects how bboxes/masks are handled.
        image_size: Target image size after augmentation.

    Returns:
        An albumentations Compose pipeline.
    """
    # Determine bbox and additional target params based on task
    bbox_params = None
    additional_targets = {}

    if task == "detection":
        bbox_params = A.BboxParams(
            format="pascal_voc",  # (x_min, y_min, x_max, y_max)
            label_fields=["labels"],
            min_visibility=0.3,
        )
    elif task == "segmentation":
        additional_targets = {"mask": "mask"}

    # Base transforms (always applied)
    base = [
        A.Resize(image_size, image_size),
    ]

    if preset == "none":
        transforms = base + [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]

    elif preset == "light":
        transforms = base + [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.3
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]

    elif preset == "heavy":
        transforms = base + [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.4
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=image_size // 16,
                max_width=image_size // 16,
                p=0.3,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    else:
        raise ValueError(f"Unknown augmentation preset: {preset}")

    return A.Compose(
        transforms,
        bbox_params=bbox_params,
        additional_targets=additional_targets if additional_targets else None,
    )


def _mosaic_merge(items: list[dict]) -> dict:
    """Merge 4 detection items into a 2×2 mosaic.

    Each item must have:
        image: (C, H, W) tensor
        boxes: (N, 4) cxcywh in [0, 1]
        labels: (N,) int tensor

    Returns a dict with the same keys as a normal detection item.
    """
    # Resize each image to half size
    half_imgs = [
        F.interpolate(item["image"].unsqueeze(0).float(), scale_factor=0.5, mode="bilinear", align_corners=False)[0]
        for item in items
    ]
    q0, q1, q2, q3 = half_imgs

    # Build 2×2 mosaic
    mosaic = torch.cat([
        torch.cat([q0, q1], dim=2),
        torch.cat([q2, q3], dim=2),
    ], dim=1)

    # Map boxes for each quadrant
    offsets = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
    all_boxes = []
    all_labels = []

    for item, (ox, oy) in zip(items, offsets):
        boxes = item["boxes"]
        labels = item["labels"]
        if boxes.shape[0] == 0:
            continue
        cx, cy, w, h = boxes.unbind(-1)
        new_boxes = torch.stack([ox + cx / 2, oy + cy / 2, w / 2, h / 2], dim=-1)
        # Filter out zero-area boxes
        valid = (new_boxes[:, 2] > 0) & (new_boxes[:, 3] > 0)
        all_boxes.append(new_boxes[valid])
        all_labels.append(labels[valid])

    if all_boxes:
        merged_boxes = torch.cat(all_boxes, dim=0)
        merged_labels = torch.cat(all_labels, dim=0)
    else:
        merged_boxes = torch.zeros(0, 4, dtype=torch.float32)
        merged_labels = torch.zeros(0, dtype=torch.long)

    result = dict(items[0])  # copy extra keys (e.g. image_id) from primary item
    result["image"] = mosaic
    result["boxes"] = merged_boxes
    result["labels"] = merged_labels
    return result


def get_val_transform(image_size: int = 518) -> A.Compose:
    """Get validation/inference transform (resize + normalize only)."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


class MosaicDetectionDataset(Dataset):
    """Wraps an FFTDataset and applies Mosaic augmentation for detection.

    Combines 4 images into a 2×2 grid. Boxes from each quadrant are mapped
    to the corresponding region in [0,1] space. Toggle ``self.enabled = False``
    to fall back to single-image mode (used for the latter half of DEIM training).
    """

    enabled: bool = True  # toggled by trainer epoch schedule

    def __init__(self, dataset, mosaic_prob: float = 1.0):
        self.dataset = dataset
        self.mosaic_prob = mosaic_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        import random
        if not self.enabled or random.random() > self.mosaic_prob:
            return self.dataset[idx]
        # Sample 3 additional random indices
        indices = [idx] + random.choices(range(len(self.dataset)), k=3)
        items = [self.dataset[i] for i in indices]
        return _mosaic_merge(items)
