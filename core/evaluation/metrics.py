"""Task-specific evaluation metrics: accuracy, mAP, mIoU."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch


def compute_metrics(
    predictions: list[torch.Tensor | dict],
    targets: list[torch.Tensor | dict],
    task: Literal["classification", "detection", "segmentation"],
    num_classes: int | None = None,
) -> dict[str, float]:
    """Compute task-specific metrics.

    Args:
        predictions: List of model predictions.
        targets: List of ground truth targets.
        task: Task type.
        num_classes: Number of classes (required for segmentation mIoU).

    Returns:
        Dictionary of metric name -> value.
    """
    if task == "classification":
        return _classification_metrics(predictions, targets)
    elif task == "detection":
        return _detection_metrics(predictions, targets)
    elif task == "segmentation":
        return _segmentation_metrics(predictions, targets, num_classes or 2)
    else:
        raise ValueError(f"Unknown task: {task}")


def _classification_metrics(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> dict[str, float]:
    """Compute classification accuracy and per-class precision/recall."""
    all_preds = torch.cat([p.argmax(dim=-1) if p.ndim > 1 else p for p in predictions])
    all_targets = torch.cat(targets)

    correct = (all_preds == all_targets).sum().item()
    total = all_targets.numel()
    accuracy = correct / total if total > 0 else 0.0

    # Per-class accuracy
    num_classes = max(all_targets.max().item() + 1, all_preds.max().item() + 1)
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_targets == c
        if mask.sum() > 0:
            per_class_acc[f"class_{c}_acc"] = (
                (all_preds[mask] == c).float().mean().item()
            )

    return {"accuracy": accuracy, "num_correct": correct, "num_total": total, **per_class_acc}


def _detection_metrics(
    predictions: list[dict],
    targets: list[dict],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute detection mAP at the given IoU threshold.

    Simplified mAP computation for evaluation. For full COCO mAP,
    use pycocotools.
    """
    all_tp = 0
    all_fp = 0
    all_fn = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred.get("pred_boxes", torch.zeros(0, 4))
        pred_scores = pred.get("pred_scores", torch.zeros(0))
        target_boxes = target.get("boxes", torch.zeros(0, 4))

        if len(pred_boxes) == 0:
            all_fn += len(target_boxes)
            continue

        if len(target_boxes) == 0:
            all_fp += len(pred_boxes)
            continue

        # Compute IoU matrix
        ious = _box_iou(pred_boxes, target_boxes)

        matched_targets = set()
        # Sort predictions by score (descending)
        if len(pred_scores) > 0:
            order = pred_scores.argsort(descending=True)
        else:
            order = torch.arange(len(pred_boxes))

        for pred_idx in order:
            if len(ious) == 0:
                all_fp += 1
                continue
            best_iou, best_target = ious[pred_idx].max(dim=0)
            if best_iou >= iou_threshold and best_target.item() not in matched_targets:
                all_tp += 1
                matched_targets.add(best_target.item())
            else:
                all_fp += 1

        all_fn += len(target_boxes) - len(matched_targets)

    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP@50": precision,  # Simplified; use pycocotools for proper mAP
    }


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format

    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / union_area.clamp(min=1e-8)


def _segmentation_metrics(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    num_classes: int,
) -> dict[str, float]:
    """Compute segmentation mIoU and per-class IoU."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred, target in zip(predictions, targets):
        pred_labels = pred.argmax(dim=0) if pred.ndim > 2 else pred  # (H, W)
        pred_np = pred_labels.cpu().numpy().flatten()
        target_np = target.cpu().numpy().flatten()

        # Only count valid pixels
        valid = (target_np >= 0) & (target_np < num_classes)
        pred_np = pred_np[valid]
        target_np = target_np[valid]

        for p, t in zip(pred_np, target_np):
            confusion[t, p] += 1

    # Compute per-class IoU
    per_class_iou = {}
    ious = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        iou = tp / max(tp + fp + fn, 1)
        per_class_iou[f"class_{c}_iou"] = iou
        if confusion[c, :].sum() > 0:  # Only include classes present in GT
            ious.append(iou)

    miou = np.mean(ious) if ious else 0.0
    pixel_acc = np.diag(confusion).sum() / max(confusion.sum(), 1)

    return {
        "mIoU": float(miou),
        "pixel_accuracy": float(pixel_acc),
        **per_class_iou,
    }
