"""Inference utilities for running predictions with trained models."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image

from core.decoders.base import BaseDecoder

logger = logging.getLogger(__name__)


def run_inference(
    decoder: BaseDecoder,
    image_paths: list[str | Path] | str | Path,
    device: str = "auto",
    batch_size: int = 16,
) -> list[dict]:
    """Run inference on a list of images using a trained decoder.

    Args:
        decoder: Trained decoder with attached frozen encoder.
        image_paths: Single image path or list of paths.
        device: Device to run inference on ('auto', 'cuda', 'cpu').
        batch_size: Batch size for inference.

    Returns:
        List of prediction dictionaries, one per image.
    """
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    decoder = decoder.to(device)
    decoder.eval()

    transform = decoder.encoder.get_transform()
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []

        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)

        batch_tensor = torch.stack(images).to(device)

        with torch.no_grad():
            features = decoder.encoder.forward_features(batch_tensor)
            predictions = decoder(features)

        # Process predictions based on task
        for j, path in enumerate(batch_paths):
            result = {"image_path": str(path)}

            if decoder.task == "classification":
                logits = predictions[j]  # (num_classes,)
                probs = torch.softmax(logits, dim=0)
                pred_class = probs.argmax().item()
                result["predicted_class"] = pred_class
                result["confidence"] = probs[pred_class].item()
                result["probabilities"] = probs.cpu().tolist()

            elif decoder.task == "detection":
                pred_logits = predictions["pred_logits"][j]  # (Q, C+1)
                pred_boxes = predictions["pred_boxes"][j]  # (Q, 4)

                # Filter out background and low-confidence predictions
                probs = torch.softmax(pred_logits, dim=-1)
                scores, labels = probs[:, :-1].max(dim=-1)  # Exclude no-object class

                # Keep predictions above threshold
                keep = scores > 0.5
                result["boxes"] = pred_boxes[keep].cpu().tolist()
                result["labels"] = labels[keep].cpu().tolist()
                result["scores"] = scores[keep].cpu().tolist()

            elif decoder.task == "segmentation":
                mask = predictions[j]  # (num_classes, H, W)
                pred_mask = mask.argmax(dim=0)  # (H, W)
                result["mask"] = pred_mask.cpu()
                result["class_areas"] = {
                    int(c): int((pred_mask == c).sum())
                    for c in pred_mask.unique()
                }

            results.append(result)

    return results
