"""Synthetic VQA dataset built from Oxford Pets COCO annotations.

Each __getitem__ call randomly selects one of four QA templates and returns
image + tokenized question/answer pairs for teacher-forced LM training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Breeds known to be cats (lowercase, underscore-separated); rest treated as dogs
_CAT_BREEDS = {
    "abyssinian", "bengal", "birman", "bombay", "british_shorthair",
    "egyptian_mau", "maine_coon", "persian", "ragdoll", "russian_blue",
    "siamese", "sphynx",
}


def _breed_to_species(breed: str) -> str:
    return "cat" if breed.lower().replace(" ", "_") in _CAT_BREEDS else "dog"


def _region_from_box(cx: float, cy: float) -> str:
    """Convert normalized center coordinates to a spatial description string."""
    row = "top" if cy < 0.4 else ("bottom" if cy > 0.6 else "center")
    col = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
    if row == "center" and col == "center":
        return "center"
    if row == "center":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


# Each template is a callable (breed, species, region) → (question, answer)
_QA_TEMPLATES = [
    lambda breed, species, region: (
        "What breed of animal is in this image?",
        f"This is a {breed}.",
    ),
    lambda breed, species, region: (
        "What type of animal is shown here?",
        f"This is a {species}.",
    ),
    lambda breed, species, region: (
        f"Where is the {breed} located in the image?",
        f"The {breed} is in the {region} of the image.",
    ),
    lambda breed, species, region: (
        "Describe what you see.",
        f"I can see a {breed} in the {region} of the image.",
    ),
]


class PetsVQADataset(Dataset):
    """Oxford Pets VQA dataset with synthetic question-answer pairs.

    Reads COCO-format annotations and generates QA pairs on the fly.
    Each __getitem__ randomly selects one of four QA templates.

    Args:
        annotations_json: Path to COCO-format annotations JSON.
        images_dir: Directory containing image files.
        tokenizer: HuggingFace tokenizer (must match the LLM).
        transform: Image transform (e.g. encoder.get_transform()).
        max_length: Maximum token sequence length for padding/truncation.
    """

    def __init__(
        self,
        annotations_json: str | Path,
        images_dir: str | Path,
        tokenizer,
        transform=None,
        max_length: int = 256,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

        with open(annotations_json) as f:
            coco = json.load(f)

        cat_id_to_name: dict[int, str] = {
            c["id"]: c["name"] for c in coco["categories"]
        }
        img_id_to_info: dict[int, dict] = {
            img["id"]: img for img in coco["images"]
        }

        # Group annotations by image id
        img_anns: dict[int, list] = {}
        for ann in coco["annotations"]:
            img_anns.setdefault(ann["image_id"], []).append(ann)

        self.samples: list[dict] = []
        for img_id, anns in img_anns.items():
            info = img_id_to_info[img_id]
            img_w, img_h = info["width"], info["height"]

            # Primary annotation determines the QA label
            ann = anns[0]
            breed = cat_id_to_name[ann["category_id"]].replace("_", " ")
            x, y, bw, bh = ann["bbox"]  # COCO bbox: [x, y, w, h] in pixels
            cx = (x + bw / 2) / img_w
            cy = (y + bh / 2) / img_h

            self.samples.append({
                "image_path": str(self.images_dir / info["file_name"]),
                "breed": breed,
                "species": _breed_to_species(breed),
                "cx": cx,
                "cy": cy,
                # All boxes for optional detection context (cxcywh normalized)
                "boxes": [
                    [
                        (a["bbox"][0] + a["bbox"][2] / 2) / img_w,
                        (a["bbox"][1] + a["bbox"][3] / 2) / img_h,
                        a["bbox"][2] / img_w,
                        a["bbox"][3] / img_h,
                    ]
                    for a in anns
                ],
                "class_labels": [a["category_id"] - 1 for a in anns],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        breed = sample["breed"]
        species = sample["species"]
        region = _region_from_box(sample["cx"], sample["cy"])

        template_fn = random.choice(_QA_TEMPLATES)
        question, answer = template_fn(breed, species, region)

        # TinyLlama / Zephyr chat format
        question_text = f"<|user|>\n{question}</s>\n<|assistant|>\n"
        answer_text = f"{answer}</s>"
        full_text = question_text + answer_text

        # Tokenize the full sequence (question + answer)
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)         # (T,)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Build labels: -100 for question and padding tokens, token id for answer
        question_enc = self.tokenizer(
            question_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        question_len = question_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:question_len] = -100          # mask question tokens
        labels[attention_mask == 0] = -100    # mask padding tokens

        boxes = (
            torch.tensor(sample["boxes"], dtype=torch.float32)
            if sample["boxes"]
            else torch.zeros(0, 4)
        )
        class_labels = (
            torch.tensor(sample["class_labels"], dtype=torch.long)
            if sample["class_labels"]
            else torch.zeros(0, dtype=torch.long)
        )

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "boxes": boxes,
            "class_labels": class_labels,
        }

    def split(
        self, val_ratio: float = 0.2, seed: int = 42
    ) -> tuple[_PetsVQASubset, _PetsVQASubset]:
        """Split into train and validation subsets (80/20 by default)."""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.samples))
        val_size = int(len(indices) * val_ratio)
        val_idx = indices[:val_size].tolist()
        train_idx = indices[val_size:].tolist()
        return _PetsVQASubset(self, train_idx), _PetsVQASubset(self, val_idx)

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Collate a batch, padding variable-length box tensors."""
        images = torch.stack([item["image"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        max_boxes = max(item["boxes"].shape[0] for item in batch)
        max_boxes = max(max_boxes, 1)
        B = len(batch)
        padded_boxes = torch.zeros(B, max_boxes, 4, dtype=torch.float32)
        padded_cls = torch.full((B, max_boxes), -1, dtype=torch.long)

        for i, item in enumerate(batch):
            n = item["boxes"].shape[0]
            if n > 0:
                padded_boxes[i, :n] = item["boxes"]
                padded_cls[i, :n] = item["class_labels"]

        return {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "boxes": padded_boxes,
            "class_labels": padded_cls,
        }


class _PetsVQASubset(Dataset):
    """Lightweight index-based subset of a PetsVQADataset."""

    def __init__(self, dataset: PetsVQADataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[self.indices[idx]]
