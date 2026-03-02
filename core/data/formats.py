"""Import datasets from common annotation formats (COCO, VOC, YOLO, CSV)."""

from __future__ import annotations

import csv
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)


def load_coco(
    annotation_file: str | Path,
    images_dir: str | Path,
    task: str = "detection",
    transform=None,
    **kwargs,
) -> "FFTDataset":
    """Load dataset from COCO JSON format.

    Args:
        annotation_file: Path to the COCO annotations JSON file.
        images_dir: Directory containing the images.
        task: 'detection' or 'segmentation'.
        transform: Optional image transform.

    Returns:
        An FFTDataset instance.
    """
    from core.data.dataset import FFTDataset

    annotation_file = Path(annotation_file)
    images_dir = Path(images_dir)

    with open(annotation_file) as f:
        coco = json.load(f)

    # Build lookup tables
    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    class_names = [categories[k] for k in sorted(categories.keys())]
    cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(sorted(categories.keys()))}

    image_id_to_info = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    image_annotations: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        image_id = ann["image_id"]
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)

    samples = []
    for image_id, img_info in image_id_to_info.items():
        image_path = images_dir / img_info["file_name"]
        if not image_path.exists():
            logger.warning("Image not found: %s", image_path)
            continue

        anns = image_annotations.get(image_id, [])

        if task == "detection":
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann["bbox"]  # COCO format: x, y, width, height
                boxes.append([x, y, x + w, y + h])  # Convert to xyxy
                labels.append(cat_id_to_idx[ann["category_id"]])

            samples.append(
                {
                    "image_path": str(image_path),
                    "boxes": boxes,
                    "labels": labels,
                }
            )

        elif task == "segmentation":
            # For segmentation, we'd need to render polygon masks
            # This is a simplified version that handles mask paths
            samples.append(
                {
                    "image_path": str(image_path),
                    "annotations": anns,
                    "mask_path": None,  # Will be rendered on-the-fly
                }
            )

    logger.info(
        "Loaded COCO %s dataset: %d images, %d classes",
        task,
        len(samples),
        len(class_names),
    )
    return FFTDataset(
        samples, task=task, transform=transform, class_names=class_names, **kwargs
    )


def load_voc(
    annotations_dir: str | Path,
    images_dir: str | Path,
    task: str = "detection",
    transform=None,
    **kwargs,
) -> "FFTDataset":
    """Load dataset from Pascal VOC XML format.

    Args:
        annotations_dir: Directory containing VOC XML annotation files.
        images_dir: Directory containing the images.
        task: 'detection' or 'segmentation'.
        transform: Optional image transform.

    Returns:
        An FFTDataset instance.
    """
    from core.data.dataset import FFTDataset

    annotations_dir = Path(annotations_dir)
    images_dir = Path(images_dir)

    class_names_set: set[str] = set()
    raw_samples = []

    for xml_file in sorted(annotations_dir.glob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename_elem = root.find("filename")
        if filename_elem is None or filename_elem.text is None:
            continue
        filename = filename_elem.text
        image_path = images_dir / filename

        if not image_path.exists():
            logger.warning("Image not found: %s", image_path)
            continue

        boxes = []
        labels_str = []

        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None or name_elem.text is None:
                continue
            class_name = name_elem.text
            class_names_set.add(class_name)

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))

            boxes.append([xmin, ymin, xmax, ymax])
            labels_str.append(class_name)

        raw_samples.append(
            {
                "image_path": str(image_path),
                "boxes": boxes,
                "labels_str": labels_str,
            }
        )

    # Convert class names to indices
    class_names = sorted(class_names_set)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = []
    for raw in raw_samples:
        samples.append(
            {
                "image_path": raw["image_path"],
                "boxes": raw["boxes"],
                "labels": [class_to_idx[n] for n in raw["labels_str"]],
            }
        )

    logger.info(
        "Loaded VOC %s dataset: %d images, %d classes",
        task,
        len(samples),
        len(class_names),
    )
    return FFTDataset(
        samples, task=task, transform=transform, class_names=class_names, **kwargs
    )


def load_yolo(
    labels_dir: str | Path,
    images_dir: str | Path,
    class_names: list[str] | None = None,
    class_names_file: str | Path | None = None,
    image_size: tuple[int, int] = (518, 518),
    transform=None,
    **kwargs,
) -> "FFTDataset":
    """Load dataset from YOLO TXT format.

    YOLO format: each line is `class_id cx cy w h` (normalized 0-1).

    Args:
        labels_dir: Directory containing YOLO .txt label files.
        images_dir: Directory containing images (same basenames as labels).
        class_names: List of class names ordered by class ID.
        class_names_file: Path to a file with one class name per line.
        image_size: Default image size for denormalization.
        transform: Optional image transform.

    Returns:
        An FFTDataset instance.
    """
    from core.data.dataset import FFTDataset

    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)

    if class_names_file:
        with open(class_names_file) as f:
            class_names = [line.strip() for line in f if line.strip()]
    if class_names is None:
        class_names = []

    samples = []
    img_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}

    for label_file in sorted(labels_dir.glob("*.txt")):
        # Find matching image
        stem = label_file.stem
        image_path = None
        for ext in img_extensions:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            logger.warning("No image found for label: %s", label_file)
            continue

        boxes = []
        labels = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # Convert from YOLO (normalized center) to xyxy (pixel coords)
                img_w, img_h = image_size
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h

                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

                # Track max class_id for auto class_names
                if not class_names:
                    while len(class_names) <= class_id:
                        class_names.append(f"class_{len(class_names)}")

        samples.append(
            {"image_path": str(image_path), "boxes": boxes, "labels": labels}
        )

    logger.info(
        "Loaded YOLO dataset: %d images, %d classes",
        len(samples),
        len(class_names),
    )
    return FFTDataset(
        samples,
        task="detection",
        transform=transform,
        class_names=class_names,
        **kwargs,
    )


def load_csv(
    csv_file: str | Path,
    images_dir: str | Path,
    task: str = "classification",
    image_col: str = "image",
    label_col: str = "label",
    transform=None,
    **kwargs,
) -> "FFTDataset":
    """Load dataset from a CSV file.

    Args:
        csv_file: Path to the CSV file.
        images_dir: Base directory for resolving image paths.
        task: Task type (currently supports 'classification').
        image_col: Column name for image filenames.
        label_col: Column name for labels.
        transform: Optional image transform.

    Returns:
        An FFTDataset instance.
    """
    from core.data.dataset import FFTDataset

    csv_file = Path(csv_file)
    images_dir = Path(images_dir)

    samples = []
    class_names_set: set[str] = set()

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = []
        for row in reader:
            image_path = images_dir / row[image_col]
            label_str = row[label_col]
            class_names_set.add(label_str)
            raw_rows.append({"image_path": str(image_path), "label_str": label_str})

    class_names = sorted(class_names_set)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for raw in raw_rows:
        samples.append(
            {
                "image_path": raw["image_path"],
                "label": class_to_idx[raw["label_str"]],
            }
        )

    logger.info(
        "Loaded CSV %s dataset: %d images, %d classes",
        task,
        len(samples),
        len(class_names),
    )
    return FFTDataset(
        samples, task=task, transform=transform, class_names=class_names, **kwargs
    )
