"""End-to-end detection test using the Penn-Fudan Pedestrian dataset.

Downloads the Penn-Fudan Pedestrian dataset (~13MB, 170 images),
converts annotations to COCO format, and trains a DETRLite decoder
on top of a frozen DINOv2 encoder.

Usage:
    python examples/detection_e2e.py [--epochs 3] [--batch-size 2] [--subset 20]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def download_penn_fudan(dest_dir: Path) -> Path:
    """Download and extract the Penn-Fudan dataset."""
    zip_path = dest_dir / "PennFudanPed.zip"
    extracted_dir = dest_dir / "PennFudanPed"

    if extracted_dir.exists():
        logger.info("Dataset already extracted at %s", extracted_dir)
        return extracted_dir

    logger.info("Downloading Penn-Fudan dataset from %s ...", DATASET_URL)
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    logger.info("Download complete (%.1f MB)", zip_path.stat().st_size / 1e6)

    logger.info("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

    zip_path.unlink()
    return extracted_dir


def parse_penn_fudan_annotation(ann_path: Path) -> list[dict]:
    """Parse a Penn-Fudan annotation file and return bounding boxes.

    The annotation format has lines like:
        Bounding box for object 1 ... : (160, 182) - (302, 431)
    """
    boxes = []
    text = ann_path.read_text()
    pattern = r"Bounding box for object \d+.*?:\s*\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)"
    for match in re.finditer(pattern, text):
        xmin, ymin, xmax, ymax = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        boxes.append({
            "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax,
        })
    return boxes


def convert_to_coco(penn_fudan_dir: Path, output_dir: Path, max_images: int | None = None) -> Path:
    """Convert Penn-Fudan annotations to COCO JSON format.

    Args:
        penn_fudan_dir: Root of the extracted PennFudanPed directory.
        output_dir: Where to create the COCO-format dataset.
        max_images: Limit number of images (for quick testing).

    Returns:
        Path to the output directory.
    """
    images_src = penn_fudan_dir / "PNGImages"
    annotations_src = penn_fudan_dir / "Annotation"

    images_dst = output_dir / "images"
    images_dst.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "pedestrian"}],
    }

    image_id = 0
    ann_id = 0

    ann_files = sorted(annotations_src.glob("*"))
    if max_images:
        ann_files = ann_files[:max_images]

    for ann_file in ann_files:
        # Find matching image
        stem = ann_file.stem
        img_file = images_src / f"{stem}.png"
        if not img_file.exists():
            continue

        # Copy image
        dst_img = images_dst / img_file.name
        shutil.copy2(img_file, dst_img)

        # Get image dimensions
        from PIL import Image
        with Image.open(dst_img) as img:
            w, h = img.size

        coco["images"].append({
            "id": image_id,
            "file_name": img_file.name,
            "width": w,
            "height": h,
        })

        # Parse annotations
        boxes = parse_penn_fudan_annotation(ann_file)
        for box in boxes:
            bw = box["xmax"] - box["xmin"]
            bh = box["ymax"] - box["ymin"]
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [box["xmin"], box["ymin"], bw, bh],  # COCO format: x, y, w, h
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

        image_id += 1

    ann_path = output_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(
        "Created COCO dataset: %d images, %d annotations",
        len(coco["images"]),
        len(coco["annotations"]),
    )
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="E2E detection test with Penn-Fudan dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--subset", type=int, default=20, help="Number of images to use (0 for all)")
    parser.add_argument("--encoder", default="dinov2_vits14",
                        help="Encoder variant: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14")
    parser.add_argument("--decoder-size", default=None, choices=["small", "medium", "large"],
                        help="Decoder size preset (overrides individual decoder args)")
    parser.add_argument("--num-queries", type=int, default=None, help="Number of object queries (max detections)")
    parser.add_argument("--num-decoder-layers", type=int, default=None, help="Number of transformer decoder layers")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Decoder hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", default="./output_detection", help="Output directory")
    parser.add_argument("--data-dir", default=None, help="Directory to cache downloaded dataset")
    args = parser.parse_args()

    # Decoder size presets
    DECODER_PRESETS = {
        "small":  {"num_queries": 20,  "num_decoder_layers": 2, "hidden_dim": 128},
        "medium": {"num_queries": 50,  "num_decoder_layers": 4, "hidden_dim": 256},
        "large":  {"num_queries": 100, "num_decoder_layers": 6, "hidden_dim": 256},
    }

    if args.decoder_size:
        preset = DECODER_PRESETS[args.decoder_size]
    else:
        preset = DECODER_PRESETS["small"]

    # Individual args override preset values
    num_queries = args.num_queries or preset["num_queries"]
    num_decoder_layers = args.num_decoder_layers or preset["num_decoder_layers"]
    hidden_dim = args.hidden_dim or preset["hidden_dim"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a persistent cache dir for the dataset or a temp dir
    if args.data_dir:
        data_cache = Path(args.data_dir)
        data_cache.mkdir(parents=True, exist_ok=True)
    else:
        data_cache = Path(tempfile.mkdtemp(prefix="fft_pennfudan_"))

    # Step 1: Download dataset
    print("=" * 60)
    print("Step 1: Downloading Penn-Fudan Pedestrian Dataset")
    print("=" * 60)
    penn_fudan_dir = download_penn_fudan(data_cache)

    # Step 2: Convert to COCO format
    print("\n" + "=" * 60)
    print("Step 2: Converting to COCO format")
    print("=" * 60)
    coco_dir = data_cache / "coco_format"
    subset = args.subset if args.subset > 0 else None
    convert_to_coco(penn_fudan_dir, coco_dir, max_images=subset)

    # Step 3: Load encoder
    print("\n" + "=" * 60)
    print(f"Step 3: Loading encoder ({args.encoder})")
    print("=" * 60)
    from core.encoders.dinov2 import DINOv2Encoder
    encoder = DINOv2Encoder(args.encoder)
    print(f"  Encoder loaded: embed_dim={encoder.embed_dim}, patch_size={encoder.patch_size}")

    # Step 4: Create decoder
    print("\n" + "=" * 60)
    print(f"Step 4: Creating DETRLite decoder ({args.decoder_size or 'small'})")
    print("=" * 60)
    from core.decoders.detection import DETRLiteDecoder
    decoder = DETRLiteDecoder(
        encoder,
        num_classes=1,  # 1 class: pedestrian
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        hidden_dim=hidden_dim,
    )
    print(f"  Config: queries={num_queries}, layers={num_decoder_layers}, hidden_dim={hidden_dim}")
    print(f"  Decoder: {decoder.num_trainable_params():,} trainable parameters")

    # Step 5: Load dataset
    print("\n" + "=" * 60)
    print("Step 5: Loading dataset")
    print("=" * 60)
    from core.data.dataset import FFTDataset
    dataset = FFTDataset.from_folder(
        coco_dir,
        task="detection",
        transform=encoder.get_transform(),
    )
    print(f"  Dataset: {len(dataset)} images, classes={dataset.class_names}")

    # Step 6: Train
    print("\n" + "=" * 60)
    print(f"Step 6: Training ({args.epochs} epochs, batch_size={args.batch_size})")
    print("=" * 60)
    from core.training.trainer import Trainer
    trainer = Trainer(
        decoder=decoder,
        train_dataset=dataset,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        scheduler="cosine",
        warmup_epochs=1,
        early_stopping_patience=0,  # Disable for short runs
        checkpoint_dir=output_dir / "checkpoints",
        num_workers=0,  # 0 for Windows compatibility
        accelerator="cpu",
    )

    results = trainer.fit()
    print(f"\n  Training complete!")
    print(f"  Epochs trained: {results['epochs_trained']}")
    print(f"  Best val loss: {results['best_val_loss']:.4f}")

    # Step 7: Save
    print("\n" + "=" * 60)
    print("Step 7: Saving model")
    print("=" * 60)
    weights_path = output_dir / "decoder_weights.pt"
    trainer.save(weights_path)
    print(f"  Saved to: {weights_path}")
    print(f"  File size: {weights_path.stat().st_size / 1e6:.2f} MB")

    print("\n" + "=" * 60)
    print("SUCCESS: End-to-end detection pipeline works!")
    print("=" * 60)


if __name__ == "__main__":
    main()
