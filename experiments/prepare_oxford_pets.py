"""Download Oxford-IIIT Pets and convert trimap segmentation masks to bounding boxes.

Saves a COCO-format dataset to experiments/datasets/oxford_pets/ with 37 breed classes.
Each image gets one bounding box derived from the foreground (pixel==1) extent of the mask.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


OUT_DIR = Path(__file__).parent / "datasets" / "oxford_pets"
RAW_DIR = Path(__file__).parent / "datasets" / "oxford_pets_raw"


def main():
    import torchvision.datasets as tvd

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if (OUT_DIR / "annotations.json").exists():
        with open(OUT_DIR / "annotations.json") as f:
            d = json.load(f)
        print(f"Already prepared: {len(d['images'])} images, {len(d['categories'])} classes")
        return OUT_DIR

    print("Downloading Oxford-IIIT Pets (~800 MB)...")
    ds = tvd.OxfordIIITPet(
        root=str(RAW_DIR),
        split="trainval",
        target_types=["category", "segmentation"],
        download=True,
    )

    # Build category list from the dataset's class names
    class_names = ds.classes  # list of 37 breed names
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    cat_id_map = {i: i + 1 for i in range(len(class_names))}  # 0-indexed → 1-indexed

    images_out = OUT_DIR / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    images_list = []
    annotations_list = []
    ann_id = 0
    skipped = 0

    print(f"Converting {len(ds)} images...")
    for idx in tqdm(range(len(ds))):
        img_pil, (cat_idx, mask_pil) = ds[idx]

        # Derive source filename from the dataset's internal list
        img_fname = Path(ds._images[idx]).name  # e.g. "Abyssinian_1.jpg"
        dst = images_out / img_fname

        w, h = img_pil.size

        # Trimap: 1=foreground, 2=background, 3=boundary — use foreground extent
        mask_arr = np.array(mask_pil)
        fg = mask_arr == 1
        if not fg.any():
            skipped += 1
            continue

        rows = np.where(np.any(fg, axis=1))[0]
        cols = np.where(np.any(fg, axis=0))[0]
        y1, y2 = int(rows[0]), int(rows[-1])
        x1, x2 = int(cols[0]), int(cols[-1])
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            skipped += 1
            continue

        # Copy image to output dir
        if not dst.exists():
            img_pil.save(dst)

        images_list.append({"id": idx, "file_name": img_fname, "width": w, "height": h})
        annotations_list.append({
            "id": ann_id,
            "image_id": idx,
            "category_id": cat_id_map[int(cat_idx)],
            "bbox": [x1, y1, bw, bh],   # COCO: x, y, w, h
            "area": bw * bh,
            "iscrowd": 0,
        })
        ann_id += 1

    coco_out = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories,
    }
    with open(OUT_DIR / "annotations.json", "w") as f:
        json.dump(coco_out, f)

    print(f"\nDone.")
    print(f"  Images:      {len(images_list)}  (skipped {skipped} with empty masks)")
    print(f"  Annotations: {len(annotations_list)}")
    print(f"  Classes:     {len(categories)}")
    print(f"  Path:        {OUT_DIR}")

    # Clean up raw download to save space (~800 MB)
    print("\nCleaning up raw download...")
    shutil.rmtree(RAW_DIR, ignore_errors=True)

    return OUT_DIR


if __name__ == "__main__":
    main()
