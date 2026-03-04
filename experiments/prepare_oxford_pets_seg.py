"""Download Oxford-IIIT Pets and save trimap masks for semantic segmentation.

Produces datasets/oxford_pets_seg/ with:
  images/   — symlinked or copied from oxford_pets/images/
  masks/    — grayscale PNGs with class IDs:
                0 = background  (trimap pixel == 2)
                1 = pet         (trimap pixel == 1)
                2 = boundary    (trimap pixel == 3)

Usage:
    python experiments/prepare_oxford_pets_seg.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

OUT_DIR    = Path(__file__).parent / "datasets" / "oxford_pets_seg"
PETS_DIR   = Path(__file__).parent / "datasets" / "oxford_pets"
RAW_DIR    = Path(__file__).parent / "datasets" / "oxford_pets_raw_seg"

# Trimap → class mapping
# trimap 1=foreground(pet) → 1, trimap 2=background → 0, trimap 3=boundary → 2
_TRIMAP_TO_CLASS = {1: 1, 2: 0, 3: 2}


def main() -> None:
    import torchvision.datasets as tvd

    masks_dir  = OUT_DIR / "masks"
    images_dir = OUT_DIR / "images"

    existing = list(masks_dir.glob("*.png")) if masks_dir.exists() else []
    if len(existing) > 100:
        print(f"Already prepared: {len(existing)} masks in {masks_dir}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Oxford-IIIT Pets trimaps (~800 MB)...")
    ds = tvd.OxfordIIITPet(
        root=str(RAW_DIR),
        split="trainval",
        target_types=["category", "segmentation"],
        download=True,
    )

    # Build a lookup table for faster remapping
    remap_table = np.zeros(256, dtype=np.uint8)
    for trimap_val, class_id in _TRIMAP_TO_CLASS.items():
        remap_table[trimap_val] = class_id

    print(f"Processing {len(ds)} images...")
    saved = 0
    for idx in tqdm(range(len(ds))):
        img_pil, (_cat_idx, mask_pil) = ds[idx]

        img_fname = Path(ds._images[idx]).name   # e.g. "Abyssinian_1.jpg"
        stem      = Path(img_fname).stem          # e.g. "Abyssinian_1"
        mask_out  = masks_dir / f"{stem}.png"

        if not mask_out.exists():
            mask_arr   = np.array(mask_pil, dtype=np.uint8)
            class_arr  = remap_table[mask_arr]
            Image.fromarray(class_arr, mode="L").save(mask_out)

        # Copy image if not already present from detection dataset
        dst_img = images_dir / img_fname
        if not dst_img.exists():
            src_img = PETS_DIR / "images" / img_fname
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            else:
                img_pil.save(dst_img)

        saved += 1

    print(f"\nDone: {saved} mask/image pairs in {OUT_DIR}")
    print("  Class IDs: 0=background  1=pet  2=boundary")

    print("\nCleaning up raw download...")
    shutil.rmtree(RAW_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
