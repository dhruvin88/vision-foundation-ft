"""Local smoke test — semantic segmentation on Oxford-IIIT Pets trimaps.

DINOv2 ViT-S/14 (frozen) + UPerNetHead, 3 classes:
  0 = background, 1 = pet, 2 = boundary

Run prepare_oxford_pets_seg.py first, then:
    python experiments/train_seg_pets.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR   = Path(__file__).parent / "datasets" / "oxford_pets_seg"
OUT_DIR    = Path(__file__).parent / "results" / "seg_pets"
EPOCHS     = 5
BATCH      = 4
LR         = 1e-4
WORKERS    = 0
INPUT_SIZE = 224   # keeps it fast on CPU/GPU; 224/14=16 patches
NUM_CLASSES = 3    # background, pet, boundary


class SegDatasetWithResize:
    """Wraps FFTDataset segmentation samples and resizes masks to output_size."""

    def __init__(self, samples: list[dict], transform, output_size: int) -> None:
        self.samples     = samples
        self.transform   = transform
        self.output_size = output_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s     = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        mask  = Image.open(s["mask_path"])  # grayscale, NEAREST for class IDs

        if self.transform:
            image = self.transform(image)

        # Resize mask to output_size with NEAREST interpolation (preserves class IDs)
        mask = mask.resize((self.output_size, self.output_size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return {"image": image, "mask": mask}

    def split(self, val_ratio: float = 0.2, seed: int = 42):
        rng     = np.random.RandomState(seed)
        indices = rng.permutation(len(self.samples))
        val_n   = int(len(indices) * val_ratio)
        val_idx, train_idx = indices[:val_n], indices[val_n:]

        train_samples = [self.samples[i] for i in train_idx]
        val_samples   = [self.samples[i] for i in val_idx]

        train_ds = SegDatasetWithResize(train_samples, self.transform, self.output_size)
        val_ds   = SegDatasetWithResize(val_samples,   self.transform, self.output_size)
        return train_ds, val_ds


def main() -> None:
    from core.encoders import create_encoder
    from core.decoders.segmentation import UPerNetHead
    from core.training.trainer import Trainer

    if not (DATA_DIR / "masks").exists():
        print(f"Dataset not found at {DATA_DIR}")
        print("Run first:  python experiments/prepare_oxford_pets_seg.py")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build encoder + decoder
    encoder = create_encoder("dinov2_vits14", input_size=INPUT_SIZE)
    decoder = UPerNetHead(
        encoder,
        num_classes=NUM_CLASSES,
        fpn_channels=256,
        output_size=INPUT_SIZE,
    )
    print(f"Decoder trainable params: {decoder.num_trainable_params():,}")

    # Collect (image_path, mask_path) pairs
    images_dir = DATA_DIR / "images"
    masks_dir  = DATA_DIR / "masks"
    samples    = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            mask_path = masks_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                samples.append({"image_path": str(img_path), "mask_path": str(mask_path)})
    print(f"Dataset: {len(samples)} image/mask pairs")

    dataset  = SegDatasetWithResize(samples, encoder.get_transform(), output_size=INPUT_SIZE)
    train_ds, val_ds = dataset.split()
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    t0 = time.time()
    trainer = Trainer(
        decoder=decoder,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH,
        scheduler="cosine",
        warmup_epochs=1,
        checkpoint_dir=OUT_DIR / "checkpoints",
        num_workers=WORKERS,
        training_mode="standard",
        lora_rank=0,
    )

    results = trainer.fit()
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  val_loss: {results.get('best_val_loss', float('nan')):.4f}")
    print(f"  Epochs:   {results.get('epochs_trained', '?')}")
    print(f"  Time:     {elapsed / 60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
