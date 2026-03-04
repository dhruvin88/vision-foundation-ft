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

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR    = Path(__file__).parent / "datasets" / "oxford_pets_seg"
OUT_DIR     = Path(__file__).parent / "results" / "seg_pets"
EPOCHS      = 5
BATCH       = 4
LR          = 1e-4
WORKERS     = 0
INPUT_SIZE  = 224   # 224/14=16 patches; keeps smoke test fast
NUM_CLASSES = 3     # background, pet, boundary


def main() -> None:
    from core.encoders import create_encoder
    from core.data.dataset import FFTDataset
    from core.decoders.segmentation import UPerNetHead
    from core.training.trainer import Trainer

    if not (DATA_DIR / "masks").exists():
        print(f"Dataset not found at {DATA_DIR}")
        print("Run first:  python experiments/prepare_oxford_pets_seg.py")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    encoder = create_encoder("dinov2_vits14", input_size=INPUT_SIZE)
    decoder = UPerNetHead(
        encoder,
        num_classes=NUM_CLASSES,
        fpn_channels=256,
        output_size=INPUT_SIZE,
    )
    print(f"Decoder trainable params: {decoder.num_trainable_params():,}")

    dataset = FFTDataset.from_folder(
        DATA_DIR,
        task="segmentation",
        transform=encoder.get_transform(),
        output_size=INPUT_SIZE,
    )
    print(f"Dataset: {len(dataset)} image/mask pairs")
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
