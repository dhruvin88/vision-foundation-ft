"""DEIM + LoRA training on Oxford-IIIT Pets (mask-derived boxes).

dinov2_vits14 + RTDETRDecoder + LoRA rank=4, training_mode="deim".
Results saved to experiments/results/deim_pets_lora/.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR  = Path(__file__).parent / "datasets" / "oxford_pets"
OUT_DIR   = Path(__file__).parent / "results" / "deim_pets_lora"
EPOCHS    = 60
BATCH     = 8
LR        = 1e-4
WORKERS   = 0
LORA_RANK = 4


def main():
    from core.encoders import create_encoder
    from core.data.dataset import FFTDataset
    from core.training.trainer import Trainer
    from core.cli import _create_decoder

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "annotations.json") as f:
        meta = json.load(f)
    num_classes = len(meta["categories"])

    print("=" * 60)
    print("DEIM + LoRA - Oxford-IIIT Pets (mask -> bbox)")
    print(f"  Classes:    {num_classes}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Batch size: {BATCH}")
    print(f"  LR:         {LR}")
    print(f"  LoRA rank:  {LORA_RANK}")
    print("=" * 60)

    t0 = time.time()
    encoder = create_encoder("dinov2_vits14", input_size=384)
    decoder = _create_decoder("rtdetr", "detection", encoder, num_classes)
    print(f"Decoder trainable params: {decoder.num_trainable_params():,}")

    dataset = FFTDataset.from_folder(
        DATA_DIR, task="detection", transform=encoder.get_transform()
    )
    print(f"Dataset size: {len(dataset)}")

    trainer = Trainer(
        decoder=decoder,
        train_dataset=dataset,
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH,
        scheduler="cosine",
        warmup_epochs=2,
        augmentation="light",
        early_stopping_patience=20,
        checkpoint_dir=OUT_DIR / "checkpoints",
        num_workers=WORKERS,
        training_mode="standard",
        lora_rank=LORA_RANK,
    )

    results = trainer.fit()
    elapsed = time.time() - t0

    map50 = results.get("val_map50", float("nan"))
    mmap  = results.get("val_map",   float("nan"))
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  val_loss:  {results.get('best_val_loss', float('nan')):.4f}")
    print(f"  val_map50: {map50:.4f}")
    print(f"  val_map:   {mmap:.4f}")
    print(f"  Epochs:    {results.get('epochs_trained', '?')}")
    print(f"  Time:      {elapsed / 60:.1f} min")
    print("=" * 60)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({**results, "elapsed_sec": elapsed}, f, indent=2)


if __name__ == "__main__":
    main()
