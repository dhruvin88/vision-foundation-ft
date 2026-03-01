"""Frozen-encoder training on Oxford-IIIT Pets (mask-derived boxes).

dinov2_vits14 + RTDETRDecoder, no LoRA.
  - num_queries=10   (Oxford Pets = 1 object/image, 10 queries is sufficient)
  - num_decoder_layers=6  (was 4)
  - dim_feedforward=2048  (was 1024)
  - deeper cls_head (3-layer MLP)
Results saved to experiments/results/deim_pets_no_lora/.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR          = Path(__file__).parent / "datasets" / "oxford_pets"
OUT_DIR           = Path(__file__).parent / "results" / "deim_pets_no_lora"
EPOCHS            = 60
BATCH             = 8
LR                = 1e-4
WORKERS           = 0
NUM_QUERIES       = 10
NUM_DEC_LAYERS    = 6
DIM_FEEDFORWARD   = 2048


def main():
    from core.encoders import create_encoder
    from core.data.dataset import FFTDataset
    from core.training.trainer import Trainer
    from core.decoders.rtdetr import RTDETRDecoder

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "annotations.json") as f:
        meta = json.load(f)
    num_classes = len(meta["categories"])

    print("=" * 60)
    print("Frozen encoder - Oxford-IIIT Pets (mask -> bbox)")
    print(f"  Classes:         {num_classes}")
    print(f"  Epochs:          {EPOCHS}")
    print(f"  Batch size:      {BATCH}")
    print(f"  LR:              {LR}")
    print(f"  LoRA:            disabled")
    print(f"  num_queries:     {NUM_QUERIES}")
    print(f"  decoder_layers:  {NUM_DEC_LAYERS}")
    print(f"  dim_feedforward: {DIM_FEEDFORWARD}")
    print("=" * 60)

    t0 = time.time()
    encoder = create_encoder("dinov2_vits14", input_size=378)
    decoder = RTDETRDecoder(
        encoder,
        num_classes=num_classes,
        num_queries=NUM_QUERIES,
        num_decoder_layers=NUM_DEC_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
    )
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
        lora_rank=0,
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
