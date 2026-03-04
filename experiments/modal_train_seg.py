"""Modal cloud training for semantic segmentation: DINOv2 + UPerNetHead.

Oxford-IIIT Pets trimap segmentation, 3 classes:
  0 = background, 1 = pet, 2 = boundary

Uses an A10G (24 GB VRAM), 30 epochs, optional LoRA rank=4.

Usage:
    # Upload seg dataset to Modal volume (one-time):
    modal run experiments/modal_train_seg.py::upload_dataset

    # Run training:
    modal run --detach experiments/modal_train_seg.py::main

    # Download results:
    modal volume get fft-results /seg_pets ./experiments/results_modal/seg_pets
"""

from __future__ import annotations

from pathlib import Path

import modal

from experiments.modal_utils import modal_ignore

ROOT = Path(__file__).parent.parent

# ── Persistent Modal volumes ──────────────────────────────────────────────────
dataset_vol = modal.Volume.from_name("fft-datasets",  create_if_missing=True)
results_vol = modal.Volume.from_name("fft-results",   create_if_missing=True)
hub_vol     = modal.Volume.from_name("fft-hub-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        add_python="3.11",
    )
    .pip_install(
        "torchvision>=0.20.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics>=1.0",
        "albumentations>=1.3.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10",
        "tqdm>=4.65.0",
    )
    .add_local_dir(ROOT, remote_path="/app", ignore=modal_ignore)
)

app = modal.App("fft-seg", image=image)


# ── Training function ─────────────────────────────────────────────────────────
@app.function(
    gpu="A10G",
    image=image,
    volumes={
        "/datasets":          dataset_vol,
        "/results":           results_vol,
        "/root/.cache/torch": hub_vol,
    },
    timeout=60 * 60 * 4,   # 4-hour cap
    secrets=[],
)
def train_seg(
    encoder_name:  str   = "dinov2_vits14",
    num_classes:   int   = 3,
    fpn_channels:  int   = 256,
    input_size:    int   = 448,
    epochs:        int   = 30,
    batch_size:    int   = 8,
    lr:            float = 1e-4,
    lora_rank:     int   = 4,
    num_workers:   int   = 4,
) -> dict:
    """Run UPerNet segmentation training on Modal A10G."""
    import json
    import sys
    import time

    sys.path.insert(0, "/app")

    from core.encoders import create_encoder
    from core.data.dataset import FFTDataset
    from core.decoders.segmentation import UPerNetHead
    from core.training.trainer import Trainer

    data_dir = Path("/datasets/oxford_pets_seg")
    out_dir  = Path("/results/seg_pets")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Segmentation Training: DINOv2 + UPerNetHead")
    print(f"  Encoder:     {encoder_name}")
    print(f"  num_classes: {num_classes}  (0=bg, 1=pet, 2=boundary)")
    print(f"  input_size:  {input_size}")
    print(f"  epochs:      {epochs}, batch={batch_size}, lr={lr}")
    print(f"  LoRA rank:   {lora_rank}")
    print("=" * 60)

    encoder   = create_encoder(encoder_name, input_size=input_size)
    dataset   = FFTDataset.from_folder(
        data_dir,
        task="segmentation",
        transform=encoder.get_transform(),
        output_size=input_size,
    )
    print(f"Dataset: {len(dataset)} image/mask pairs")
    train_ds, val_ds = dataset.split()
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    decoder = UPerNetHead(
        encoder,
        num_classes=num_classes,
        fpn_channels=fpn_channels,
        output_size=input_size,
    )
    print(f"Decoder trainable params: {decoder.num_trainable_params():,}")

    t0      = time.time()
    trainer = Trainer(
        decoder=decoder,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        scheduler="cosine",
        warmup_epochs=2,
        early_stopping_patience=10,
        checkpoint_dir=out_dir / "checkpoints",
        num_workers=num_workers,
        training_mode="standard",
        lora_rank=lora_rank,   # Trainer handles enable_lora() internally
    )
    results = trainer.fit()
    elapsed = time.time() - t0

    out = {
        **results,
        "elapsed_sec":  elapsed,
        "encoder_name": encoder_name,
        "num_classes":  num_classes,
        "input_size":   input_size,
        "lora_rank":    lora_rank,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)

    results_vol.commit()
    print(f"\nTotal time: {elapsed / 60:.1f} min")
    print(f"val_loss: {results.get('best_val_loss', float('nan')):.4f}")
    return out


# ── Dataset upload helper ─────────────────────────────────────────────────────
@app.local_entrypoint()
def upload_dataset():
    """Upload the Oxford Pets seg dataset to the fft-datasets Modal volume.

    Run once:
        modal run experiments/modal_train_seg.py::upload_dataset
    """
    ds_dir = ROOT / "experiments" / "datasets" / "oxford_pets_seg"
    if not (ds_dir / "masks").exists():
        print("Dataset not found. Run first:")
        print("  python experiments/prepare_oxford_pets_seg.py")
        return

    n_masks  = len(list((ds_dir / "masks").glob("*.png")))
    n_images = len(list((ds_dir / "images").glob("*")))
    print(f"Uploading {ds_dir}  ({n_images} images, {n_masks} masks) ...")
    with dataset_vol.batch_upload(force=True) as batch:
        batch.put_directory(str(ds_dir), "/oxford_pets_seg")
    print("Upload complete.")


# ── Main entrypoint ───────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    encoder:    str   = "dinov2_vits14",
    epochs:     int   = 30,
    batch_size: int   = 8,
    lr:         float = 1e-4,
    lora_rank:  int   = 4,
    input_size: int   = 448,
):
    """Submit segmentation training to Modal.

    Examples:
        # Default (30 epochs, LoRA rank=4):
        modal run --detach experiments/modal_train_seg.py::main

        # Quick smoke-test (3 epochs):
        modal run experiments/modal_train_seg.py::main -- --epochs 3

        # Larger encoder:
        modal run --detach experiments/modal_train_seg.py::main -- --encoder dinov2_vitb14
    """
    print("Submitting segmentation job to Modal A10G:")
    print(f"  Encoder:    {encoder}")
    print(f"  Epochs:     {epochs}, batch={batch_size}, lr={lr}")
    print(f"  LoRA rank:  {lora_rank}")
    print(f"  Input size: {input_size}")

    result = train_seg.remote(
        encoder_name=encoder,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lora_rank=lora_rank,
        input_size=input_size,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  val_loss:       {result.get('best_val_loss', float('nan')):.4f}")
    print(f"  epochs_trained: {result.get('epochs_trained', '?')}")
    print(f"  total_time:     {result['elapsed_sec'] / 60:.1f} min")
    print("\nResults saved to Modal volume 'fft-results/seg_pets/'")
    print("Download with:")
    print("  modal volume get fft-results /seg_pets ./experiments/results_modal/seg_pets")
