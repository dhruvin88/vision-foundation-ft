"""Modal cloud training for VLM: DINOv2 + MLP projector + Phi-3.5-mini.

Two-stage training on an A10G (24 GB VRAM):
  Stage 1 (3 epochs)  -- projector alignment, LLM frozen, lr=1e-3
  Stage 2 (10 epochs) -- instruction tuning with LoRA rank=8, lr=2e-5

Usage:
    # Upload Oxford Pets dataset to Modal volume (one-time):
    modal run experiments/modal_train_vlm.py::upload_dataset

    # Run training:
    modal run experiments/modal_train_vlm.py

    # Download results:
    modal volume get fft-results /vlm_pets ./experiments/results_modal/vlm_pets
"""

from __future__ import annotations

from pathlib import Path

import modal

# ── Root of the local repo ────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── Persistent Modal volumes ──────────────────────────────────────────────────
dataset_vol = modal.Volume.from_name("fft-datasets",  create_if_missing=True)
results_vol = modal.Volume.from_name("fft-results",   create_if_missing=True)
hub_vol     = modal.Volume.from_name("fft-hub-cache", create_if_missing=True)
hf_vol      = modal.Volume.from_name("fft-hf-cache",  create_if_missing=True)

# ── Container image ───────────────────────────────────────────────────────────
_SKIP = {".venv", "__pycache__", ".git", ".pytest_cache", "checkpoints",
         "experiments/datasets", "experiments/results", "experiments/results_modal"}


def _ignore(path: Path) -> bool:
    parts = set(path.parts)
    path_str = path.as_posix()
    return bool(parts & {".venv", "__pycache__", ".git", ".pytest_cache", "checkpoints"}) or \
           any(s in path_str for s in ["experiments/datasets", "experiments/results",
                                        "experiments/results_modal"])


image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        add_python="3.11",
    )
    .pip_install(
        "torchvision>=0.20.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics[detection]>=1.0",
        "albumentations>=1.3.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10",
        "transformers>=4.50.0",
        "accelerate>=0.26.0",
        "sentencepiece>=0.1.99",
        "tqdm>=4.65.0",
    )
    .add_local_dir(ROOT, remote_path="/app", ignore=_ignore)
)

app = modal.App("fft-vlm", image=image)


# ── Training function ─────────────────────────────────────────────────────────
@app.function(
    gpu="A10G",          # 24 GB VRAM — runs Phi-3.5-mini full bfloat16
    image=image,
    volumes={
        "/datasets":          dataset_vol,
        "/results":           results_vol,
        "/root/.cache/torch": hub_vol,
        "/root/.cache/huggingface": hf_vol,
    },
    timeout=60 * 60 * 10,   # 10-hour hard cap
    secrets=[],
)
def train_vlm(
    llm_name: str = "microsoft/Phi-3.5-mini-instruct",
    encoder_name: str = "dinov2_vits14",
    s1_epochs: int = 3,
    s2_epochs: int = 10,
    batch_size: int = 4,
    s1_lr: float = 1e-3,
    s2_lr: float = 2e-5,
    lora_rank: int = 8,
    pool_patches: int = 2,
    input_size: int = 224,
    num_workers: int = 4,
) -> dict:
    """Run two-stage VLM training on Modal A10G."""
    import json
    import sys
    import time

    sys.path.insert(0, "/app")

    from core.encoders import create_encoder
    from core.decoders.vlm import VLMDecoder
    from core.data.vqa_dataset import PetsVQADataset
    from core.training.vlm_trainer import VLMTrainer

    ann_json = "/datasets/oxford_pets/annotations.json"
    img_dir  = "/datasets/oxford_pets/images"
    out_dir  = Path("/results/vlm_pets")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"VLM Training: DINOv2 + {llm_name.split('/')[-1]}")
    print(f"  Stage 1: {s1_epochs} epochs, lr={s1_lr}")
    print(f"  Stage 2: {s2_epochs} epochs, lr={s2_lr}, LoRA rank={lora_rank}")
    print("=" * 60)

    t0 = time.time()

    encoder = create_encoder(encoder_name, input_size=input_size)
    decoder = VLMDecoder(
        encoder,
        llm_name=llm_name,
        freeze_llm=True,
        lora_rank=0,
        pool_patches=pool_patches,
    )

    dataset = PetsVQADataset(
        annotations_json=ann_json,
        images_dir=img_dir,
        tokenizer=decoder.tokenizer,
        transform=encoder.get_transform(),
    )
    train_ds, val_ds = dataset.split()
    print(f"Dataset: {len(dataset)} total -> {len(train_ds)} train / {len(val_ds)} val")

    # Stage 1: projector alignment
    print("\nStage 1: Projector alignment (MLP only, LLM frozen)")
    print(f"  Trainable params: {decoder.num_trainable_params():,}")
    stage1 = VLMTrainer(
        decoder=decoder,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=s1_lr,
        epochs=s1_epochs,
        batch_size=batch_size,
        stage=1,
        checkpoint_dir=out_dir / "checkpoints" / "stage1",
        num_workers=num_workers,
        warmup_epochs=1,
        early_stopping_patience=0,
    )
    results1 = stage1.fit()
    print(f"  Stage 1 best val_loss: {results1['best_val_loss']:.4f}")

    # Stage 2: instruction tuning with LoRA
    print(f"\nStage 2: Instruction tuning (projector + LLM LoRA rank={lora_rank})")
    decoder.enable_llm_lora(rank=lora_rank)
    print(f"  Trainable params: {decoder.num_trainable_params():,}")
    stage2 = VLMTrainer(
        decoder=decoder,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=s2_lr,
        epochs=s2_epochs,
        batch_size=batch_size,
        stage=2,
        checkpoint_dir=out_dir / "checkpoints" / "stage2",
        num_workers=num_workers,
        warmup_epochs=2,
        early_stopping_patience=5,
    )
    results2 = stage2.fit()

    elapsed = time.time() - t0

    # Generation sample
    import torch
    print("\nGeneration sample (first val image):")
    try:
        sample = val_ds[0]
        image_t = sample["image"].unsqueeze(0)
        device = next(decoder.parameters()).device
        with torch.no_grad():
            features = encoder.forward_features(image_t.to(device))
        q = "What breed of animal is in this image?"
        enc = decoder.tokenizer(
            f"<|user|>\n{q}<|end|>\n<|assistant|>\n", return_tensors="pt"
        )
        answers = decoder.generate(
            features, enc["input_ids"].to(device),
            enc["attention_mask"].to(device), max_new_tokens=32,
        )
        print(f"  Q: {q}")
        print(f"  A: {answers[0]}")
    except Exception as exc:
        print(f"  (generation skipped: {exc})")

    results = {
        "stage1": results1,
        "stage2": results2,
        "elapsed_sec": elapsed,
        "llm_name": llm_name,
        "encoder_name": encoder_name,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    results_vol.commit()
    print(f"\nTotal time: {elapsed / 60:.1f} min")
    return results


# ── Dataset upload helper ─────────────────────────────────────────────────────
@app.local_entrypoint()
def upload_dataset():
    """Upload the Oxford Pets dataset to the fft-datasets Modal volume.

    Run once:
        modal run experiments/modal_train_vlm.py::upload_dataset
    """
    ds_dir = ROOT / "experiments" / "datasets" / "oxford_pets"
    if not (ds_dir / "annotations.json").exists():
        print("Dataset not found. Run first:")
        print("  python experiments/prepare_oxford_pets.py")
        return

    print(f"Uploading {ds_dir} to Modal volume 'fft-datasets/oxford_pets' ...")
    with dataset_vol.batch_upload(force=True) as batch:
        batch.put_directory(str(ds_dir), "/oxford_pets")
    print("Upload complete.")


# ── Main entrypoint ───────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    llm: str = "microsoft/Phi-3.5-mini-instruct",
    encoder: str = "dinov2_vits14",
    s1_epochs: int = 3,
    s2_epochs: int = 10,
    batch_size: int = 4,
    lora_rank: int = 8,
):
    """Submit VLM training to Modal.

    Examples:
        # Default (Phi-3.5-mini, vits14):
        modal run experiments/modal_train_vlm.py

        # Larger encoder:
        modal run experiments/modal_train_vlm.py -- --encoder dinov2_vitb14

        # Quick smoke-test (1+2 epochs):
        modal run experiments/modal_train_vlm.py -- --s1-epochs 1 --s2-epochs 2
    """
    print(f"Submitting VLM job to Modal A10G:")
    print(f"  LLM:     {llm}")
    print(f"  Encoder: {encoder}")
    print(f"  Stages:  {s1_epochs} + {s2_epochs} epochs, LoRA rank={lora_rank}")

    result = train_vlm.remote(
        llm_name=llm,
        encoder_name=encoder,
        s1_epochs=s1_epochs,
        s2_epochs=s2_epochs,
        batch_size=batch_size,
        lora_rank=lora_rank,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Stage 1 best val_loss:  {result['stage1']['best_val_loss']:.4f}")
    print(f"Stage 2 best val_loss:  {result['stage2']['best_val_loss']:.4f}")
    print(f"Stage 2 val_token_acc:  {result['stage2'].get('val_token_acc', float('nan')):.4f}")
    print(f"Total time:             {result['elapsed_sec'] / 60:.1f} min")
    print("\nResults saved to Modal volume 'fft-results/vlm_pets/'")
    print("Download with:")
    print("  modal volume get fft-results /vlm_pets ./experiments/results_modal/vlm_pets")
