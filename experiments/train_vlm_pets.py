"""VLM fine-tuning: DINOv2 + MLP projector + Phi-3.5-mini on Oxford Pets.

Two-stage training:
  Stage 1 (3 epochs)  — align projector only (LLM frozen), lr=1e-3
  Stage 2 (10 epochs) — instruction tuning with LoRA on LLM, lr=2e-5

Prerequisites:
  pip install transformers accelerate sentencepiece
  python experiments/prepare_oxford_pets.py   # download + prep dataset
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent / "datasets" / "oxford_pets"
OUT_DIR  = Path(__file__).parent / "results" / "vlm_pets"
WORKERS  = 0

# Stage 1 hyperparams
S1_EPOCHS = 3
S1_LR     = 1e-3
S1_BATCH  = 4

# Stage 2 hyperparams
S2_EPOCHS    = 10
S2_LR        = 2e-5
S2_BATCH     = 4
S2_LORA_RANK = 8


def main() -> None:
    from core.encoders import create_encoder
    from core.decoders.vlm import VLMDecoder
    from core.data.vqa_dataset import PetsVQADataset
    from core.training.vlm_trainer import VLMTrainer

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VLM Fine-tuning: DINOv2 + Phi-3.5-mini on Oxford Pets")
    print(f"  Stage 1: projector alignment  ({S1_EPOCHS} epochs, lr={S1_LR})")
    print(f"  Stage 2: instruction tuning   ({S2_EPOCHS} epochs, lr={S2_LR}, LoRA rank={S2_LORA_RANK})")
    print("=" * 60)

    t0 = time.time()

    # ── Build encoder + decoder ───────────────────────────────────────────────
    encoder = create_encoder("dinov2_vits14", input_size=224)
    decoder = VLMDecoder(encoder, freeze_llm=True, lora_rank=0, pool_patches=2)

    print(f"\nEncoder:           dinov2_vits14 (embed_dim={encoder.embed_dim})")
    print(f"Visual tokens:     {decoder.num_visual_tokens}  (16×16 → 8×8 after 2×2 pool)")
    print(f"Projector params:  {sum(p.numel() for p in decoder.projector.parameters()):,}")
    print(f"LLM total params:  {sum(p.numel() for p in decoder.llm.parameters()):,}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    ann_json = DATA_DIR / "annotations.json"
    img_dir  = DATA_DIR / "images"

    dataset = PetsVQADataset(
        annotations_json=ann_json,
        images_dir=img_dir,
        tokenizer=decoder.tokenizer,
        transform=encoder.get_transform(),
    )
    train_ds, val_ds = dataset.split()

    print(f"\nDataset: {len(dataset)} total → {len(train_ds)} train / {len(val_ds)} val")

    # ── Stage 1: Projector alignment ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Stage 1: Projector alignment  (MLP only, LLM frozen)")
    print("─" * 60)
    print(f"  Trainable params: {decoder.num_trainable_params():,}")

    stage1 = VLMTrainer(
        decoder=decoder,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=S1_LR,
        epochs=S1_EPOCHS,
        batch_size=S1_BATCH,
        stage=1,
        checkpoint_dir=OUT_DIR / "checkpoints" / "stage1",
        num_workers=WORKERS,
        warmup_epochs=1,
        early_stopping_patience=0,
    )
    results1 = stage1.fit()

    print(f"\n  Stage 1 best val_loss: {results1['best_val_loss']:.4f}")

    # ── Stage 2: Instruction tuning with LoRA ─────────────────────────────────
    print("\n" + "─" * 60)
    print(f"Stage 2: Instruction tuning  (projector + LLM LoRA rank={S2_LORA_RANK})")
    print("─" * 60)

    decoder.enable_llm_lora(rank=S2_LORA_RANK)
    print(f"  Trainable params: {decoder.num_trainable_params():,}")

    stage2 = VLMTrainer(
        decoder=decoder,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=S2_LR,
        epochs=S2_EPOCHS,
        batch_size=S2_BATCH,
        stage=2,
        checkpoint_dir=OUT_DIR / "checkpoints" / "stage2",
        num_workers=WORKERS,
        warmup_epochs=2,
        early_stopping_patience=5,
    )
    results2 = stage2.fit()

    elapsed = time.time() - t0

    # ── Quick generation sample ───────────────────────────────────────────────
    print("\nGeneration sample (first val image):")
    try:
        import torch
        sample = val_ds[0]
        image = sample["image"].unsqueeze(0)
        device = next(decoder.parameters()).device
        image = image.to(device)

        with torch.no_grad():
            features = encoder.forward_features(image)

        question = "What breed of animal is in this image?"
        q_text = f"<|user|>\n{question}\n<|end|>\n<|assistant|>\n"
        enc = decoder.tokenizer(q_text, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        answers = decoder.generate(features, ids, mask, max_new_tokens=32)
        print(f"  Q: {question}")
        print(f"  A: {answers[0]}")
    except Exception as exc:
        print(f"  (generation skipped: {exc})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("Stage 1 (alignment):")
    print(f"  best_val_loss:  {results1['best_val_loss']:.4f}")
    print(f"  epochs_trained: {results1['epochs_trained']}")
    print("Stage 2 (instruction tuning):")
    print(f"  best_val_loss:  {results2['best_val_loss']:.4f}")
    print(f"  val_token_acc:  {results2.get('val_token_acc', float('nan')):.4f}")
    print(f"  epochs_trained: {results2['epochs_trained']}")
    print(f"  Total time:     {elapsed / 60:.1f} min")
    print("=" * 60)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(
            {"stage1": results1, "stage2": results2, "elapsed_sec": elapsed},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
