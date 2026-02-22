"""Fast benchmarks on filtered COCO datasets — 50 epochs, batch 16, best combo only.

Default: vitb14+rtdetr (new default decoder) on both datasets.
--all-combos: runs rtdetr, detr_lite, and detr_multiscale for both encoders.
Estimated time: ~2-3 hours default (vs. ~8-10 hours for full suite).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_experiment(dataset_path: Path, dataset_name: str, encoder_name: str,
                   decoder_type: str, num_classes: int,
                   epochs: int = 50, batch_size: int = 16, lr: float = 1e-4):
    """Fast detection experiment — 50 epochs, batch 16, patience 20."""
    from core.encoders import create_encoder
    from core.data.dataset import FFTDataset
    from core.training.trainer import Trainer
    from core.cli import _create_decoder

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Encoder: {encoder_name}  |  Decoder: {decoder_type}")
    print(f"  Classes: {num_classes}  |  Epochs: {epochs}  |  Batch: {batch_size}")
    print(f"{'='*60}")

    # Encoder
    t0 = time.time()
    encoder = create_encoder(encoder_name)
    # detr_multiscale/fpn need intermediate layers pre-set; rtdetr sets them itself
    if decoder_type in ["detr_multiscale", "fpn"]:
        encoder.intermediate_layers = encoder.default_intermediate_layers()
    encoder_load_time = time.time() - t0

    # Decoder
    decoder = _create_decoder(decoder_type, "detection", encoder, num_classes)
    num_params = decoder.num_trainable_params()

    # Dataset
    dataset = FFTDataset.from_folder(
        dataset_path, task="detection", transform=encoder.get_transform()
    )

    # Results dir
    output_dir = (
        Path(__file__).parent / "results" / f"{dataset_name}_fast" / encoder_name / decoder_type
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        decoder=decoder,
        train_dataset=dataset,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        scheduler="cosine",
        warmup_epochs=5,              # Shorter warmup for faster runs
        augmentation="light",
        early_stopping_patience=20,   # Stop sooner (was 30)
        checkpoint_dir=output_dir / "checkpoints",
    )

    train_start = time.time()
    results = trainer.fit()
    train_time = time.time() - train_start

    results_dict = {
        "dataset": dataset_name,
        "encoder": encoder_name,
        "decoder": decoder_type,
        "task": "detection",
        "num_classes": num_classes,
        "num_train_samples": len(dataset),
        "num_params": num_params,
        "epochs_trained": results["epochs_trained"],
        "batch_size": batch_size,
        "lr": lr,
        "encoder_load_time": encoder_load_time,
        "train_time": train_time,
        "train_time_per_epoch": train_time / max(results["epochs_trained"], 1),
        **results,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    trainer.save(output_dir / "decoder_weights.pt")

    map50 = results.get("val_map50", "N/A")
    mmap  = results.get("val_map",   "N/A")
    map50_str = f"{map50:.4f}" if isinstance(map50, float) else map50
    mmap_str  = f"{mmap:.4f}"  if isinstance(mmap,  float) else mmap

    print(f"\nDone:  epochs={results['epochs_trained']}  "
          f"val_loss={results['best_val_loss']:.4f}  "
          f"mAP@50={map50_str}  mAP@50:95={mmap_str}  "
          f"time={train_time/3600:.2f}h")

    return results_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--all-combos", action="store_true",
                        help="Run all 4 encoder/decoder combos (slower)")
    args = parser.parse_args()

    datasets_dir = Path(__file__).parent / "datasets"

    # Use existing prepared datasets
    coco15_dir = datasets_dir / "coco_15cls"
    coco20_dir = datasets_dir / "coco_20cls"

    datasets = [
        ("coco_15cls", coco15_dir),
        ("coco_20cls", coco20_dir),
    ]

    # Best combo only by default; optionally run all
    if args.all_combos:
        encoder_decoder_pairs = [
            ("dinov2_vits14", "rtdetr"),
            #("dinov2_vits14", "detr_lite"),
            ("dinov2_vits14", "detr_multiscale"),
            ("dinov2_vitb14", "rtdetr"),
            #("dinov2_vitb14", "detr_lite"),
            ("dinov2_vitb14", "detr_multiscale"),
        ]
    else:
        encoder_decoder_pairs = [
            ("dinov2_vitb14", "rtdetr"),  # New default decoder
        ]

    all_results = []
    total = len(datasets) * len(encoder_decoder_pairs)
    run_idx = 0

    print("\n" + "=" * 60)
    print("FAST DETECTION BENCHMARKS")
    print(f"  {len(datasets)} datasets × {len(encoder_decoder_pairs)} combo(s) = {total} experiments")
    print(f"  Config: {args.epochs} epochs, batch {args.batch_size}, patience 20")
    print("=" * 60)

    for ds_name, ds_path in datasets:
        if not (ds_path / "annotations.json").exists():
            print(f"Dataset not found: {ds_path} — skipping")
            continue

        with open(ds_path / "annotations.json") as f:
            num_classes = len(json.load(f)["categories"])

        for encoder, decoder in encoder_decoder_pairs:
            run_idx += 1
            print(f"\n[{run_idx}/{total}]")
            try:
                r = run_experiment(
                    ds_path, ds_name, encoder, decoder, num_classes,
                    epochs=args.epochs, batch_size=args.batch_size,
                )
                all_results.append(r)
            except Exception as e:
                print(f"ERROR: {ds_name} + {encoder} + {decoder}: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<16} {'Encoder':<16} {'Decoder':<18} {'mAP@50':>8} {'mAP:95':>8} {'Ep':>4}")
    print("-" * 72)
    for r in all_results:
        map50 = r.get("val_map50", float("nan"))
        mmap  = r.get("val_map",   float("nan"))
        print(f"{r['dataset']:<16} {r['encoder']:<16} {r['decoder']:<18} "
              f"{map50:>8.4f} {mmap:>8.4f} {r['epochs_trained']:>4}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
