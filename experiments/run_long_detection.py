"""Long training run for detection models — 100 epochs to observe convergence."""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_long_detection(coco_path, encoder_name, decoder_type, num_classes,
                       epochs=100, batch_size=8, lr=1e-4):
    from core.encoders import create_encoder
    from core.data.dataset import FFTDataset
    from core.training.trainer import Trainer
    from core.cli import _create_decoder

    print(f"\n{'='*60}")
    print(f"Detection (100 ep): {encoder_name} + {decoder_type}")
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
        coco_path, task="detection", transform=encoder.get_transform()
    )

    # Results dir (separate from 20-epoch runs)
    output_dir = (Path(__file__).parent / "results" / "coco_subset_100ep"
                  / encoder_name / decoder_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        decoder=decoder,
        train_dataset=dataset,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        scheduler="cosine",
        warmup_epochs=10,          # longer warmup for longer run
        augmentation="light",
        early_stopping_patience=30,  # generous — don't quit too early
        checkpoint_dir=output_dir / "checkpoints",
    )

    train_start = time.time()
    results = trainer.fit()
    train_time = time.time() - train_start

    results_dict = {
        "dataset": "coco_subset_100ep",
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

    print(f"\nComplete!")
    print(f"  Epochs trained:   {results['epochs_trained']}")
    print(f"  Train time:       {train_time/3600:.2f}h  ({train_time/results['epochs_trained']:.1f}s/epoch)")
    print(f"  Best val loss:    {results['best_val_loss']:.4f}")
    print(f"  mAP@50:           {map50_str}")
    print(f"  mAP@50:95:        {mmap_str}")
    print(f"  Trainable params: {num_params:,}")

    return results_dict


def main():
    base_dir = Path(__file__).parent.parent
    coco_path = base_dir / "experiments" / "datasets" / "coco_detection_subset"

    if not (coco_path / "annotations.json").exists():
        print(f"COCO dataset not found at {coco_path}")
        return

    with open(coco_path / "annotations.json") as f:
        num_classes = len(json.load(f)["categories"])

    experiments = [
        ("dinov2_vits14", "rtdetr"),
        ("dinov2_vitb14", "rtdetr"),
        ("dinov2_vits14", "detr_lite"),
        ("dinov2_vitb14", "detr_lite"),
    ]

    print("\n" + "=" * 60)
    print("LONG TRAINING RUN — 100 epochs, patience=30")
    print("=" * 60)
    print(f"Dataset: {coco_path.name}  |  Classes: {num_classes}")
    print(f"Experiments: {len(experiments)}")
    # Rough time estimate
    approx_times = {"rtdetr": 120, "detr_lite": 70, "detr_multiscale": 100}  # s/epoch avg
    total_s = sum(approx_times.get(d, 90) * 100 for _, d in experiments)
    print(f"Est. total time: ~{total_s/3600:.1f}h")
    print("=" * 60)

    all_results = []
    for i, (encoder, decoder) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}]")
        try:
            r = run_long_detection(coco_path, encoder, decoder, num_classes)
            all_results.append(r)
        except Exception as e:
            print(f"ERROR: {encoder} + {decoder} failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Encoder':<16} {'Decoder':<18} {'mAP@50':>8} {'mAP@50:95':>10} {'Epochs':>7}")
    print("-" * 62)
    for r in all_results:
        map50 = r.get("val_map50", float("nan"))
        mmap  = r.get("val_map",   float("nan"))
        print(f"{r['encoder']:<16} {r['decoder']:<18} "
              f"{map50:>8.4f} {mmap:>10.4f} {r['epochs_trained']:>7}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
