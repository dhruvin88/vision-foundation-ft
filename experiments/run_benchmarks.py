"""Run comprehensive benchmarks comparing encoders and decoders."""

import json
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np


def run_classification_experiment(
    dataset_path,
    dataset_name,
    encoder_name,
    decoder_type,
    num_classes,
    epochs=20,
    batch_size=32,
    lr=1e-3
):
    """Run a single classification experiment."""
    from core.encoders import create_encoder
    from core.data.dataset import FFTDataset
    from core.training.trainer import Trainer
    from core.cli import _create_decoder

    print(f"\n{'='*60}")
    print(f"Classification: {dataset_name}")
    print(f"Encoder: {encoder_name}, Decoder: {decoder_type}")
    print(f"{'='*60}")

    # Load encoder
    start_time = time.time()
    encoder = create_encoder(encoder_name)
    encoder_load_time = time.time() - start_time

    # Create decoder
    decoder = _create_decoder(decoder_type, "classification", encoder, num_classes)
    num_params = decoder.num_trainable_params()

    # Load dataset
    dataset = FFTDataset.from_folder(
        dataset_path, task="classification", transform=encoder.get_transform()
    )

    # Train
    output_dir = Path("experiments/results") / dataset_name / encoder_name / decoder_type
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        decoder=decoder,
        train_dataset=dataset,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        scheduler="cosine",
        warmup_epochs=3,
        augmentation="light",
        early_stopping_patience=10,
        checkpoint_dir=output_dir / "checkpoints",
    )

    train_start = time.time()
    results = trainer.fit()
    train_time = time.time() - train_start

    # Save results
    results_dict = {
        "dataset": dataset_name,
        "encoder": encoder_name,
        "decoder": decoder_type,
        "task": "classification",
        "num_classes": num_classes,
        "num_train_samples": len(dataset),
        "num_params": num_params,
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "encoder_load_time": encoder_load_time,
        "train_time": train_time,
        "train_time_per_epoch": train_time / epochs,
        **results,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Save weights
    trainer.save(output_dir / "decoder_weights.pt")

    print(f"\n✓ Complete!")
    print(f"  Train time: {train_time:.1f}s ({train_time/epochs:.1f}s/epoch)")
    print(f"  Best val loss: {results.get('best_val_loss', 'N/A'):.4f}")
    print(f"  Val accuracy: {results.get('val_accuracy', 'N/A'):.4f}")
    print(f"  Params: {num_params:,}")

    return results_dict


def run_detection_experiment(
    dataset_path,
    dataset_name,
    encoder_name,
    decoder_type,
    num_classes,
    epochs=20,
    batch_size=8,
    lr=1e-4
):
    """Run a single detection experiment."""
    from core.encoders import create_encoder
    from core.data.dataset import FFTDataset
    from core.training.trainer import Trainer
    from core.cli import _create_decoder

    print(f"\n{'='*60}")
    print(f"Detection: {dataset_name}")
    print(f"Encoder: {encoder_name}, Decoder: {decoder_type}")
    print(f"{'='*60}")

    # Load encoder
    start_time = time.time()
    encoder = create_encoder(encoder_name)

    # FPN needs multi-scale features pre-set; RTDETRDecoder sets them itself
    if decoder_type == "fpn":
        encoder.intermediate_layers = encoder.default_intermediate_layers()

    encoder_load_time = time.time() - start_time

    # Create decoder
    decoder = _create_decoder(decoder_type, "detection", encoder, num_classes)
    num_params = decoder.num_trainable_params()

    # Load dataset
    dataset = FFTDataset.from_coco(
        dataset_path / "annotations.json",
        dataset_path / "images",
        transform=encoder.get_transform()
    )

    # Train
    output_dir = Path("experiments/results") / dataset_name / encoder_name / decoder_type
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        decoder=decoder,
        train_dataset=dataset,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        scheduler="cosine",
        warmup_epochs=5,
        augmentation="light",
        early_stopping_patience=15,
        checkpoint_dir=output_dir / "checkpoints",
    )

    train_start = time.time()
    results = trainer.fit()
    train_time = time.time() - train_start

    # Save results
    results_dict = {
        "dataset": dataset_name,
        "encoder": encoder_name,
        "decoder": decoder_type,
        "task": "detection",
        "num_classes": num_classes,
        "num_train_samples": len(dataset),
        "num_params": num_params,
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "encoder_load_time": encoder_load_time,
        "train_time": train_time,
        "train_time_per_epoch": train_time / epochs,
        **results,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Save weights
    trainer.save(output_dir / "decoder_weights.pt")

    print(f"\n✓ Complete!")
    print(f"  Train time: {train_time:.1f}s ({train_time/epochs:.1f}s/epoch)")
    print(f"  Best val loss: {results.get('best_val_loss', 'N/A'):.4f}")
    print(f"  Val mAP: {results.get('val_map', results.get('val_f1', 'N/A'))}")
    print(f"  Params: {num_params:,}")

    return results_dict


def main():
    datasets_dir = Path("experiments/datasets")
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print("\n" + "="*60)
    print("BENCHMARK SUITE: ENCODER & DECODER COMPARISON")
    print("="*60)

    # Classification experiments
    encoders = ["dinov2_vits14", "dinov2_vitb14", "dinov3_vits16", "dinov3_vitb16"]
    cls_decoders = ["linear_probe", "mlp"]

    # CIFAR-10 experiments
    cifar_path = datasets_dir / "cifar10_subset" / "train"
    if cifar_path.exists():
        for encoder in encoders:
            for decoder in cls_decoders:
                try:
                    result = run_classification_experiment(
                        cifar_path.parent,
                        "cifar10",
                        encoder,
                        decoder,
                        num_classes=10,
                        epochs=20,
                        batch_size=32,
                        lr=1e-3
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR: {encoder} + {decoder} on CIFAR-10 failed: {e}")

    # Flowers102 experiments
    flowers_path = datasets_dir / "flowers102_subset" / "train"
    if flowers_path.exists():
        for encoder in encoders:
            for decoder in cls_decoders:
                try:
                    result = run_classification_experiment(
                        flowers_path.parent,
                        "flowers102",
                        encoder,
                        decoder,
                        num_classes=20,
                        epochs=25,
                        batch_size=32,
                        lr=1e-3
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR: {encoder} + {decoder} on Flowers102 failed: {e}")

    # Detection experiments
    det_decoders = ["rtdetr", "detr_lite", "fpn"]
    coco_path = datasets_dir / "coco_detection_subset"

    if coco_path.exists():
        # Determine number of classes from COCO annotations
        with open(coco_path / "annotations.json") as f:
            coco_data = json.load(f)
            num_classes = len(coco_data['categories'])

        for encoder in encoders:
            for decoder in det_decoders:
                try:
                    result = run_detection_experiment(
                        coco_path,
                        "coco_subset",
                        encoder,
                        decoder,
                        num_classes=num_classes,
                        epochs=20,
                        batch_size=8,
                        lr=1e-4
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR: {encoder} + {decoder} on COCO failed: {e}")

    # Save all results
    with open(results_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("ALL BENCHMARKS COMPLETE!")
    print(f"Results saved to: {results_dir}")
    print("="*60)

    # Generate summary
    generate_summary(all_results, results_dir)


def generate_summary(results, output_dir):
    """Generate a summary markdown table."""
    summary_lines = []
    summary_lines.append("# Benchmark Results Summary\n")
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Classification results
    summary_lines.append("\n## Classification Results\n")
    summary_lines.append("| Dataset | Encoder | Decoder | Accuracy | Val Loss | Params | Train Time | Time/Epoch |")
    summary_lines.append("|---------|---------|---------|----------|----------|--------|------------|------------|")

    cls_results = [r for r in results if r['task'] == 'classification']
    for r in sorted(cls_results, key=lambda x: (x['dataset'], x['encoder'], x['decoder'])):
        summary_lines.append(
            f"| {r['dataset']} | {r['encoder']} | {r['decoder']} | "
            f"{r.get('val_accuracy', 0):.4f} | {r.get('best_val_loss', 0):.4f} | "
            f"{r['num_params']:,} | {r['train_time']:.0f}s | {r['train_time_per_epoch']:.1f}s |"
        )

    # Detection results
    summary_lines.append("\n## Detection Results\n")
    summary_lines.append("| Dataset | Encoder | Decoder | mAP/F1 | Val Loss | Params | Train Time | Time/Epoch |")
    summary_lines.append("|---------|---------|---------|--------|----------|--------|------------|------------|")

    det_results = [r for r in results if r['task'] == 'detection']
    for r in sorted(det_results, key=lambda x: (x['dataset'], x['encoder'], x['decoder'])):
        metric = r.get('val_map', r.get('val_f1', 0))
        summary_lines.append(
            f"| {r['dataset']} | {r['encoder']} | {r['decoder']} | "
            f"{metric:.4f} | {r.get('best_val_loss', 0):.4f} | "
            f"{r['num_params']:,} | {r['train_time']:.0f}s | {r['train_time_per_epoch']:.1f}s |"
        )

    summary_lines.append("\n## Key Observations\n")
    summary_lines.append("- DINOv3 encoders use patch_size=16 (default 512×512 input)")
    summary_lines.append("- DINOv2 encoders use patch_size=14 (default 518×518 input)")
    summary_lines.append("- FPN decoder requires multi-scale features (intermediate layers)")
    summary_lines.append("- Training times include encoder loading and full train+val loops\n")

    summary_path = output_dir / "SUMMARY.md"
    summary_path.write_text('\n'.join(summary_lines))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
