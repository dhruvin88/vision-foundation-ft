"""Run DINOv2-only benchmarks comparing encoders and decoders."""

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

    # Load dataset (classification expects class folders directly, so use train/ subdirectory)
    dataset = FFTDataset.from_folder(
        dataset_path / "train", task="classification", transform=encoder.get_transform()
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

    print(f"\nComplete!")
    print(f"  Train time: {train_time:.1f}s ({train_time/epochs:.1f}s/epoch)")
    best_val_loss = results.get('best_val_loss', 'N/A')
    val_loss_str = f"{best_val_loss:.4f}" if isinstance(best_val_loss, (int, float)) else best_val_loss
    print(f"  Best val loss: {val_loss_str}")
    val_acc = results.get('val_acc', 'N/A')
    val_acc_str = f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else val_acc
    print(f"  Val accuracy: {val_acc_str}")
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

    # Multi-scale decoders need intermediate layer features pre-set;
    # RTDETRDecoder sets its own intermediate_layers in __init__
    if decoder_type in ["fpn", "detr_multiscale"]:
        encoder.intermediate_layers = encoder.default_intermediate_layers()

    encoder_load_time = time.time() - start_time

    # Create decoder
    decoder = _create_decoder(decoder_type, "detection", encoder, num_classes)
    num_params = decoder.num_trainable_params()

    # Load dataset (from_folder expects root with annotations.json and images/)
    dataset = FFTDataset.from_folder(
        dataset_path,
        task="detection",
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

    print(f"\nComplete!")
    print(f"  Train time: {train_time:.1f}s ({train_time/epochs:.1f}s/epoch)")
    best_val_loss = results.get('best_val_loss', 'N/A')
    val_loss_str = f"{best_val_loss:.4f}" if isinstance(best_val_loss, (int, float)) else best_val_loss
    print(f"  Best val loss: {val_loss_str}")
    val_map50 = results.get('val_map50', 'N/A')
    val_map50_str = f"{val_map50:.4f}" if isinstance(val_map50, (int, float)) else val_map50
    val_map = results.get('val_map', 'N/A')
    val_map_str = f"{val_map:.4f}" if isinstance(val_map, (int, float)) else val_map
    print(f"  Val mAP@50: {val_map50_str}  mAP@50:95: {val_map_str}")
    print(f"  Params: {num_params:,}")

    return results_dict


def main():
    # Use absolute paths
    base_dir = Path(__file__).parent.parent
    datasets_dir = base_dir / "experiments" / "datasets"
    results_dir = base_dir / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print("\n" + "="*60)
    print("DINOV2 BENCHMARK SUITE")
    print("="*60)

    # DINOv2 encoders only (skip DINOv3 - requires HF auth)
    encoders = [#"dinov2_vits14",
                "dinov2_vitb14"]
    cls_decoders = ["linear_probe", "mlp"]
    det_decoders = ["rtdetr", "detr_lite", "detr_multiscale"]

    # CIFAR-10 experiments
    
    print("\n[1/3] CIFAR-10 Classification")
    cifar_path = datasets_dir / "cifar10_subset"
    if (cifar_path / "train").exists():
        for encoder in encoders:
            for decoder in cls_decoders:
                try:
                    result = run_classification_experiment(
                        cifar_path,
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
                    import traceback
                    traceback.print_exc()
    else:
        print(f"  CIFAR-10 dataset not found at {cifar_path}")
    
    # Flowers102 experiments
    
    print("\n[2/3] Flowers102 Classification")
    flowers_path = datasets_dir / "flowers102_subset"
    if (flowers_path / "train").exists():
        for encoder in encoders:
            for decoder in cls_decoders:
                try:
                    result = run_classification_experiment(
                        flowers_path,
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
                    import traceback
                    traceback.print_exc()
    else:
        print(f"  Flowers102 dataset not found at {flowers_path}")
    
    # Detection experiments
    print("\n[3/3] COCO Detection")
    coco_path = datasets_dir / "coco_detection_subset"

    if (coco_path / "annotations.json").exists():
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
                    import traceback
                    traceback.print_exc()
    else:
        print(f"  COCO dataset not found at {coco_path}")

    # Save all results
    with open(results_dir / "dinov2_results.json", 'w') as f:
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
    summary_lines.append("# DINOv2 Benchmark Results\n")
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

    summary_lines.append("\n## Notes\n")
    summary_lines.append("- DINOv2 encoders: patch_size=14, default input 518x518")
    summary_lines.append("- FPN decoder uses multi-scale features (intermediate layers)")
    summary_lines.append("- Training times include encoder loading and full train+val loops")
    summary_lines.append("- DINOv3 results will be added after HuggingFace authentication\n")

    summary_path = output_dir / "DINOV2_SUMMARY.md"
    summary_path.write_text('\n'.join(summary_lines))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
