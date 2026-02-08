#!/usr/bin/env python3
"""Object detection fine-tuning example.

Fine-tune a detection head on DINOv2 features using COCO-format annotations.

Dataset structure:
    my_dataset/
        images/
            img001.jpg
            img002.jpg
        annotations.json   (COCO format)

Usage:
    python examples/detection_example.py --data ./my_dataset --num-classes 5
"""

import argparse
import sys
sys.path.insert(0, ".")

import sdk as fft


def main():
    parser = argparse.ArgumentParser(description="Detection fine-tuning example")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--encoder", type=str, default="dinov2_vitb14", help="Encoder variant")
    parser.add_argument("--head", type=str, default="detr_lite", choices=["detr_lite", "fpn"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of object queries (DETR)")
    parser.add_argument("--output", type=str, default="./output_det", help="Output directory")
    args = parser.parse_args()

    # 1. Load encoder
    print(f"Loading encoder: {args.encoder}")
    encoder = fft.Encoder(args.encoder)

    # 2. Create detection head
    print(f"Creating {args.head} detection head for {args.num_classes} classes")
    decoder = fft.DetectionHead(
        encoder,
        num_classes=args.num_classes,
        head_type=args.head,
    )
    print(f"Decoder has {decoder.num_trainable_params():,} trainable parameters")

    # 3. Load dataset (COCO format)
    print(f"Loading dataset from: {args.data}")
    dataset = fft.Dataset.from_folder(
        args.data, task="detection", transform=encoder.get_transform()
    )
    print(f"Dataset: {len(dataset)} images")

    # 4. Train
    trainer = fft.Trainer(
        decoder=decoder,
        train_dataset=dataset,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=f"{args.output}/checkpoints",
    )

    results = trainer.fit()
    print(f"\nTraining Results: {results}")

    # 5. Save
    weights_path = f"{args.output}/detection_weights.pt"
    trainer.save(weights_path)
    print(f"Saved weights to: {weights_path}")


if __name__ == "__main__":
    main()
