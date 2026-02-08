#!/usr/bin/env python3
"""Classification fine-tuning example.

Fine-tune a classification head on DINOv2 features using a folder of images
organized by class (one subfolder per class).

Dataset structure:
    my_dataset/
        cats/
            img001.jpg
            img002.jpg
        dogs/
            img001.jpg
            img002.jpg

Usage:
    python examples/classification_example.py --data ./my_dataset --num-classes 2
"""

import argparse
import sys
sys.path.insert(0, ".")

import sdk as fft


def main():
    parser = argparse.ArgumentParser(description="Classification fine-tuning example")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--encoder", type=str, default="dinov2_vitb14", help="Encoder variant")
    parser.add_argument("--head", type=str, default="linear", choices=["linear", "mlp", "transformer"])
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output", type=str, default="./output_cls", help="Output directory")
    args = parser.parse_args()

    # 1. Load encoder (frozen automatically)
    print(f"Loading encoder: {args.encoder}")
    encoder = fft.Encoder(args.encoder)

    # 2. Create classification head
    print(f"Creating {args.head} classification head for {args.num_classes} classes")
    decoder = fft.ClassificationHead(encoder, num_classes=args.num_classes, head_type=args.head)
    print(f"Decoder has {decoder.num_trainable_params():,} trainable parameters")

    # 3. Load dataset
    print(f"Loading dataset from: {args.data}")
    dataset = fft.Dataset.from_folder(args.data, task="classification", transform=encoder.get_transform())
    print(f"Dataset: {len(dataset)} images, stats: {dataset.get_stats()}")

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

    # 5. Save model
    weights_path = f"{args.output}/classification_weights.pt"
    trainer.save(weights_path)
    print(f"Saved weights to: {weights_path}")

    # 6. Generate inference script
    fft.generate_inference_script(
        decoder_class=type(decoder).__name__,
        task="classification",
        encoder_name=args.encoder,
        num_classes=args.num_classes,
        weights_path=weights_path,
        output_path=f"{args.output}/inference.py",
    )
    print(f"Generated inference script at: {args.output}/inference.py")


if __name__ == "__main__":
    main()
