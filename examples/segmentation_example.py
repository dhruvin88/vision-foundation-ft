#!/usr/bin/env python3
"""Semantic segmentation fine-tuning example.

Fine-tune a segmentation head on DINOv2 features.

Dataset structure:
    my_dataset/
        images/
            img001.jpg
            img002.jpg
        masks/
            img001.png   (class index per pixel, 0-indexed)
            img002.png

Usage:
    python examples/segmentation_example.py --data ./my_dataset --num-classes 3
"""

import argparse
import sys
sys.path.insert(0, ".")

import sdk as fft


def main():
    parser = argparse.ArgumentParser(description="Segmentation fine-tuning example")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--encoder", type=str, default="dinov2_vitb14", help="Encoder variant")
    parser.add_argument(
        "--head", type=str, default="linear",
        choices=["linear", "upernet", "mask_transformer"],
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--output", type=str, default="./output_seg", help="Output directory")
    args = parser.parse_args()

    # 1. Load encoder
    print(f"Loading encoder: {args.encoder}")
    encoder = fft.Encoder(args.encoder)

    # 2. Create segmentation head
    print(f"Creating {args.head} segmentation head for {args.num_classes} classes")
    decoder = fft.SegmentationHead(
        encoder,
        num_classes=args.num_classes,
        head_type=args.head,
    )
    print(f"Decoder has {decoder.num_trainable_params():,} trainable parameters")

    # 3. Load dataset
    print(f"Loading dataset from: {args.data}")
    dataset = fft.Dataset.from_folder(
        args.data, task="segmentation", transform=encoder.get_transform()
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
    weights_path = f"{args.output}/segmentation_weights.pt"
    trainer.save(weights_path)
    print(f"Saved weights to: {weights_path}")

    # 6. Run inference on a test image
    print("\nTo run inference:")
    print(f"  from sdk import Encoder, SegmentationHead, run_inference, load_decoder_weights")
    print(f"  encoder = Encoder('{args.encoder}')")
    print(f"  decoder = SegmentationHead(encoder, num_classes={args.num_classes}, head_type='{args.head}')")
    print(f"  load_decoder_weights(decoder, '{weights_path}')")
    print(f"  results = run_inference(decoder, 'path/to/test_image.jpg')")


if __name__ == "__main__":
    main()
