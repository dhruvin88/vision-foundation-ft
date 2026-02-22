"""Command-line interface for Foundation Model Fine-Tuning."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fft",
        description="Foundation Model Fine-Tuning CLI",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a decoder head")
    train_parser.add_argument("--data", required=True, help="Path to dataset directory")
    train_parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "detection", "segmentation"],
        help="Task type",
    )
    train_parser.add_argument(
        "--encoder",
        default="dinov3_vitb16",
        help="Encoder model name (default: dinov3_vitb16)",
    )
    train_parser.add_argument(
        "--decoder",
        default="auto",
        help="Decoder head type (default: auto-select based on task)",
    )
    train_parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument(
        "--augmentation",
        default="light",
        choices=["none", "light", "heavy"],
        help="Augmentation preset",
    )
    train_parser.add_argument("--output", default="./output", help="Output directory")
    train_parser.add_argument("--scheduler", default="cosine", help="LR scheduler type")
    train_parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    train_parser.add_argument(
        "--early-stopping", type=int, default=10, help="Early stopping patience (0 to disable)"
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference with a trained model")
    predict_parser.add_argument("--image", required=True, help="Path to image or directory")
    predict_parser.add_argument("--weights", required=True, help="Path to decoder weights (.pt)")
    predict_parser.add_argument("--encoder", default="dinov3_vitb16", help="Encoder model name")
    predict_parser.add_argument("--decoder", required=True, help="Decoder class name")
    predict_parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
    predict_parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "detection", "segmentation"],
    )

    # Info command
    subparsers.add_parser("info", help="Show available encoders and decoders")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "train":
        _run_train(args)
    elif args.command == "predict":
        _run_predict(args)
    elif args.command == "info":
        _run_info()
    else:
        parser.print_help()
        sys.exit(1)


def _run_train(args: argparse.Namespace) -> None:
    """Execute training from CLI args."""
    from core.encoders import create_encoder
    from core.data.dataset import FFTDataset
    from core.training.trainer import Trainer

    # Load encoder
    print(f"Loading encoder: {args.encoder}")
    encoder = create_encoder(args.encoder)

    # Create decoder
    decoder = _create_decoder(args.decoder, args.task, encoder, args.num_classes)
    print(f"Created decoder: {type(decoder).__name__} ({decoder.num_trainable_params()} params)")

    # Load dataset
    print(f"Loading dataset from: {args.data}")
    dataset = FFTDataset.from_folder(
        args.data, task=args.task, transform=encoder.get_transform()
    )
    print(f"Dataset: {len(dataset)} samples")

    # Train
    output_dir = Path(args.output)
    trainer = Trainer(
        decoder=decoder,
        train_dataset=dataset,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        augmentation=args.augmentation,
        early_stopping_patience=args.early_stopping,
        checkpoint_dir=output_dir / "checkpoints",
    )

    results = trainer.fit()
    print(f"\nTraining complete! Results: {json.dumps(results, indent=2)}")

    # Save decoder weights
    weights_path = output_dir / "decoder_weights.pt"
    trainer.save(weights_path)
    print(f"Saved decoder weights to: {weights_path}")

    # Generate inference script
    from core.export.script_gen import generate_inference_script

    generate_inference_script(
        decoder_class=type(decoder).__name__,
        task=args.task,
        encoder_name=args.encoder,
        num_classes=args.num_classes,
        weights_path=str(weights_path),
        output_path=output_dir / "inference.py",
    )
    print(f"Generated inference script: {output_dir / 'inference.py'}")


def _run_predict(args: argparse.Namespace) -> None:
    """Execute inference from CLI args."""
    from core.encoders import create_encoder
    from core.evaluation.inference import run_inference
    from core.export.weights import load_decoder_weights

    encoder = create_encoder(args.encoder)
    decoder = _create_decoder(args.decoder, args.task, encoder, args.num_classes)
    load_decoder_weights(decoder, args.weights)

    image_path = Path(args.image)
    if image_path.is_dir():
        image_paths = list(image_path.glob("*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}]
    else:
        image_paths = [image_path]

    results = run_inference(decoder, image_paths)
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if not isinstance(v, type(None))}, indent=2, default=str))


def _run_info() -> None:
    """Print available encoders and decoders."""
    from core.encoders import ALL_ENCODER_VARIANTS

    print("Available Encoders:")
    print("-" * 50)
    for name, config in ALL_ENCODER_VARIANTS.items():
        print(f"  {name}: embed_dim={config['embed_dim']}, patch_size={config['patch_size']}")

    print("\nAvailable Decoders:")
    print("-" * 50)
    print("  Classification:")
    print("    - linear_probe (LinearProbe): Single linear layer")
    print("    - mlp (MLPHead): 2-layer MLP with dropout")
    print("    - transformer (TransformerHead): Cross-attention decoder")
    print("  Detection:")
    print("    - rtdetr (RTDETRDecoder): RT-DETRv2 with multi-scale, VFL, CDN [default]")
    print("    - detr_lite (DETRLiteDecoder): Lightweight DETR-style")
    print("    - detr_multiscale (DETRMultiScaleDecoder): DETR with FPN-style multi-scale features")
    print("    - fpn (FPNHead): Feature Pyramid Network + anchors")
    print("  Segmentation:")
    print("    - linear_seg (LinearSegHead): Per-patch linear classifier")
    print("    - upernet (UPerNetHead): UPerNet multi-scale decoder")
    print("    - mask_transformer (MaskTransformerHead): Mask prediction via dot-product")


def _create_decoder(decoder_name: str, task: str, encoder, num_classes: int):
    """Create a decoder by name or auto-select based on task."""
    from core.decoders.classification import LinearProbe, MLPHead, TransformerHead
    from core.decoders.detection import DETRLiteDecoder, DETRMultiScaleDecoder, FPNHead
    from core.decoders.rtdetr import RTDETRDecoder
    from core.decoders.segmentation import LinearSegHead, UPerNetHead, MaskTransformerHead

    decoder_map = {
        "linear_probe": LinearProbe,
        "mlp": MLPHead,
        "transformer": TransformerHead,
        "detr_lite": DETRLiteDecoder,
        "detr_multiscale": DETRMultiScaleDecoder,
        "fpn": FPNHead,
        "rtdetr": RTDETRDecoder,
        "linear_seg": LinearSegHead,
        "upernet": UPerNetHead,
        "mask_transformer": MaskTransformerHead,
    }

    # Auto-select defaults
    auto_defaults = {
        "classification": "linear_probe",
        "detection": "rtdetr",
        "segmentation": "linear_seg",
    }

    if decoder_name == "auto":
        decoder_name = auto_defaults.get(task, "linear_probe")

    if decoder_name not in decoder_map:
        raise ValueError(f"Unknown decoder: {decoder_name}. Choose from: {list(decoder_map.keys())}")

    return decoder_map[decoder_name](encoder, num_classes)


if __name__ == "__main__":
    main()
