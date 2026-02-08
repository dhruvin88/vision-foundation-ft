"""Generate standalone inference scripts for trained models."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

INFERENCE_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""Auto-generated inference script for {decoder_class} ({task} task).

Usage:
    python {script_name} --image path/to/image.jpg --weights {weights_path}
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def load_model(weights_path: str, device: str = "auto"):
    """Load the encoder + decoder model."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load frozen DINOv2 encoder
    encoder = torch.hub.load("facebookresearch/dinov2", "{encoder_name}", pretrained=True)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Load decoder weights
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    decoder_state = checkpoint.get("decoder_state_dict", checkpoint)
    metadata = checkpoint.get("metadata", {{}})

    # Reconstruct decoder
    from core.decoders.{task} import {decoder_class}
    from core.encoders.dinov2 import DINOv2Encoder

    enc = DINOv2Encoder("{encoder_name}")
    num_classes = metadata.get("num_classes", {num_classes})

    # Import and instantiate the decoder
    from core.decoders import {decoder_class}
    decoder = {decoder_class}(enc, num_classes={num_classes})
    decoder.load_state_dict({{**{{k: v for k, v in decoder.state_dict().items() if k.startswith("encoder.")}}, **decoder_state}})
    decoder = decoder.to(device)
    decoder.eval()

    return decoder, device


def preprocess(image_path: str) -> torch.Tensor:
    """Preprocess an image for inference."""
    transform = transforms.Compose([
        transforms.Resize(({input_size}, {input_size}), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def predict(decoder, image_tensor: torch.Tensor, device: str):
    """Run inference."""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        features = decoder.encoder.forward_features(image_tensor)
        output = decoder(features)
    return output


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="{weights_path}", help="Path to decoder weights")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    args = parser.parse_args()

    decoder, device = load_model(args.weights, args.device)
    image_tensor = preprocess(args.image)
    output = predict(decoder, image_tensor, device)

    # Task-specific output formatting
    task = "{task}"
    if task == "classification":
        probs = torch.softmax(output[0], dim=0)
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()
        print(f"Predicted class: {{pred_class}} (confidence: {{confidence:.4f}})")
        print(f"All probabilities: {{probs.cpu().tolist()}}")
    elif task == "detection":
        pred_logits = output["pred_logits"][0]
        pred_boxes = output["pred_boxes"][0]
        probs = torch.softmax(pred_logits, dim=-1)
        scores, labels = probs[:, :-1].max(dim=-1)
        keep = scores > 0.5
        print(f"Detected {{keep.sum().item()}} objects:")
        for box, label, score in zip(pred_boxes[keep], labels[keep], scores[keep]):
            print(f"  Class {{label.item()}}: {{box.cpu().tolist()}} (score: {{score.item():.4f}})")
    elif task == "segmentation":
        mask = output[0].argmax(dim=0)
        print(f"Segmentation mask shape: {{mask.shape}}")
        for c in mask.unique():
            area = (mask == c).sum().item()
            print(f"  Class {{c.item()}}: {{area}} pixels")


if __name__ == "__main__":
    main()
'''


def generate_inference_script(
    decoder_class: str,
    task: str,
    encoder_name: str = "dinov2_vitb14",
    num_classes: int = 2,
    weights_path: str = "model_weights.pt",
    input_size: int = 518,
    output_path: str | Path = "inference.py",
) -> str:
    """Generate a standalone Python inference script.

    Args:
        decoder_class: Name of the decoder class (e.g., 'LinearProbe').
        task: Task type ('classification', 'detection', 'segmentation').
        encoder_name: DINOv2 model variant name.
        num_classes: Number of classes.
        weights_path: Path to the saved decoder weights.
        input_size: Input image size.
        output_path: Where to write the generated script.

    Returns:
        The generated script as a string.
    """
    script = INFERENCE_SCRIPT_TEMPLATE.format(
        decoder_class=decoder_class,
        task=task,
        encoder_name=encoder_name,
        num_classes=num_classes,
        weights_path=weights_path,
        input_size=input_size,
        script_name=Path(output_path).name,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script)

    logger.info("Generated inference script: %s", output_path)
    return script
