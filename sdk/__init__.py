"""Foundation Model Fine-Tuning SDK.

A simple, high-level API for fine-tuning vision foundation models.

Example:
    import foundation_ft as fft

    encoder = fft.Encoder("dinov2_vitb14")
    decoder = fft.ClassificationHead(encoder, num_classes=5, head_type="mlp")
    dataset = fft.Dataset.from_folder("./my_images/", task="classification")
    trainer = fft.Trainer(decoder, dataset, lr=1e-3, epochs=20)
    results = trainer.fit()
    trainer.save("./my_model.pt")
"""

from core.encoders.dinov2 import DINOv2Encoder
from core.data.dataset import FFTDataset
from core.training.trainer import Trainer
from core.evaluation.inference import run_inference
from core.evaluation.metrics import compute_metrics
from core.export.weights import save_decoder_weights, load_decoder_weights
from core.export.script_gen import generate_inference_script
from core.decoders.classification import LinearProbe, MLPHead, TransformerHead
from core.decoders.detection import DETRLiteDecoder, FPNHead
from core.decoders.segmentation import LinearSegHead, UPerNetHead, MaskTransformerHead


class Encoder:
    """High-level encoder wrapper.

    Args:
        model_name: DINOv2 variant name (e.g., 'dinov2_vitb14').
        input_size: Input image size (default 518).
        intermediate_layers: List of layer indices for multi-scale feature extraction.
            Required for FPN (detection) and UPerNet (segmentation) heads.
            If None, only final-layer features are extracted.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        input_size: int = 518,
        intermediate_layers: list[int] | None = None,
    ):
        self._encoder = DINOv2Encoder(
            model_name=model_name,
            input_size=input_size,
            intermediate_layers=intermediate_layers,
        )

    @property
    def model(self):
        return self._encoder

    @property
    def embed_dim(self):
        return self._encoder.embed_dim

    def get_transform(self):
        return self._encoder.get_transform()


def ClassificationHead(encoder, num_classes: int, head_type: str = "linear"):
    """Create a classification head.

    Args:
        encoder: An Encoder instance or DINOv2Encoder.
        num_classes: Number of output classes.
        head_type: 'linear', 'mlp', or 'transformer'.

    Returns:
        A classification decoder.
    """
    enc = encoder._encoder if isinstance(encoder, Encoder) else encoder
    heads = {
        "linear": LinearProbe,
        "mlp": MLPHead,
        "transformer": TransformerHead,
    }
    if head_type not in heads:
        raise ValueError(f"Unknown head_type: {head_type}. Choose from: {list(heads.keys())}")
    return heads[head_type](enc, num_classes)


def DetectionHead(encoder, num_classes: int, head_type: str = "detr_lite", **kwargs):
    """Create a detection head.

    Args:
        encoder: An Encoder instance or DINOv2Encoder.
        num_classes: Number of object classes.
        head_type: 'detr_lite' or 'fpn'.

    Returns:
        A detection decoder.
    """
    enc = encoder._encoder if isinstance(encoder, Encoder) else encoder
    heads = {
        "detr_lite": DETRLiteDecoder,
        "fpn": FPNHead,
    }
    if head_type not in heads:
        raise ValueError(f"Unknown head_type: {head_type}. Choose from: {list(heads.keys())}")

    # FPN needs multi-scale features — auto-enable if not already configured
    if head_type == "fpn" and hasattr(enc, "intermediate_layers"):
        if enc.intermediate_layers is None:
            enc.intermediate_layers = enc.default_intermediate_layers()

    return heads[head_type](enc, num_classes, **kwargs)


def SegmentationHead(encoder, num_classes: int, head_type: str = "linear", **kwargs):
    """Create a segmentation head.

    Args:
        encoder: An Encoder instance or DINOv2Encoder.
        num_classes: Number of segmentation classes.
        head_type: 'linear', 'upernet', or 'mask_transformer'.

    Returns:
        A segmentation decoder.
    """
    enc = encoder._encoder if isinstance(encoder, Encoder) else encoder
    heads = {
        "linear": LinearSegHead,
        "upernet": UPerNetHead,
        "mask_transformer": MaskTransformerHead,
    }
    if head_type not in heads:
        raise ValueError(f"Unknown head_type: {head_type}. Choose from: {list(heads.keys())}")

    # UPerNet needs multi-scale features — auto-enable if not already configured
    if head_type == "upernet" and hasattr(enc, "intermediate_layers"):
        if enc.intermediate_layers is None:
            enc.intermediate_layers = enc.default_intermediate_layers()

    return heads[head_type](enc, num_classes, **kwargs)


# Convenience alias
Dataset = FFTDataset

__all__ = [
    "Encoder",
    "ClassificationHead",
    "DetectionHead",
    "SegmentationHead",
    "Dataset",
    "Trainer",
    "run_inference",
    "compute_metrics",
    "save_decoder_weights",
    "load_decoder_weights",
    "generate_inference_script",
]
