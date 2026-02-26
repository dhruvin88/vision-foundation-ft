"""Encoder modules for loading and freezing foundation models."""

from core.encoders.base import BaseEncoder
from core.encoders.dinov2 import DINOv2Encoder, ALL_VARIANTS as _DINOV2_VARIANTS
from core.encoders.dinov3 import DINOv3Encoder, DINOV3_VARIANTS as _DINOV3_VARIANTS

ALL_ENCODER_VARIANTS = {**_DINOV2_VARIANTS, **_DINOV3_VARIANTS}
DEFAULT_ENCODER = "dinov3_vitb16"


def create_encoder(model_name: str = DEFAULT_ENCODER, **kwargs) -> BaseEncoder:
    """Create an encoder by model name.

    Args:
        model_name: Encoder variant name (e.g., 'dinov3_vitb16', 'dinov2_vitb14').
        **kwargs: Passed to the encoder constructor (input_size, intermediate_layers).

    Returns:
        A frozen encoder instance.
    """
    if model_name.startswith("dinov3_"):
        return DINOv3Encoder(model_name, **kwargs)
    elif model_name.startswith("dinov2_"):
        return DINOv2Encoder(model_name, **kwargs)
    else:
        raise ValueError(
            f"Unknown encoder: {model_name}. "
            f"Choose from: {list(ALL_ENCODER_VARIANTS.keys())}"
        )


__all__ = [
    "BaseEncoder",
    "DINOv2Encoder",
    "DINOv3Encoder",
    "ALL_ENCODER_VARIANTS",
    "DEFAULT_ENCODER",
    "create_encoder",
]
