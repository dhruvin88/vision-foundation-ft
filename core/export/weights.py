"""Save and load decoder weights."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from core.decoders.base import BaseDecoder

logger = logging.getLogger(__name__)


def save_decoder_weights(
    decoder: BaseDecoder,
    path: str | Path,
    include_metadata: bool = True,
) -> None:
    """Save only the decoder/head weights (not the encoder).

    The saved file is small since the encoder is frozen and not included.

    Args:
        decoder: The decoder to save.
        path: Output path for the .pt file.
        include_metadata: Whether to include metadata about the model config.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect only decoder state dict (exclude encoder)
    decoder_state = {}
    encoder_prefix = "encoder."
    for key, value in decoder.state_dict().items():
        if not key.startswith(encoder_prefix):
            decoder_state[key] = value

    save_dict = {"decoder_state_dict": decoder_state}

    if include_metadata:
        save_dict["metadata"] = {
            "decoder_class": type(decoder).__name__,
            "task": decoder.task,
            "num_classes": decoder.num_classes,
            "embed_dim": decoder._embed_dim,
            "num_trainable_params": decoder.num_trainable_params(),
        }

    torch.save(save_dict, path)

    file_size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(
        "Saved decoder weights to %s (%.2f MB, %d params)",
        path,
        file_size_mb,
        decoder.num_trainable_params(),
    )


def load_decoder_weights(
    decoder: BaseDecoder,
    path: str | Path,
    strict: bool = True,
) -> dict:
    """Load decoder weights from a saved file.

    Args:
        decoder: The decoder to load weights into.
        path: Path to the .pt file.
        strict: Whether to enforce strict state dict matching.

    Returns:
        Metadata dictionary from the saved file (if available).
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    decoder_state = checkpoint.get("decoder_state_dict", checkpoint)

    # Load only non-encoder parameters
    current_state = decoder.state_dict()
    encoder_prefix = "encoder."
    for key in list(decoder_state.keys()):
        if key.startswith(encoder_prefix):
            del decoder_state[key]

    # Build full state dict with existing encoder weights
    full_state = {}
    for key, value in current_state.items():
        if key.startswith(encoder_prefix):
            full_state[key] = value
        elif key in decoder_state:
            full_state[key] = decoder_state[key]
        elif strict:
            raise KeyError(f"Missing key in checkpoint: {key}")
        else:
            full_state[key] = value

    decoder.load_state_dict(full_state, strict=strict)

    metadata = checkpoint.get("metadata", {})
    logger.info("Loaded decoder weights from %s", path)
    return metadata
