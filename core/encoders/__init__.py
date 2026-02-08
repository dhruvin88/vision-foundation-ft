"""Encoder modules for loading and freezing foundation models."""

from core.encoders.base import BaseEncoder
from core.encoders.dinov2 import DINOv2Encoder

__all__ = ["BaseEncoder", "DINOv2Encoder"]
