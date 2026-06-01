"""
VintageGAN Models Module

Neural network architectures for VintageGAN.

Modules:
    generator: U-Net with ResNet34 encoder and conditioning
    discriminator: PatchGAN discriminator with conditioning
    attention: Self-attention and CBAM modules
    detectors: Optional defect detection networks
"""

from .attention import SelfAttention, ChannelAttention, SpatialAttention, CBAM
from .generator import Generator, ConditionProjection, SpectralNorm
from .discriminator import Discriminator, MultiScaleDiscriminator

__all__ = [
    # Attention modules
    "SelfAttention",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    # Generator
    "Generator",
    "ConditionProjection",
    # Discriminator
    "Discriminator",
    "MultiScaleDiscriminator",
    # Utilities
    "SpectralNorm",
]


def __getattr__(name):
    """Lazily import detector utilities so basic model imports stay lightweight."""
    if name in {
        "DefectDetector",
        "MultiDefectDetector",
        "create_detectors",
        "train_detectors",
    }:
        from .detectors import (
            DefectDetector,
            MultiDefectDetector,
            create_detectors,
            train_detectors,
        )

        exports = {
            "DefectDetector": DefectDetector,
            "MultiDefectDetector": MultiDefectDetector,
            "create_detectors": create_detectors,
            "train_detectors": train_detectors,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
__author__ = "VintageGAN Project"
