"""
VintageGAN Models Module

Neural network architectures for VintageGAN.

Modules:
    generator: U-Net with ResNet34 encoder and conditioning
    discriminator: PatchGAN discriminator with conditioning
    attention: Self-attention and CBAM modules
    detectors: Defect detection networks (to be implemented Day 10)
"""

from .attention import SelfAttention, ChannelAttention, SpatialAttention, CBAM
from .generator import Generator, ConditionProjection, SpectralNorm
from .discriminator import Discriminator, MultiScaleDiscriminator
from .detectors import DefectDetector, MultiDefectDetector, create_detectors, train_detectors

__all__ = [
    # Attention modules
    'SelfAttention',
    'ChannelAttention',
    'SpatialAttention',
    'CBAM',
    
    # Generator
    'Generator',
    'ConditionProjection',
    
    # Discriminator
    'Discriminator',
    'MultiScaleDiscriminator',
    
    # Detectors
    'DefectDetector',
    'MultiDefectDetector',
    'create_detectors',
    'train_detectors',
    
    # Utilities
    'SpectralNorm',
]

__version__ = '0.1.0'
__author__ = 'VintageGAN Project'
