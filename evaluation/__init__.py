"""
VintageGAN Evaluation Module

Evaluation metrics and analysis tools.

Author: VintageGAN Project
Date: 2024
"""

from .metrics import (
    calculate_fid,
    calculate_ssim,
    calculate_psnr,
    calculate_inception_score,
    calculate_condition_accuracy,
    evaluate_model
)

__all__ = [
    'calculate_fid',
    'calculate_ssim',
    'calculate_psnr',
    'calculate_inception_score',
    'calculate_condition_accuracy',
    'evaluate_model'
]

__version__ = '1.0.0'
__author__ = 'VintageGAN Project'
