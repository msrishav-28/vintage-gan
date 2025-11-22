"""
VintageGAN Training Module

This package contains dataset loaders, training loops, and utilities
for training the VintageGAN model.

Modules:
    dataset: PyTorch datasets and dataloaders
    download_data: Data download and preparation utilities
    pretrain: Generator pretraining (Phase 1)
    gan_train: GAN fine-tuning (Phase 3)
    losses: Loss functions (perceptual, pixel, consistency)
"""

from .dataset import (
    ImageNetSubsetDataset,
    VintageGANDataset,
    FilmSetDataset,
    create_dataloaders,
    validate_dataloader
)

from .download_data import (
    download_imagenet_subset,
    download_filmset,
    verify_dataset,
    create_dummy_dataset
)

__all__ = [
    # Dataset classes
    'ImageNetSubsetDataset',
    'VintageGANDataset',
    'FilmSetDataset',
    'create_dataloaders',
    'validate_dataloader',
    
    # Data utilities
    'download_imagenet_subset',
    'download_filmset',
    'verify_dataset',
    'create_dummy_dataset',
]

__version__ = '0.1.0'
__author__ = 'VintageGAN Project'
