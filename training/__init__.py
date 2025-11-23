"""
VintageGAN Training Module

Data loading and training utilities for VintageGAN.

Modules:
    dataset: Dataset and DataLoader implementations
    download_data: Data download and preparation utilities
    losses: Loss functions for training
    pretrain: Generator pretraining script
    gan_train: GAN training script
"""

from .dataset import (
    ImageNetSubsetDataset,
    VintageGANDataset,
    FilmSetDataset,
    create_dataloaders,
    validate_dataloader,
    tensor_to_numpy,
    numpy_to_tensor
)

from .download_data import (
    download_imagenet_subset,
    download_filmset,
    verify_dataset,
    create_dummy_dataset
)

from .losses import (
    VGGPerceptualLoss,
    PixelLoss,
    AdversarialLoss,
    ConsistencyLoss,
    VintageGANLoss
)

__all__ = [
    # Datasets
    'ImageNetSubsetDataset',
    'VintageGANDataset',
    'FilmSetDataset',
    'create_dataloaders',
    'validate_dataloader',
    
    # Utilities
    'tensor_to_numpy',
    'numpy_to_tensor',
    'download_imagenet_subset',
    'download_filmset',
    'verify_dataset',
    'create_dummy_dataset',
    
    # Losses
    'VGGPerceptualLoss',
    'PixelLoss',
    'AdversarialLoss',
    'ConsistencyLoss',
    'VintageGANLoss'
]

__version__ = '1.0.0'
__author__ = 'VintageGAN Project'
