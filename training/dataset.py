"""
VintageGAN Dataset Implementation
Specification Reference: Section 3 (Dataset Creation) and Section 9 Day 1-2

This module implements PyTorch Dataset classes for loading and preprocessing
image pairs (clean, defected) with corresponding condition vectors.

Classes:
    ImageNetSubsetDataset: Loads clean ImageNet images
    VintageGANDataset: Main dataset with synthetic defect pairs
    FilmSetDataset: Real vintage photos for FID evaluation
    
Author: VintageGAN Project
Date: 2024
"""

import os
from typing import Tuple, Optional, Dict, List, Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import yaml
from tqdm import tqdm


class ImageNetSubsetDataset(Dataset):
    """
    Dataset for loading clean ImageNet images.
    
    Specification Reference: Section 3.1 - Base Dataset
    - Size: 10,000 training images, 1,000 validation images
    - Resolution: All images resized to 512×512 with center crop
    - Format: Normalized tensors [-1, 1]
    
    Args:
        root_dir: Path to ImageNet subset directory
        split: 'train' or 'val'
        image_size: Target image size (default: 512)
        transform: Optional transform to apply
        normalize_range: Output normalization range (default: [-1, 1])
    
    Returns:
        Tuple of (image_tensor, image_path) where image_tensor is normalized to [-1, 1]
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 512,
        transform: Optional[Callable] = None,
        normalize_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.normalize_range = normalize_range
        
        # Validate split
        if split not in ["train", "val"]:
            raise ValueError(f"Split must be 'train' or 'val', got '{split}'")
        
        # Get image paths
        self.image_paths = self._get_image_paths()
        
        # Validate dataset size (Specification Section 3.1)
        expected_size = 10000 if split == "train" else 1000
        if len(self.image_paths) < expected_size:
            print(f"Warning: Found {len(self.image_paths)} images, "
                  f"expected {expected_size} for {split} split")
        
        # Setup transforms
        self.transform = transform if transform is not None else self._default_transform()
    
    def _get_image_paths(self) -> List[Path]:
        """
        Recursively find all image files in the directory.
        
        Returns:
            List of Path objects for valid images
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
        image_paths = []
        
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            # If split directory doesn't exist, try root directory directly
            split_dir = self.root_dir
        
        if split_dir.exists():
            for ext in valid_extensions:
                image_paths.extend(split_dir.rglob(f'*{ext}'))
        
        return sorted(image_paths)
    
    def _default_transform(self) -> transforms.Compose:
        """
        Default preprocessing pipeline as per Specification Section 3.1.
        
        - Center crop to target size
        - Convert to tensor
        - Normalize to [-1, 1] range (for Tanh output)
        
        Returns:
            Composed transform pipeline
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])
    
    def __len__(self) -> int:
        """Return total number of images in dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Load and preprocess single image.
        
        Args:
            idx: Index of image to load
        
        Returns:
            Tuple of (image_tensor, image_path)
            - image_tensor: (3, 512, 512) normalized to [-1, 1]
            - image_path: String path to source image
        
        Raises:
            RuntimeError: If image cannot be loaded
        """
        img_path = self.image_paths[idx]
        
        try:
            # Load image using PIL
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            return image_tensor, str(img_path)
        
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {str(e)}")
    
    def get_sample_images(self, num_samples: int = 10) -> List[torch.Tensor]:
        """
        Get random sample images for visualization.
        
        Args:
            num_samples: Number of samples to return
        
        Returns:
            List of image tensors
        """
        indices = np.random.choice(len(self), size=min(num_samples, len(self)), replace=False)
        return [self[idx][0] for idx in indices]


class VintageGANDataset(Dataset):
    """
    Main dataset class for VintageGAN training.
    
    Specification Reference: Section 3.3 - Training Pair Generation
    
    This dataset generates training pairs of (clean_image, defected_image, condition_vector)
    on-the-fly or loads pre-generated pairs from disk.
    
    Dataset Structure:
    - Training: 10,000 clean × 5 variants = 50,000 pairs
    - Validation: 1,000 clean × 3 variants = 3,000 pairs
    
    Args:
        clean_dataset: ImageNetSubsetDataset instance for clean images
        defect_generator: Function that applies defects to images
        num_variants: Number of defect variants per clean image
        pregenerated_path: Optional path to pre-generated pairs
        augmentation_config: Data augmentation configuration
        return_clean: Whether to return clean image as well
    
    Returns:
        Dictionary containing:
        - 'clean': Clean image tensor (3, 512, 512) if return_clean=True
        - 'defected': Defected image tensor (3, 512, 512)
        - 'condition': Condition vector (6,) normalized to [0, 1]
        - 'image_id': Unique identifier for the image pair
    """
    
    def __init__(
        self,
        clean_dataset: ImageNetSubsetDataset,
        defect_generator: Optional[Callable] = None,
        num_variants: int = 5,
        pregenerated_path: Optional[str] = None,
        augmentation_config: Optional[Dict] = None,
        return_clean: bool = True,
        cache_in_memory: bool = False
    ):
        super().__init__()
        
        self.clean_dataset = clean_dataset
        self.defect_generator = defect_generator
        self.num_variants = num_variants
        self.pregenerated_path = pregenerated_path
        self.return_clean = return_clean
        self.cache_in_memory = cache_in_memory
        
        # Total dataset size
        self.total_size = len(clean_dataset) * num_variants
        
        # Setup augmentation (Specification Section 3.3)
        self.augmentation = self._setup_augmentation(augmentation_config)
        
        # Cache for in-memory loading
        self.cache = {} if cache_in_memory else None
        
        # Validate configuration
        if defect_generator is None and pregenerated_path is None:
            raise ValueError("Must provide either defect_generator or pregenerated_path")
    
    def _setup_augmentation(self, config: Optional[Dict]) -> Optional[Callable]:
        """
        Setup data augmentation pipeline per Specification Section 3.3.
        
        Data Augmentation:
        - Random horizontal flips (p=0.5)
        - Random crops (480×480 then resize to 512×512)
        - Color jitter (brightness ±10%, contrast ±10%)
        - Applied to BOTH clean and defected images
        
        Args:
            config: Augmentation configuration dictionary
        
        Returns:
            Augmentation function or None
        """
        if config is None:
            return None
        
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            augmentation = A.Compose([
                A.HorizontalFlip(p=config.get('horizontal_flip_prob', 0.5)),
                A.RandomResizedCrop(
                    height=512, 
                    width=512, 
                    scale=(0.9, 1.0),  # 480×480 to 512×512
                    p=0.5
                ),
                A.ColorJitter(
                    brightness=config.get('brightness_jitter', 0.1),
                    contrast=config.get('contrast_jitter', 0.1),
                    saturation=0.0,  # Don't modify saturation
                    hue=0.0,  # Don't modify hue
                    p=0.3
                ),
            ])
            return augmentation
        
        except ImportError:
            print("Warning: albumentations not installed, skipping augmentation")
            return None
    
    def __len__(self) -> int:
        """Return total number of training pairs."""
        return self.total_size
    
    def _generate_random_conditions(self) -> np.ndarray:
        """
        Generate random condition vector.
        
        Specification Reference: Section 2.1.1 - Conditioning Vector Format
        
        Returns:
            NumPy array of shape (6,) with values in [0.0, 1.0]:
            [grain_intensity, scratch_density, dust_count, 
             vignette_strength, color_shift, blur_amount]
        """
        return np.random.uniform(0.0, 1.0, size=6).astype(np.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training pair at index.
        
        Args:
            idx: Index of training pair
        
        Returns:
            Dictionary with keys: 'clean', 'defected', 'condition', 'image_id'
        
        Raises:
            RuntimeError: If defect generation fails
        """
        # Check cache first
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        
        # Determine which clean image and which variant
        clean_idx = idx // self.num_variants
        variant_idx = idx % self.num_variants
        
        # Load clean image
        clean_image, img_path = self.clean_dataset[clean_idx]
        
        # Generate random conditions (or use pre-generated)
        # Set seed for reproducibility of variants
        np.random.seed(hash(f"{img_path}_{variant_idx}") % (2**32))
        condition = self._generate_random_conditions()
        np.random.seed()  # Reset seed
        
        # Convert tensor to numpy for defect generation
        # Denormalize from [-1, 1] to [0, 255]
        clean_np = self._tensor_to_numpy(clean_image)
        
        # Apply defects
        if self.defect_generator is not None:
            defected_np = self.defect_generator(clean_np, condition)
        else:
            # Load pre-generated (to be implemented)
            raise NotImplementedError("Pre-generated pairs loading not yet implemented")
        
        # Apply augmentation to both images if specified
        if self.augmentation is not None and self.clean_dataset.split == "train":
            # Apply same augmentation to both
            augmented = self.augmentation(image=clean_np, mask=defected_np)
            clean_np = augmented['image']
            defected_np = augmented['mask']
        
        # Convert back to tensors
        clean_tensor = self._numpy_to_tensor(clean_np)
        defected_tensor = self._numpy_to_tensor(defected_np)
        condition_tensor = torch.from_numpy(condition).float()
        
        # Create output dictionary
        output = {
            'defected': defected_tensor,
            'condition': condition_tensor,
            'image_id': f"{clean_idx:06d}_{variant_idx:02d}"
        }
        
        if self.return_clean:
            output['clean'] = clean_tensor
        
        # Cache if enabled
        if self.cache is not None:
            self.cache[idx] = output
        
        return output
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor from [-1, 1] to numpy uint8 [0, 255].
        
        Args:
            tensor: PyTorch tensor (3, H, W) in range [-1, 1]
        
        Returns:
            NumPy array (H, W, 3) in range [0, 255] uint8
        """
        # Denormalize from [-1, 1] to [0, 1]
        array = (tensor * 0.5 + 0.5).numpy()
        # Convert to [0, 255]
        array = (array * 255).astype(np.uint8)
        # Transpose from (C, H, W) to (H, W, C)
        array = np.transpose(array, (1, 2, 0))
        return array
    
    def _numpy_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """
        Convert numpy uint8 [0, 255] to tensor [-1, 1].
        
        Args:
            array: NumPy array (H, W, 3) in range [0, 255] uint8
        
        Returns:
            PyTorch tensor (3, H, W) in range [-1, 1]
        """
        # Convert to float [0, 1]
        array = array.astype(np.float32) / 255.0
        # Transpose from (H, W, C) to (C, H, W)
        array = np.transpose(array, (2, 0, 1))
        # Convert to tensor
        tensor = torch.from_numpy(array).float()
        # Normalize to [-1, 1]
        tensor = (tensor - 0.5) / 0.5
        return tensor


class FilmSetDataset(Dataset):
    """
    Dataset for loading real vintage photographs.
    
    Specification Reference: Section 3.1 and Section 5.1.1 (FID Evaluation)
    
    Used as reference distribution for FID score calculation.
    
    Args:
        root_dir: Path to FilmSet directory containing real vintage photos
        image_size: Target image size (default: 512)
        transform: Optional custom transform
    
    Returns:
        Tuple of (image_tensor, image_path) normalized to [-1, 1]
    """
    
    def __init__(
        self,
        root_dir: str,
        image_size: int = 512,
        transform: Optional[Callable] = None
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        
        # Get all vintage photo paths
        self.image_paths = self._get_image_paths()
        
        print(f"FilmSet: Found {len(self.image_paths)} vintage photos")
        
        # Setup transforms
        self.transform = transform if transform is not None else self._default_transform()
    
    def _get_image_paths(self) -> List[Path]:
        """Find all vintage photo files."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
        image_paths = []
        
        for ext in valid_extensions:
            image_paths.extend(self.root_dir.rglob(f'*{ext}'))
        
        return sorted(image_paths)
    
    def _default_transform(self) -> transforms.Compose:
        """Default preprocessing for vintage photos."""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self) -> int:
        """Return total number of vintage photos."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Load and preprocess single vintage photo."""
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor, str(img_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {str(e)}")


def create_dataloaders(
    config_path: str,
    defect_generator: Optional[Callable] = None
) -> Dict[str, DataLoader]:
    """
    Create train and validation dataloaders from configuration.
    
    Specification Reference: Day 1-2 Validation Requirements
    - Data loader returns correct tensor shapes: (batch, 3, 512, 512)
    - Can load and visualize ImageNet samples
    
    Args:
        config_path: Path to training_config.yaml
        defect_generator: Function to apply defects (if None, returns clean images only)
    
    Returns:
        Dictionary with keys 'train' and 'val' containing DataLoader instances
    
    Example:
        >>> config_path = "configs/training_config.yaml"
        >>> dataloaders = create_dataloaders(config_path, my_defect_generator)
        >>> for batch in dataloaders['train']:
        ...     clean = batch['clean']  # (B, 3, 512, 512)
        ...     defected = batch['defected']  # (B, 3, 512, 512)
        ...     conditions = batch['condition']  # (B, 6)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config['dataset']
    hardware_config = config['hardware']
    aug_config = config.get('augmentation', None)
    
    # Create clean image datasets
    train_clean = ImageNetSubsetDataset(
        root_dir=dataset_config['imagenet_path'],
        split='train',
        image_size=dataset_config['image_size']
    )
    
    val_clean = ImageNetSubsetDataset(
        root_dir=dataset_config['imagenet_path'],
        split='val',
        image_size=dataset_config['image_size']
    )
    
    # Create VintageGAN datasets
    if defect_generator is not None:
        train_dataset = VintageGANDataset(
            clean_dataset=train_clean,
            defect_generator=defect_generator,
            num_variants=dataset_config['variants_per_image'],
            augmentation_config=aug_config,
            return_clean=True
        )
        
        val_dataset = VintageGANDataset(
            clean_dataset=val_clean,
            defect_generator=defect_generator,
            num_variants=dataset_config['val_variants_per_image'],
            augmentation_config=None,  # No augmentation for validation
            return_clean=True
        )
    else:
        # Return clean datasets if no defect generator provided
        print("Warning: No defect generator provided, returning clean images only")
        train_dataset = train_clean
        val_dataset = val_clean
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hardware_config['batch_size'],
        shuffle=True,
        num_workers=hardware_config['num_workers'],
        pin_memory=hardware_config['pin_memory'],
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=hardware_config['batch_size'],
        shuffle=False,
        num_workers=hardware_config['num_workers'],
        pin_memory=hardware_config['pin_memory'],
        drop_last=False
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }


def validate_dataloader(dataloader: DataLoader, num_batches: int = 5) -> None:
    """
    Validate dataloader outputs correct tensor shapes and ranges.
    
    Specification Reference: Day 1-2 Validation
    
    Args:
        dataloader: DataLoader to validate
        num_batches: Number of batches to check
    
    Raises:
        AssertionError: If validation fails
    """
    print("\n" + "="*60)
    print("DATALOADER VALIDATION")
    print("="*60)
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # Check if batch is dict (VintageGANDataset) or tuple (ImageNetSubsetDataset)
        if isinstance(batch, dict):
            # VintageGAN dataset format
            assert 'defected' in batch, "Batch missing 'defected' key"
            assert 'condition' in batch, "Batch missing 'condition' key"
            
            defected = batch['defected']
            condition = batch['condition']
            
            # Validate shapes
            assert len(defected.shape) == 4, f"Expected 4D tensor, got {len(defected.shape)}D"
            assert defected.shape[1] == 3, f"Expected 3 channels, got {defected.shape[1]}"
            assert defected.shape[2] == 512, f"Expected height 512, got {defected.shape[2]}"
            assert defected.shape[3] == 512, f"Expected width 512, got {defected.shape[3]}"
            
            # Validate condition vector
            assert condition.shape[1] == 6, f"Expected 6D condition, got {condition.shape[1]}D"
            assert torch.all(condition >= 0.0) and torch.all(condition <= 1.0), \
                "Condition values must be in [0, 1]"
            
            # Validate value ranges (should be approximately [-1, 1])
            assert defected.min() >= -1.5 and defected.max() <= 1.5, \
                f"Defected values out of range: [{defected.min():.2f}, {defected.max():.2f}]"
            
            if 'clean' in batch:
                clean = batch['clean']
                assert clean.shape == defected.shape, \
                    f"Clean and defected shapes mismatch: {clean.shape} vs {defected.shape}"
            
            print(f"Batch {i+1}/{num_batches}: ✓ Valid")
            print(f"  - Defected shape: {tuple(defected.shape)}")
            print(f"  - Defected range: [{defected.min():.3f}, {defected.max():.3f}]")
            print(f"  - Condition shape: {tuple(condition.shape)}")
            print(f"  - Condition range: [{condition.min():.3f}, {condition.max():.3f}]")
        
        else:
            # Clean image dataset format (tuple)
            images, paths = batch
            
            assert len(images.shape) == 4, f"Expected 4D tensor, got {len(images.shape)}D"
            assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
            assert images.shape[2] == 512, f"Expected height 512, got {images.shape[2]}"
            assert images.shape[3] == 512, f"Expected width 512, got {images.shape[3]}"
            
            print(f"Batch {i+1}/{num_batches}: ✓ Valid")
            print(f"  - Images shape: {tuple(images.shape)}")
            print(f"  - Images range: [{images.min():.3f}, {images.max():.3f}]")
    
    print("\n✅ All validation checks passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    Test script for dataset module.
    
    Usage:
        python training/dataset.py
    """
    print("VintageGAN Dataset Module - Test Script")
    print("="*60)
    
    # This will be tested with actual defect generator in Day 3-4
    # For now, we test the clean image loading
    
    config_path = "configs/training_config.yaml"
    
    if os.path.exists(config_path):
        print(f"\nLoading configuration from: {config_path}")
        
        try:
            # Create dataloaders (without defect generator for now)
            dataloaders = create_dataloaders(config_path, defect_generator=None)
            
            print(f"\n✓ Created train dataloader: {len(dataloaders['train'])} batches")
            print(f"✓ Created val dataloader: {len(dataloaders['val'])} batches")
            
            # Validate train loader
            print("\nValidating train dataloader...")
            validate_dataloader(dataloaders['train'], num_batches=3)
            
            # Validate val loader
            print("\nValidating val dataloader...")
            validate_dataloader(dataloaders['val'], num_batches=3)
            
        except Exception as e:
            print(f"\n❌ Error during testing: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠ Configuration file not found: {config_path}")
        print("Please ensure configs/training_config.yaml exists")
