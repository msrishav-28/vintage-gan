"""
Data Download Utilities for VintageGAN
Specification Reference: Section 3.1 - Base Dataset

This module provides utilities to download and prepare the ImageNet subset
and FilmSet (vintage photos) datasets.

Functions:
    download_imagenet_subset: Download 10k train + 1k val images from ImageNet
    download_filmset: Download vintage photos for FID evaluation
    verify_dataset: Verify downloaded dataset integrity
    
Author: VintageGAN Project
Date: 2024
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List
import requests
from tqdm import tqdm
import numpy as np
from PIL import Image
import yaml


def download_imagenet_subset(
    output_dir: str,
    num_train: int = 10000,
    num_val: int = 1000,
    seed: int = 42,
    use_huggingface: bool = True
) -> None:
    """
    Download ImageNet subset for VintageGAN training.
    
    Specification Reference: Section 3.1
    - Size: 10,000 training images, 1,000 validation images
    - Categories: Diverse natural images (landscapes, portraits, objects, architecture)
    - Resolution: Will be resized to 512×512 during loading
    
    Args:
        output_dir: Directory to save downloaded images
        num_train: Number of training images (default: 10,000)
        num_val: Number of validation images (default: 1,000)
        seed: Random seed for reproducibility (default: 42)
        use_huggingface: Whether to use Hugging Face datasets (default: True)
    
    Raises:
        ImportError: If required packages are not installed
        RuntimeError: If download fails
    
    Example:
        >>> download_imagenet_subset("data/imagenet_subset", num_train=10000, num_val=1000)
        Downloading ImageNet subset...
        ✓ Downloaded 10,000 training images
        ✓ Downloaded 1,000 validation images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print("IMAGENET SUBSET DOWNLOAD")
    print(f"{'='*60}")
    print(f"Target directory: {output_path}")
    print(f"Training images: {num_train}")
    print(f"Validation images: {num_val}")
    print(f"Random seed: {seed}")
    print(f"{'='*60}\n")
    
    if use_huggingface:
        _download_from_huggingface(train_dir, val_dir, num_train, num_val, seed)
    else:
        _download_from_kaggle(train_dir, val_dir, num_train, num_val, seed)
    
    # Verify downloaded data
    verify_dataset(output_dir, expected_train=num_train, expected_val=num_val)


def _download_from_huggingface(
    train_dir: Path,
    val_dir: Path,
    num_train: int,
    num_val: int,
    seed: int
) -> None:
    """
    Download ImageNet subset using Hugging Face datasets.
    
    This is the preferred method as it's easier to set up.
    
    Args:
        train_dir: Directory for training images
        val_dir: Directory for validation images
        num_train: Number of training images
        num_val: Number of validation images
        seed: Random seed
    
    Raises:
        ImportError: If datasets package not installed
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Hugging Face datasets not installed. "
            "Install with: pip install datasets"
        )
    
    print("Using Hugging Face datasets API...")
    print("Note: First download may take a while (downloading full ImageNet-1k)")
    print("Subsequent runs will use cached data.\n")
    
    try:
        # Load ImageNet-1k dataset
        print("Loading ImageNet-1k dataset (this may take several minutes)...")
        dataset = load_dataset(
            "imagenet-1k",
            split="train",
            streaming=True,  # Stream to avoid downloading entire dataset
            trust_remote_code=True
        )
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Sample random indices
        total_available = 1281167  # Total ImageNet-1k training images
        sampled_indices = np.random.choice(
            total_available,
            size=num_train + num_val,
            replace=False
        )
        sampled_indices = sorted(sampled_indices.tolist())
        
        print(f"Sampling {num_train + num_val} random images from {total_available} available...")
        
        # Download and save images
        downloaded = 0
        saved_count = 0
        
        for idx, example in enumerate(tqdm(dataset, total=num_train + num_val)):
            if idx not in sampled_indices:
                continue
            
            # Get image
            image = example['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to train or val directory
            if saved_count < num_train:
                save_dir = train_dir
                image_name = f"train_{saved_count:06d}.jpg"
            else:
                save_dir = val_dir
                image_name = f"val_{saved_count - num_train:06d}.jpg"
            
            save_path = save_dir / image_name
            image.save(save_path, quality=95)
            
            saved_count += 1
            
            if saved_count >= num_train + num_val:
                break
        
        print(f"\n✓ Downloaded {num_train} training images to {train_dir}")
        print(f"✓ Downloaded {num_val} validation images to {val_dir}")
    
    except Exception as e:
        print(f"\n❌ Error downloading from Hugging Face: {str(e)}")
        print("\nAlternative: Download ImageNet manually and place in data/imagenet_subset/")
        print("Or use Kaggle API with use_huggingface=False")
        raise


def _download_from_kaggle(
    train_dir: Path,
    val_dir: Path,
    num_train: int,
    num_val: int,
    seed: int
) -> None:
    """
    Download ImageNet subset using Kaggle API.
    
    Requires Kaggle API credentials setup:
    1. Install kaggle: pip install kaggle
    2. Download API credentials from https://www.kaggle.com/settings
    3. Place kaggle.json in ~/.kaggle/
    
    Args:
        train_dir: Directory for training images
        val_dir: Directory for validation images
        num_train: Number of training images
        num_val: Number of validation images
        seed: Random seed
    
    Raises:
        ImportError: If kaggle package not installed
        RuntimeError: If Kaggle credentials not found
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Kaggle API not installed. "
            "Install with: pip install kaggle"
        )
    
    print("Using Kaggle API...")
    print("Ensure you have set up Kaggle credentials (kaggle.json)\n")
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download ImageNet dataset
    # Note: This requires Kaggle competition acceptance
    print("Downloading from Kaggle (requires competition acceptance)...")
    print("Please accept ImageNet competition at: https://www.kaggle.com/c/imagenet-object-localization-challenge")
    
    try:
        # Download dataset
        api.competition_download_files(
            'imagenet-object-localization-challenge',
            path=str(train_dir.parent)
        )
        
        # Extract and sample images
        print("\nExtracting and sampling images...")
        # Implementation depends on specific Kaggle dataset structure
        # This is a placeholder - actual implementation would need to:
        # 1. Extract downloaded archive
        # 2. Sample random images
        # 3. Organize into train/val splits
        
        raise NotImplementedError(
            "Kaggle download requires manual setup. "
            "Please use Hugging Face method (use_huggingface=True) "
            "or download ImageNet manually."
        )
    
    except Exception as e:
        print(f"\n❌ Kaggle download failed: {str(e)}")
        raise


def download_filmset(
    output_dir: str,
    source_url: Optional[str] = None,
    num_images: int = 3000
) -> None:
    """
    Download real vintage photographs for FID evaluation.
    
    Specification Reference: Section 5.1.1 - FID Calculation
    
    FilmSet should contain real vintage/analog photographs to serve as
    the reference distribution for FID score calculation.
    
    Args:
        output_dir: Directory to save vintage photos
        source_url: URL to download FilmSet (if None, provides instructions)
        num_images: Target number of vintage photos (default: 3000)
    
    Note:
        If no public FilmSet is available, you can create one by:
        1. Collecting vintage photos from public domain sources
        2. Using datasets like:
           - Unsplash vintage collection
           - Pexels vintage/retro photos
           - Public domain archives (e.g., Library of Congress)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("FILMSET DOWNLOAD")
    print(f"{'='*60}")
    print(f"Target directory: {output_path}")
    print(f"Target number of images: {num_images}")
    print(f"{'='*60}\n")
    
    if source_url is None:
        print("⚠ No FilmSet source URL provided.")
        print("\nOptions to create FilmSet dataset:")
        print("\n1. Use Unsplash API:")
        print("   - Search for 'vintage', 'analog', 'film' keywords")
        print("   - Download ~3000 images")
        print("   - Requires Unsplash API key (free)")
        print("\n2. Manual collection:")
        print("   - Download vintage photos from public domain sources")
        print("   - Place in:", output_path)
        print("   - Supported formats: .jpg, .jpeg, .png")
        print("\n3. Use existing vintage photo datasets:")
        print("   - Search Kaggle, Hugging Face for vintage/retro datasets")
        print(f"{'='*60}\n")
        return
    
    # If source URL is provided, download from there
    print(f"Downloading from: {source_url}")
    # Implementation would depend on specific source
    raise NotImplementedError("FilmSet download from URL not yet implemented")


def verify_dataset(
    dataset_dir: str,
    expected_train: int = 10000,
    expected_val: int = 1000,
    check_images: bool = True
) -> bool:
    """
    Verify dataset integrity and size.
    
    Args:
        dataset_dir: Root directory of dataset
        expected_train: Expected number of training images
        expected_val: Expected number of validation images
        check_images: Whether to verify images can be loaded (slower)
    
    Returns:
        True if validation passes, False otherwise
    
    Raises:
        RuntimeError: If critical validation fails
    """
    dataset_path = Path(dataset_dir)
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    
    print(f"\n{'='*60}")
    print("DATASET VERIFICATION")
    print(f"{'='*60}")
    
    # Check directories exist
    if not train_dir.exists():
        raise RuntimeError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise RuntimeError(f"Validation directory not found: {val_dir}")
    
    # Count images
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
    
    train_images = [f for f in train_dir.rglob('*') if f.suffix in valid_extensions]
    val_images = [f for f in val_dir.rglob('*') if f.suffix in valid_extensions]
    
    print(f"Training images found: {len(train_images)} (expected: {expected_train})")
    print(f"Validation images found: {len(val_images)} (expected: {expected_val})")
    
    # Check counts
    if len(train_images) < expected_train * 0.9:  # Allow 10% tolerance
        print(f"⚠ Warning: Training set smaller than expected")
    if len(val_images) < expected_val * 0.9:
        print(f"⚠ Warning: Validation set smaller than expected")
    
    # Optionally verify images can be loaded
    if check_images:
        print("\nVerifying images can be loaded...")
        sample_size = min(100, len(train_images) + len(val_images))
        sample_images = np.random.choice(
            train_images + val_images,
            size=sample_size,
            replace=False
        )
        
        corrupted = []
        for img_path in tqdm(sample_images, desc="Checking images"):
            try:
                img = Image.open(img_path)
                img.verify()  # Verify it's a valid image
            except Exception as e:
                corrupted.append(img_path)
        
        if corrupted:
            print(f"\n⚠ Warning: {len(corrupted)} corrupted images found:")
            for path in corrupted[:5]:  # Show first 5
                print(f"  - {path}")
            return False
        else:
            print(f"✓ All {sample_size} sampled images are valid")
    
    print(f"\n✅ Dataset verification complete!")
    print(f"{'='*60}\n")
    return True


def create_dummy_dataset(
    output_dir: str,
    num_train: int = 100,
    num_val: int = 20,
    image_size: int = 512
) -> None:
    """
    Create dummy dataset for testing purposes.
    
    Useful for development and testing without downloading full ImageNet.
    
    Args:
        output_dir: Directory to save dummy images
        num_train: Number of training images
        num_val: Number of validation images
        image_size: Size of generated images
    
    Example:
        >>> create_dummy_dataset("data/imagenet_subset", num_train=100, num_val=20)
        ✓ Created 100 dummy training images
        ✓ Created 20 dummy validation images
    """
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("CREATING DUMMY DATASET")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"Training images: {num_train}")
    print(f"Validation images: {num_val}")
    print(f"Image size: {image_size}×{image_size}")
    print(f"{'='*60}\n")
    
    # Create training images
    print("Generating training images...")
    for i in tqdm(range(num_train)):
        # Generate random colored image
        img_array = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        img.save(train_dir / f"train_{i:06d}.jpg", quality=95)
    
    # Create validation images
    print("Generating validation images...")
    for i in tqdm(range(num_val)):
        img_array = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        img.save(val_dir / f"val_{i:06d}.jpg", quality=95)
    
    print(f"\n✓ Created {num_train} dummy training images")
    print(f"✓ Created {num_val} dummy validation images")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    """
    Command-line interface for data download utilities.
    
    Usage:
        # Create dummy dataset for testing
        python training/download_data.py --dummy
        
        # Download full ImageNet subset (requires setup)
        python training/download_data.py --full
        
        # Verify existing dataset
        python training/download_data.py --verify
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="VintageGAN Data Download Utilities")
    parser.add_argument(
        '--mode',
        choices=['dummy', 'full', 'verify'],
        default='dummy',
        help='Operation mode: dummy (test data), full (download), verify (check existing)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/imagenet_subset',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--num-train',
        type=int,
        default=10000,
        help='Number of training images'
    )
    parser.add_argument(
        '--num-val',
        type=int,
        default=1000,
        help='Number of validation images'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'dummy':
        # Create small dummy dataset for testing
        create_dummy_dataset(
            args.output_dir,
            num_train=min(100, args.num_train),
            num_val=min(20, args.num_val)
        )
    
    elif args.mode == 'full':
        # Download full ImageNet subset
        download_imagenet_subset(
            args.output_dir,
            num_train=args.num_train,
            num_val=args.num_val
        )
    
    elif args.mode == 'verify':
        # Verify existing dataset
        verify_dataset(
            args.output_dir,
            expected_train=args.num_train,
            expected_val=args.num_val
        )
