"""
Unit Tests for VintageGAN Dataset Module
Specification Reference: Day 1-2 - Write unit tests for data loading

Tests cover:
1. ImageNetSubsetDataset functionality
2. VintageGANDataset with dummy defect generator
3. DataLoader creation and validation
4. Tensor shape and value range verification
5. Augmentation pipeline

Author: VintageGAN Project
Date: 2024
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

import pytest
import torch
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import (
    ImageNetSubsetDataset,
    VintageGANDataset,
    FilmSetDataset,
    create_dataloaders,
    validate_dataloader,
)
from training.download_data import create_dummy_dataset


class TestImageNetSubsetDataset:
    """Test suite for ImageNetSubsetDataset class."""

    @pytest.fixture
    def dummy_dataset_dir(self, tmp_path):
        """Create temporary dummy dataset for testing."""
        dataset_dir = tmp_path / "imagenet_subset"
        create_dummy_dataset(str(dataset_dir), num_train=10, num_val=5, image_size=512)
        return dataset_dir

    def test_dataset_initialization(self, dummy_dataset_dir):
        """Test dataset can be initialized correctly."""
        dataset = ImageNetSubsetDataset(
            root_dir=str(dummy_dataset_dir), split="train", image_size=512
        )

        assert len(dataset) > 0, "Dataset should contain images"
        assert dataset.image_size == 512, "Image size should be 512"
        assert dataset.split == "train", "Split should be 'train'"

    def test_dataset_length(self, dummy_dataset_dir):
        """Test dataset returns correct length."""
        train_dataset = ImageNetSubsetDataset(
            root_dir=str(dummy_dataset_dir), split="train"
        )
        val_dataset = ImageNetSubsetDataset(
            root_dir=str(dummy_dataset_dir), split="val"
        )

        assert (
            len(train_dataset) == 10
        ), f"Expected 10 train images, got {len(train_dataset)}"
        assert len(val_dataset) == 5, f"Expected 5 val images, got {len(val_dataset)}"

    def test_getitem_returns_correct_format(self, dummy_dataset_dir):
        """Test __getitem__ returns (tensor, path) tuple."""
        dataset = ImageNetSubsetDataset(root_dir=str(dummy_dataset_dir), split="train")

        image, path = dataset[0]

        assert isinstance(image, torch.Tensor), "Image should be torch.Tensor"
        assert isinstance(path, str), "Path should be string"
        assert image.dim() == 3, f"Image should be 3D tensor, got {image.dim()}D"

    def test_image_shape(self, dummy_dataset_dir):
        """Test images have correct shape (3, 512, 512)."""
        dataset = ImageNetSubsetDataset(
            root_dir=str(dummy_dataset_dir), split="train", image_size=512
        )

        image, _ = dataset[0]

        assert image.shape == (
            3,
            512,
            512,
        ), f"Expected shape (3, 512, 512), got {image.shape}"

    def test_image_value_range(self, dummy_dataset_dir):
        """Test images are normalized to [-1, 1]."""
        dataset = ImageNetSubsetDataset(root_dir=str(dummy_dataset_dir), split="train")

        image, _ = dataset[0]

        assert (
            image.min() >= -1.5
        ), f"Image min value {image.min()} below expected range"
        assert image.max() <= 1.5, f"Image max value {image.max()} above expected range"
        # Most values should be in [-1, 1], allowing some tolerance for numerical precision

    def test_invalid_split_raises_error(self, dummy_dataset_dir):
        """Test invalid split parameter raises ValueError."""
        with pytest.raises(ValueError, match="Split must be 'train' or 'val'"):
            ImageNetSubsetDataset(
                root_dir=str(dummy_dataset_dir), split="test"  # Invalid split
            )

    def test_get_sample_images(self, dummy_dataset_dir):
        """Test get_sample_images returns correct number of samples."""
        dataset = ImageNetSubsetDataset(root_dir=str(dummy_dataset_dir), split="train")

        samples = dataset.get_sample_images(num_samples=5)

        assert len(samples) == 5, f"Expected 5 samples, got {len(samples)}"
        assert all(
            isinstance(img, torch.Tensor) for img in samples
        ), "All samples should be tensors"


class TestVintageGANDataset:
    """Test suite for VintageGANDataset class."""

    @pytest.fixture
    def dummy_clean_dataset(self, tmp_path):
        """Create dummy clean dataset."""
        dataset_dir = tmp_path / "imagenet_subset"
        create_dummy_dataset(str(dataset_dir), num_train=5, num_val=2)
        return ImageNetSubsetDataset(
            root_dir=str(dataset_dir), split="train", image_size=512
        )

    @pytest.fixture
    def dummy_defect_generator(self):
        """Create dummy defect generator function."""

        def generate_defects(image: np.ndarray, conditions: np.ndarray) -> np.ndarray:
            """
            Dummy defect generator that adds small noise.

            Args:
                image: (H, W, 3) uint8 array in [0, 255]
                conditions: (6,) float array in [0, 1]

            Returns:
                Defected image (H, W, 3) uint8 in [0, 255]
            """
            # Add small noise proportional to condition intensity
            noise_intensity = np.mean(conditions) * 10
            noise = np.random.randn(*image.shape) * noise_intensity
            defected = image.astype(np.float32) + noise
            defected = np.clip(defected, 0, 255).astype(np.uint8)
            return defected

        return generate_defects

    def test_dataset_initialization(self, dummy_clean_dataset, dummy_defect_generator):
        """Test VintageGANDataset initialization."""
        dataset = VintageGANDataset(
            clean_dataset=dummy_clean_dataset,
            defect_generator=dummy_defect_generator,
            num_variants=3,
            return_clean=True,
        )

        expected_size = len(dummy_clean_dataset) * 3
        assert (
            len(dataset) == expected_size
        ), f"Expected {expected_size} pairs, got {len(dataset)}"

    def test_getitem_returns_dict(self, dummy_clean_dataset, dummy_defect_generator):
        """Test __getitem__ returns dictionary with correct keys."""
        dataset = VintageGANDataset(
            clean_dataset=dummy_clean_dataset,
            defect_generator=dummy_defect_generator,
            num_variants=2,
            return_clean=True,
        )

        batch = dataset[0]

        assert isinstance(batch, dict), "Output should be dictionary"
        assert "clean" in batch, "Dictionary should contain 'clean' key"
        assert "defected" in batch, "Dictionary should contain 'defected' key"
        assert "condition" in batch, "Dictionary should contain 'condition' key"
        assert "image_id" in batch, "Dictionary should contain 'image_id' key"

    def test_output_shapes(self, dummy_clean_dataset, dummy_defect_generator):
        """Test output tensors have correct shapes."""
        dataset = VintageGANDataset(
            clean_dataset=dummy_clean_dataset,
            defect_generator=dummy_defect_generator,
            num_variants=2,
            return_clean=True,
        )

        batch = dataset[0]

        assert batch["clean"].shape == (
            3,
            512,
            512,
        ), f"Clean shape incorrect: {batch['clean'].shape}"
        assert batch["defected"].shape == (
            3,
            512,
            512,
        ), f"Defected shape incorrect: {batch['defected'].shape}"
        assert batch["condition"].shape == (
            6,
        ), f"Condition shape incorrect: {batch['condition'].shape}"

    def test_condition_value_range(self, dummy_clean_dataset, dummy_defect_generator):
        """Test condition vector is in [0, 1] range."""
        dataset = VintageGANDataset(
            clean_dataset=dummy_clean_dataset,
            defect_generator=dummy_defect_generator,
            num_variants=2,
        )

        batch = dataset[0]
        condition = batch["condition"]

        assert torch.all(condition >= 0.0), f"Condition min {condition.min()} below 0"
        assert torch.all(condition <= 1.0), f"Condition max {condition.max()} above 1"

    def test_reproducibility_with_seed(
        self, dummy_clean_dataset, dummy_defect_generator
    ):
        """Test same index produces same conditions (reproducibility)."""
        dataset = VintageGANDataset(
            clean_dataset=dummy_clean_dataset,
            defect_generator=dummy_defect_generator,
            num_variants=3,
        )

        # Get same item twice
        batch1 = dataset[2]
        batch2 = dataset[2]

        # Conditions should be identical (deterministic based on image+variant)
        assert torch.allclose(
            batch1["condition"], batch2["condition"], atol=1e-6
        ), "Same index should produce same conditions"

    def test_different_variants_have_different_conditions(
        self, dummy_clean_dataset, dummy_defect_generator
    ):
        """Test different variants of same image have different conditions."""
        dataset = VintageGANDataset(
            clean_dataset=dummy_clean_dataset,
            defect_generator=dummy_defect_generator,
            num_variants=5,
        )

        # Get different variants of first clean image (indices 0, 1, 2, 3, 4)
        conditions = [dataset[i]["condition"] for i in range(5)]

        # Check they're different from each other
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                assert not torch.allclose(
                    conditions[i], conditions[j], atol=1e-3
                ), f"Variants {i} and {j} have identical conditions"

    def test_return_clean_false(self, dummy_clean_dataset, dummy_defect_generator):
        """Test return_clean=False doesn't include clean image."""
        dataset = VintageGANDataset(
            clean_dataset=dummy_clean_dataset,
            defect_generator=dummy_defect_generator,
            num_variants=2,
            return_clean=False,
        )

        batch = dataset[0]

        assert (
            "clean" not in batch
        ), "Should not include 'clean' when return_clean=False"
        assert "defected" in batch, "Should still include 'defected'"
        assert "condition" in batch, "Should still include 'condition'"


class TestDataLoaderCreation:
    """Test suite for dataloader creation and validation."""

    @pytest.fixture
    def config_file(self, tmp_path):
        """Create temporary config file."""
        # Create dummy dataset
        dataset_dir = tmp_path / "data" / "imagenet_subset"
        create_dummy_dataset(str(dataset_dir), num_train=8, num_val=4)

        # Create config file
        config_path = tmp_path / "config.yaml"
        config_content = f"""
dataset:
  imagenet_path: "{dataset_dir}"
  image_size: 512
  variants_per_image: 2
  val_variants_per_image: 1

hardware:
  batch_size: 2
  num_workers: 0
  pin_memory: false

augmentation:
  horizontal_flip_prob: 0.5
  brightness_jitter: 0.1
  contrast_jitter: 0.1
"""
        config_path.write_text(config_content)
        return config_path

    def test_create_dataloaders_without_defect_generator(self, config_file):
        """Test creating dataloaders without defect generator (clean images only)."""
        dataloaders = create_dataloaders(str(config_file), defect_generator=None)

        assert "train" in dataloaders, "Should contain 'train' dataloader"
        assert "val" in dataloaders, "Should contain 'val' dataloader"

        # Test loading a batch
        train_batch = next(iter(dataloaders["train"]))
        assert len(train_batch) == 2, "Should return tuple of (images, paths)"

    def test_create_dataloaders_with_defect_generator(self, config_file):
        """Test creating dataloaders with defect generator."""

        def dummy_defect_gen(image, conditions):
            """Dummy defect generator."""
            return image.copy()

        dataloaders = create_dataloaders(
            str(config_file), defect_generator=dummy_defect_gen
        )

        # Test loading a batch
        train_batch = next(iter(dataloaders["train"]))

        assert isinstance(
            train_batch, dict
        ), "Should return dictionary with VintageGAN dataset"
        assert "clean" in train_batch, "Should contain clean images"
        assert "defected" in train_batch, "Should contain defected images"
        assert "condition" in train_batch, "Should contain conditions"

    def test_batch_shapes(self, config_file):
        """Test batch tensors have correct shapes."""

        def dummy_defect_gen(image, conditions):
            return image.copy()

        dataloaders = create_dataloaders(
            str(config_file), defect_generator=dummy_defect_gen
        )
        train_batch = next(iter(dataloaders["train"]))

        batch_size = 2
        assert train_batch["clean"].shape == (
            batch_size,
            3,
            512,
            512,
        ), f"Clean batch shape incorrect: {train_batch['clean'].shape}"
        assert train_batch["defected"].shape == (
            batch_size,
            3,
            512,
            512,
        ), f"Defected batch shape incorrect: {train_batch['defected'].shape}"
        assert train_batch["condition"].shape == (
            batch_size,
            6,
        ), f"Condition batch shape incorrect: {train_batch['condition'].shape}"


class TestTensorConversion:
    """Test tensor conversion utilities."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create sample dataset for testing."""
        dataset_dir = tmp_path / "imagenet_subset"
        create_dummy_dataset(str(dataset_dir), num_train=2, num_val=1)

        clean_dataset = ImageNetSubsetDataset(root_dir=str(dataset_dir), split="train")

        def dummy_defect_gen(image, conditions):
            return image.copy()

        return VintageGANDataset(
            clean_dataset=clean_dataset,
            defect_generator=dummy_defect_gen,
            num_variants=1,
        )

    def test_tensor_to_numpy_conversion(self, sample_dataset):
        """Test _tensor_to_numpy preserves value ranges."""
        # Create a known tensor
        tensor = torch.zeros(3, 512, 512)  # All zeros
        tensor[0, :, :] = -1.0  # R channel = -1 (should become 0)
        tensor[1, :, :] = 0.0  # G channel = 0 (should become 127-128)
        tensor[2, :, :] = 1.0  # B channel = 1 (should become 255)

        # Convert using dataset method
        array = sample_dataset._tensor_to_numpy(tensor)

        assert array.shape == (512, 512, 3), f"Shape mismatch: {array.shape}"
        assert array.dtype == np.uint8, f"Dtype should be uint8, got {array.dtype}"
        assert array[0, 0, 0] == 0, f"R channel should be 0, got {array[0, 0, 0]}"
        assert (
            120 <= array[0, 0, 1] <= 135
        ), f"G channel should be ~128, got {array[0, 0, 1]}"
        assert array[0, 0, 2] == 255, f"B channel should be 255, got {array[0, 0, 2]}"

    def test_numpy_to_tensor_conversion(self, sample_dataset):
        """Test _numpy_to_tensor preserves value ranges."""
        # Create a known array
        array = np.zeros((512, 512, 3), dtype=np.uint8)
        array[:, :, 0] = 0  # R = 0 (should become -1)
        array[:, :, 1] = 128  # G = 128 (should become ~0)
        array[:, :, 2] = 255  # B = 255 (should become 1)

        # Convert using dataset method
        tensor = sample_dataset._numpy_to_tensor(array)

        assert tensor.shape == (3, 512, 512), f"Shape mismatch: {tensor.shape}"
        assert (
            torch.abs(tensor[0, 0, 0] - (-1.0)) < 0.02
        ), f"R channel should be -1, got {tensor[0, 0, 0]}"
        assert (
            torch.abs(tensor[1, 0, 0] - 0.0) < 0.02
        ), f"G channel should be 0, got {tensor[1, 0, 0]}"
        assert (
            torch.abs(tensor[2, 0, 0] - 1.0) < 0.02
        ), f"B channel should be 1, got {tensor[2, 0, 0]}"

    def test_roundtrip_conversion(self, sample_dataset):
        """Test tensor -> numpy -> tensor roundtrip."""
        original = torch.rand(3, 512, 512) * 1.8 - 0.9

        # Roundtrip conversion
        array = sample_dataset._tensor_to_numpy(original)
        recovered = sample_dataset._numpy_to_tensor(array)

        # Should be close (some loss due to uint8 quantization)
        assert recovered.shape == original.shape, "Shape should be preserved"
        # Allow some error due to quantization
        max_error = torch.abs(recovered - original).max()
        assert max_error < 0.02, f"Roundtrip error too large: {max_error}"


def test_validate_dataloader(tmp_path, capsys):
    """Test validate_dataloader function."""
    # Create dummy dataset
    dataset_dir = tmp_path / "imagenet_subset"
    create_dummy_dataset(str(dataset_dir), num_train=8, num_val=4)

    # Create config
    config_path = tmp_path / "config.yaml"
    config_content = f"""
dataset:
  imagenet_path: "{dataset_dir}"
  image_size: 512
  variants_per_image: 1
  val_variants_per_image: 1

hardware:
  batch_size: 2
  num_workers: 0
  pin_memory: false
"""
    config_path.write_text(config_content)

    # Create dataloader
    def dummy_defect_gen(image, conditions):
        return image.copy()

    dataloaders = create_dataloaders(
        str(config_path), defect_generator=dummy_defect_gen
    )

    # Validate (should not raise)
    validate_dataloader(dataloaders["train"], num_batches=2)

    # Check output was printed
    captured = capsys.readouterr()
    assert "DATALOADER VALIDATION" in captured.out
    assert "All validation checks passed" in captured.out


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
