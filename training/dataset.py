"""Dataset and DataLoader utilities for controllable vintage synthesis."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor in [-1, 1] to an HWC uint8 array."""
    normalized = (tensor.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
    try:
        array = normalized.numpy()
    except RuntimeError:
        array = np.array(normalized.tolist(), dtype=np.float32)
    array = (array * 255).astype(np.uint8)
    return np.transpose(array, (1, 2, 0))


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert an HWC uint8 image to a normalized CHW tensor in [-1, 1]."""
    array = array.astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    try:
        tensor = torch.from_numpy(array).float()
    except RuntimeError:
        tensor = torch.tensor(array.tolist(), dtype=torch.float32)
    return (tensor - 0.5) / 0.5


def _array_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to tensor with a fallback for broken NumPy/Torch ABIs."""
    try:
        return torch.from_numpy(array).float()
    except RuntimeError:
        return torch.tensor(array.tolist(), dtype=torch.float32)


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python random inside DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _resize_center_crop(image: Image.Image, image_size: int) -> Image.Image:
    """Resize shortest side then center-crop to a square."""
    width, height = image.size
    scale = image_size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.BICUBIC)
    left = (resized.width - image_size) // 2
    top = (resized.height - image_size) // 2
    return resized.crop((left, top, left + image_size, top + image_size))


class ImageNetSubsetDataset(Dataset):
    """Load clean RGB images from explicit ``train`` or ``val`` split folders."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 512,
        transform: Optional[Callable] = None,
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
        expected_count: Optional[int] = None,
        allow_split_fallback: bool = False,
    ):
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError(f"Split must be 'train' or 'val', got '{split}'")

        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.normalize_range = normalize_range
        self.allow_split_fallback = allow_split_fallback
        self.image_paths = self._get_image_paths()
        self.transform = (
            transform if transform is not None else self._default_transform()
        )

        if expected_count is not None and len(self.image_paths) < expected_count:
            print(
                f"Warning: found {len(self.image_paths)} {split} images; expected {expected_count}"
            )

    def _get_image_paths(self) -> List[Path]:
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            if not self.allow_split_fallback:
                raise RuntimeError(
                    f"Split directory not found: {split_dir}. "
                    "Use explicit train/val folders to avoid data leakage."
                )
            split_dir = self.root_dir

        paths: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(split_dir.rglob(f"*{ext}"))
        return sorted(set(paths))

    def _default_transform(self) -> Callable:
        def transform(image: Image.Image) -> torch.Tensor:
            image = _resize_center_crop(image, self.image_size)
            return numpy_to_tensor(np.array(image))

        return transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            return self.transform(image), str(img_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {img_path}: {exc}") from exc

    def get_sample_images(self, num_samples: int = 10) -> List[torch.Tensor]:
        indices = np.random.choice(
            len(self), size=min(num_samples, len(self)), replace=False
        )
        return [self[idx][0] for idx in indices]


class VintageGANDataset(Dataset):
    """Generate clean/defected/condition training pairs on the fly."""

    def __init__(
        self,
        clean_dataset: ImageNetSubsetDataset,
        defect_generator: Optional[Callable] = None,
        num_variants: int = 5,
        pregenerated_path: Optional[str] = None,
        augmentation_config: Optional[Dict] = None,
        return_clean: bool = True,
        cache_in_memory: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        if defect_generator is None and pregenerated_path is None:
            raise ValueError(
                "Must provide either defect_generator or pregenerated_path"
            )

        self.clean_dataset = clean_dataset
        self.defect_generator = defect_generator
        self.num_variants = num_variants
        self.pregenerated_path = pregenerated_path
        self.return_clean = return_clean
        self.cache_in_memory = cache_in_memory
        self.seed = seed
        self.total_size = len(clean_dataset) * num_variants
        self.augmentation = self._setup_augmentation(augmentation_config)
        self.cache = {} if cache_in_memory else None

    def _setup_augmentation(self, config: Optional[Dict]) -> Optional[Callable]:
        if config is None:
            return None

        try:
            import albumentations as A
        except ImportError:
            print("Warning: albumentations not installed, skipping augmentation")
            return None

        image_size = int(config.get("image_size", self.clean_dataset.image_size))
        try:
            crop = A.RandomResizedCrop(
                size=(image_size, image_size), scale=(0.9, 1.0), p=0.5
            )
        except TypeError:
            crop = A.RandomResizedCrop(
                height=image_size, width=image_size, scale=(0.9, 1.0), p=0.5
            )

        return A.Compose(
            [
                A.HorizontalFlip(p=config.get("horizontal_flip_prob", 0.5)),
                crop,
                A.ColorJitter(
                    brightness=config.get("brightness_jitter", 0.1),
                    contrast=config.get("contrast_jitter", 0.1),
                    saturation=0.0,
                    hue=0.0,
                    p=0.3,
                ),
            ],
            additional_targets={"target": "image"},
        )

    def __len__(self) -> int:
        return self.total_size

    def _sample_seed(self, img_path: str, variant_idx: int) -> int:
        seed_material = f"{self.seed}:{img_path}:{variant_idx}".encode("utf-8")
        digest = hashlib.blake2b(seed_material, digest_size=8).hexdigest()
        return int(digest, 16) % (2**32)

    def _generate_random_conditions(self) -> np.ndarray:
        return np.random.uniform(0.0, 1.0, size=6).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]

        clean_idx = idx // self.num_variants
        variant_idx = idx % self.num_variants
        clean_image, img_path = self.clean_dataset[clean_idx]

        rng_state = np.random.get_state()
        np.random.seed(self._sample_seed(img_path, variant_idx))
        try:
            condition = self._generate_random_conditions()
            clean_np = tensor_to_numpy(clean_image)
            if self.defect_generator is None:
                raise NotImplementedError(
                    "Pre-generated pair loading is not implemented"
                )
            defected_np = self.defect_generator(clean_np, condition)
        finally:
            np.random.set_state(rng_state)

        if self.augmentation is not None and self.clean_dataset.split == "train":
            augmented = self.augmentation(image=clean_np, target=defected_np)
            clean_np = augmented["image"]
            defected_np = augmented["target"]

        output = {
            "defected": numpy_to_tensor(defected_np),
            "condition": _array_to_tensor(condition),
            "image_id": f"{clean_idx:06d}_{variant_idx:02d}",
        }
        if self.return_clean:
            output["clean"] = numpy_to_tensor(clean_np)

        if self.cache is not None:
            self.cache[idx] = output
        return output

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor_to_numpy(tensor)

    def _numpy_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return numpy_to_tensor(array)


class FilmSetDataset(Dataset):
    """Load real vintage reference images for optional distribution metrics."""

    def __init__(
        self, root_dir: str, image_size: int = 512, transform: Optional[Callable] = None
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.image_paths = self._get_image_paths()
        self.transform = (
            transform if transform is not None else self._default_transform()
        )
        print(f"FilmSet: found {len(self.image_paths)} vintage photos")

    def _get_image_paths(self) -> List[Path]:
        paths: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(self.root_dir.rglob(f"*{ext}"))
        return sorted(set(paths))

    def _default_transform(self) -> Callable:
        def transform(image: Image.Image) -> torch.Tensor:
            image = _resize_center_crop(image, self.image_size)
            return numpy_to_tensor(np.array(image))

        return transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            return self.transform(image), str(img_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {img_path}: {exc}") from exc


def create_dataloaders(
    config_path: str,
    defect_generator: Optional[Callable] = None,
    profile: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """Create train and validation dataloaders from YAML configuration."""
    try:
        from training.config import load_config

        config = load_config(config_path, profile=profile)
    except Exception:
        text = Path(config_path).read_text(encoding="utf-8")
        try:
            config = yaml.safe_load(text)
        except yaml.YAMLError:
            config = yaml.safe_load(text.replace("\\", "/"))

    dataset_config = config["dataset"]
    hardware_config = config["hardware"]
    aug_config = config.get("augmentation")
    seed = int(config.get("reproducibility", {}).get("seed", 42))

    train_clean = ImageNetSubsetDataset(
        root_dir=dataset_config["imagenet_path"],
        split="train",
        image_size=dataset_config["image_size"],
        expected_count=dataset_config.get("num_train_images"),
        allow_split_fallback=dataset_config.get("allow_split_fallback", False),
    )
    val_clean = ImageNetSubsetDataset(
        root_dir=dataset_config["imagenet_path"],
        split="val",
        image_size=dataset_config["image_size"],
        expected_count=dataset_config.get("num_val_images"),
        allow_split_fallback=dataset_config.get("allow_split_fallback", False),
    )

    if defect_generator is not None:
        train_dataset = VintageGANDataset(
            clean_dataset=train_clean,
            defect_generator=defect_generator,
            num_variants=dataset_config["variants_per_image"],
            augmentation_config={
                **(aug_config or {}),
                "image_size": dataset_config["image_size"],
            },
            return_clean=True,
            seed=seed,
        )
        val_dataset = VintageGANDataset(
            clean_dataset=val_clean,
            defect_generator=defect_generator,
            num_variants=dataset_config["val_variants_per_image"],
            augmentation_config=None,
            return_clean=True,
            seed=seed + 1,
        )
    else:
        print("Warning: no defect generator provided, returning clean images only")
        train_dataset = train_clean
        val_dataset = val_clean

    batch_size = int(hardware_config["batch_size"])
    torch_generator = torch.Generator()
    torch_generator.manual_seed(seed)
    requested_drop_last = bool(hardware_config.get("drop_last", True))
    drop_last = requested_drop_last and len(train_dataset) >= batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=hardware_config["num_workers"],
        pin_memory=hardware_config["pin_memory"],
        drop_last=drop_last,
        worker_init_fn=_seed_worker,
        generator=torch_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=hardware_config["num_workers"],
        pin_memory=hardware_config["pin_memory"],
        drop_last=False,
        worker_init_fn=_seed_worker,
    )

    return {"train": train_loader, "val": val_loader}


def validate_dataloader(
    dataloader: DataLoader,
    num_batches: int = 5,
    expected_image_size: Optional[int] = None,
) -> None:
    """Validate dataloader tensor shapes and ranges."""
    print("\n" + "=" * 60)
    print("DATALOADER VALIDATION")
    print("=" * 60)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        if isinstance(batch, dict):
            assert "defected" in batch
            assert "condition" in batch
            defected = batch["defected"]
            condition = batch["condition"]
        else:
            defected, _ = batch
            condition = None

        assert defected.dim() == 4
        assert defected.shape[1] == 3
        if expected_image_size is not None:
            assert defected.shape[2] == expected_image_size
            assert defected.shape[3] == expected_image_size
        if condition is not None:
            assert condition.shape[1] == 6
            assert torch.all(condition >= 0.0) and torch.all(condition <= 1.0)
        assert defected.min() >= -1.5 and defected.max() <= 1.5

        if isinstance(batch, dict) and "clean" in batch:
            assert batch["clean"].shape == defected.shape

        print(f"Batch {batch_idx + 1}/{num_batches}: Valid")

    print("\nAll validation checks passed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    loaders = create_dataloaders("configs/training_config.yaml", defect_generator=None)
    validate_dataloader(loaders["train"], num_batches=3)
