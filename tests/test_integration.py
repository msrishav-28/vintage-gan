"""End-to-end smoke tests for the academic VintageGAN path."""

from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from defects import apply_vintage_defects, create_preset_conditions
from models import Discriminator, Generator
from training.dataset import ImageNetSubsetDataset, VintageGANDataset
from training.download_data import create_dummy_dataset
from training.losses import VintageGANLoss


def test_import_surface():
    import defects  # noqa: F401
    import evaluation  # noqa: F401
    import inference  # noqa: F401
    import models  # noqa: F401
    import training  # noqa: F401


def test_presets_are_six_dimensional():
    for preset in [
        "soft_fade",
        "warm_film",
        "dusty_archive",
        "scratched_negative",
        "heavy_vintage",
    ]:
        vector = create_preset_conditions(preset)
        assert vector.shape == (6,)
        assert np.all((vector >= 0.0) & (vector <= 1.0))


def test_models_accept_256_resolution():
    image = torch.randn(1, 3, 256, 256)
    condition = torch.rand(1, 6)

    generator = Generator(use_self_attention=False)
    discriminator = Discriminator()
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        generated = generator(image, condition)
        prediction = discriminator(generated, condition)

    assert generated.shape == image.shape
    assert prediction.shape == (1, 1, 16, 16)


def test_loss_runs_without_pretrained_download():
    generated = torch.randn(1, 3, 64, 64, requires_grad=True)
    target = torch.randn(1, 3, 64, 64)
    condition = torch.rand(1, 6)

    criterion = VintageGANLoss(perceptual_pretrained=False, consistency_weight=0.0)
    loss, loss_dict = criterion.compute_generator_loss(
        generated, target, condition, phase="pretrain"
    )

    assert loss.requires_grad
    assert "pixel" in loss_dict
    loss.backward()


def test_dataset_is_deterministic(tmp_path):
    dataset_dir = tmp_path / "public_images"
    create_dummy_dataset(str(dataset_dir), num_train=2, num_val=1, image_size=64)

    clean_dataset = ImageNetSubsetDataset(
        str(dataset_dir), split="train", image_size=64
    )
    vintage_dataset = VintageGANDataset(
        clean_dataset=clean_dataset,
        defect_generator=apply_vintage_defects,
        num_variants=2,
        seed=123,
    )

    first = vintage_dataset[1]
    second = vintage_dataset[1]

    assert torch.allclose(first["condition"], second["condition"])
    assert torch.allclose(first["defected"], second["defected"])


def test_tiny_training_step(tmp_path):
    dataset_dir = tmp_path / "public_images"
    create_dummy_dataset(str(dataset_dir), num_train=2, num_val=1, image_size=64)

    clean_dataset = ImageNetSubsetDataset(
        str(dataset_dir), split="train", image_size=64
    )
    vintage_dataset = VintageGANDataset(
        clean_dataset=clean_dataset,
        defect_generator=apply_vintage_defects,
        num_variants=1,
        seed=456,
    )
    batch = next(iter(DataLoader(vintage_dataset, batch_size=1)))

    generator = Generator(use_self_attention=False)
    discriminator = Discriminator()
    criterion = VintageGANLoss(perceptual_pretrained=False, consistency_weight=0.0)

    clean = batch["clean"]
    defected = batch["defected"]
    condition = batch["condition"]

    generated = generator(clean, condition)
    pred = discriminator(generated, condition)
    loss, _ = criterion.compute_generator_loss(
        generated,
        defected,
        condition,
        discriminator_pred=pred,
        phase="gan",
    )
    loss.backward()

    assert generated.shape == clean.shape
    assert loss.item() >= 0
