"""
Generator Pretraining Script
Specification Reference: Section 4.1 - Phase 1: Generator Pretraining

Pretrains the generator using perceptual + pixel loss for 40 epochs.

Author: VintageGAN Project
Date: 2024
"""

import os
import sys
from pathlib import Path
import argparse
from typing import Dict
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Generator
from training.dataset import create_dataloaders
from training.losses import VintageGANLoss
from defects import apply_vintage_defects
from training.checkpointing import load_checkpoint_file
from training.config import (
    amp_enabled,
    get_gradient_accumulation_steps,
    load_config,
    seed_everything,
    write_experiment_metadata,
)


def setup_training(config_path: str, profile: str = None) -> Dict:
    """
    Setup training configuration.

    Args:
        config_path: Path to training configuration YAML

    Returns:
        Configuration dictionary
    """
    config = load_config(config_path, profile=profile)

    # Create directories
    Path(config["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    return config


def train_epoch(
    generator: nn.Module,
    dataloader: DataLoader,
    criterion: VintageGANLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: Dict,
) -> Dict[str, float]:
    """
    Train generator for one epoch.

    Args:
        generator: Generator model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        writer: TensorBoard writer
        config: Configuration dictionary

    Returns:
        Dictionary of average losses
    """
    generator.train()

    total_losses = {"total": 0.0, "pixel": 0.0, "perceptual": 0.0}

    num_batches = len(dataloader)
    if num_batches == 0:
        raise RuntimeError("Training dataloader produced zero batches")

    accumulation_steps = get_gradient_accumulation_steps(config)
    use_amp = amp_enabled(config, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Get data
        if isinstance(batch, dict):
            clean = batch["clean"].to(device)
            defected = batch["defected"].to(device)
            conditions = batch["condition"].to(device)
        else:
            # Clean dataset only (shouldn't happen in normal training)
            images, _ = batch
            clean = images.to(device)
            defected = clean
            conditions = torch.rand(clean.size(0), 6).to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            generated = generator(clean, conditions)
            loss, loss_dict = criterion.compute_generator_loss(
                generated, defected, conditions, phase="pretrain"
            )
            scaled_loss = loss / accumulation_steps

        scaler.scale(scaled_loss).backward()

        should_step = (batch_idx + 1) % accumulation_steps == 0 or (
            batch_idx + 1
        ) == num_batches
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Update losses
        for key, value in loss_dict.items():
            total_losses[key] += value

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "pixel": f"{loss_dict['pixel']:.4f}",
                "percep": f"{loss_dict['perceptual']:.4f}",
            }
        )

        # Log to TensorBoard
        if batch_idx % config["logging"]["log_every_n_steps"] == 0:
            global_step = epoch * num_batches + batch_idx
            for key, value in loss_dict.items():
                writer.add_scalar(f"train/{key}_loss", value, global_step)

    # Compute averages
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}

    return avg_losses


@torch.no_grad()
def validate_epoch(
    generator: nn.Module,
    dataloader: DataLoader,
    criterion: VintageGANLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    output_dir: Path,
    config: Dict = None,
) -> Dict[str, float]:
    """
    Validate generator for one epoch.

    Args:
        generator: Generator model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device
        epoch: Current epoch
        writer: TensorBoard writer
        output_dir: Directory to save sample images

    Returns:
        Dictionary of average validation losses
    """
    generator.eval()

    total_losses = {"total": 0.0, "pixel": 0.0, "perceptual": 0.0}

    num_batches = len(dataloader)
    if num_batches == 0:
        raise RuntimeError("Validation dataloader produced zero batches")
    use_amp = amp_enabled(config or {}, device)

    # Save first batch for visualization
    save_samples = True

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
        if isinstance(batch, dict):
            clean = batch["clean"].to(device)
            defected = batch["defected"].to(device)
            conditions = batch["condition"].to(device)
        else:
            images, _ = batch
            clean = images.to(device)
            defected = clean
            conditions = torch.rand(clean.size(0), 6).to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            generated = generator(clean, conditions)
            loss, loss_dict = criterion.compute_generator_loss(
                generated, defected, conditions, phase="pretrain"
            )

        # Update losses
        for key, value in loss_dict.items():
            total_losses[key] += value

        # Save sample images (first batch only)
        if save_samples and batch_idx == 0:
            import torchvision

            # Save grid of images
            num_samples = min(4, clean.size(0))

            # Denormalize for visualization
            clean_vis = (clean[:num_samples] + 1) / 2
            defected_vis = (defected[:num_samples] + 1) / 2
            generated_vis = (generated[:num_samples] + 1) / 2

            # Create grid: [clean, defected, generated]
            comparison = torch.cat([clean_vis, defected_vis, generated_vis], dim=0)
            grid = torchvision.utils.make_grid(comparison, nrow=num_samples, padding=2)

            # Save to file
            output_path = output_dir / f"epoch_{epoch:03d}.png"
            torchvision.utils.save_image(grid, output_path)

            # Log to TensorBoard
            writer.add_image("validation/samples", grid, epoch)

            save_samples = False

    # Compute averages
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}

    return avg_losses


def save_checkpoint(
    generator: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    losses: Dict[str, float],
    config: Dict,
    filename: str = "checkpoint.pth",
):
    """
    Save training checkpoint.

    Args:
        generator: Generator model
        optimizer: Optimizer
        epoch: Current epoch
        losses: Loss dictionary
        config: Configuration
        filename: Checkpoint filename
    """
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_path = checkpoint_dir / filename

    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "config": config,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str, generator: nn.Module, optimizer: optim.Optimizer = None
) -> int:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        generator: Generator model
        optimizer: Optimizer (optional)

    Returns:
        Epoch number from checkpoint
    """
    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")

    generator.load_state_dict(checkpoint["generator_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)

    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch


def main():
    """Main pretraining function."""
    parser = argparse.ArgumentParser(description="VintageGAN Generator Pretraining")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Config profile to apply, e.g. local_256 or cloud_512",
    )

    args = parser.parse_args()

    # Setup
    print("=" * 60)
    print("VINTAGEGAN GENERATOR PRETRAINING")
    print("=" * 60)

    config = setup_training(args.config, profile=args.profile)
    device = torch.device(args.device)

    print(f"\nDevice: {device}")
    print(f"Config: {args.config}")

    # Set random seeds for reproducibility
    seed_everything(
        config["reproducibility"].get("seed"),
        deterministic=config["reproducibility"].get("deterministic", True),
        benchmark=config["reproducibility"].get("benchmark", False),
    )
    write_experiment_metadata(
        config,
        args.config,
        run_name=f"pretrain_{config.get('active_profile', 'base')}",
        output_dir=config["paths"]["output_dir"],
    )

    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        args.config,
        defect_generator=apply_vintage_defects,
        profile=config.get("active_profile"),
    )

    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")

    # Initialize model
    print("\nInitializing generator...")
    generator = Generator(
        condition_dim=config["model"]["generator"]["condition_dim"],
        use_spectral_norm=config["model"]["generator"]["use_spectral_norm"],
        use_self_attention=config["model"]["generator"]["use_self_attention"],
        dropout_rate=config["model"]["generator"]["dropout_rate"],
    ).to(device)

    num_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator parameters: {num_params:,}")

    # Initialize loss function
    print("\nInitializing loss function...")
    criterion = VintageGANLoss(
        perceptual_weight=config["training"]["phase1_pretrain"][
            "perceptual_loss_weight"
        ],
        pixel_weight=config["training"]["phase1_pretrain"]["pixel_loss_weight"],
        perceptual_pretrained=config.get("perceptual_loss", {}).get("pretrained", True),
    ).to(device)

    # Initialize optimizer
    print("\nInitializing optimizer...")
    optimizer = optim.Adam(
        generator.parameters(),
        lr=config["training"]["phase1_pretrain"]["learning_rate"],
        betas=tuple(config["training"]["phase1_pretrain"]["betas"]),
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["phase1_pretrain"]["epochs"]
    )

    # TensorBoard writer
    log_dir = Path(config["paths"]["log_dir"]) / "pretrain"
    writer = SummaryWriter(log_dir)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, generator, optimizer)
        start_epoch += 1

    # Training loop
    print(
        f"\nStarting training for {config['training']['phase1_pretrain']['epochs']} epochs..."
    )
    print("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, config["training"]["phase1_pretrain"]["epochs"]):
        epoch_start_time = time.time()

        # Train
        train_losses = train_epoch(
            generator,
            dataloaders["train"],
            criterion,
            optimizer,
            device,
            epoch,
            writer,
            config,
        )

        # Validate
        val_losses = validate_epoch(
            generator,
            dataloaders["val"],
            criterion,
            device,
            epoch,
            writer,
            Path(config["paths"]["output_dir"]),
            config,
        )

        # Update learning rate
        scheduler.step()

        # Log epoch summary
        epoch_time = time.time() - epoch_start_time

        print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
        print(
            f"  Train - Total: {train_losses['total']:.4f}, "
            f"Pixel: {train_losses['pixel']:.4f}, "
            f"Perceptual: {train_losses['perceptual']:.4f}"
        )
        print(
            f"  Val   - Total: {val_losses['total']:.4f}, "
            f"Pixel: {val_losses['pixel']:.4f}, "
            f"Perceptual: {val_losses['perceptual']:.4f}"
        )
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Log to TensorBoard
        writer.add_scalar("epoch/train_loss", train_losses["total"], epoch)
        writer.add_scalar("epoch/val_loss", val_losses["total"], epoch)
        writer.add_scalar("epoch/learning_rate", scheduler.get_last_lr()[0], epoch)

        # Save checkpoint every N epochs
        if (epoch + 1) % config["checkpointing"]["save_every_n_epochs"] == 0:
            save_checkpoint(
                generator,
                optimizer,
                epoch,
                val_losses,
                config,
                filename=f"generator_pretrain_epoch_{epoch:03d}.pth",
            )

        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            save_checkpoint(
                generator,
                optimizer,
                epoch,
                val_losses,
                config,
                filename="generator_pretrain_best.pth",
            )
            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")

    # Final checkpoint
    save_checkpoint(
        generator,
        optimizer,
        config["training"]["phase1_pretrain"]["epochs"] - 1,
        val_losses,
        config,
        filename="generator_pretrain_final.pth",
    )

    print("\n" + "=" * 60)
    print("PRETRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(
        f"Final model saved: {config['paths']['checkpoint_dir']}/generator_pretrain_final.pth"
    )
    print("=" * 60)

    writer.close()


if __name__ == "__main__":
    main()
