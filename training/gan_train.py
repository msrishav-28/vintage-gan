"""
GAN Training Script
Specification Reference: Section 4.1 - Phase 2 & 3

Phase 2: Discriminator Pretraining (10 epochs)
Phase 3: GAN Fine-Tuning (10 epochs)

Author: VintageGAN Project
Date: 2024
"""

import os
import sys
from pathlib import Path
import argparse
from typing import Dict, Tuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Generator, Discriminator
from models.detectors import create_detectors, train_detectors
from training.dataset import create_dataloaders
from training.losses import VintageGANLoss
from defects import apply_vintage_defects
from training.pretrain import setup_training, save_checkpoint, load_checkpoint


def train_discriminator_phase(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    criterion: VintageGANLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: Dict
) -> Dict[str, float]:
    """
    Train discriminator only (Phase 2).
    
    Generator is frozen and used to generate fake samples.
    
    Args:
        generator: Frozen generator
        discriminator: Discriminator to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Discriminator optimizer
        device: Device
        epoch: Current epoch
        writer: TensorBoard writer
        config: Configuration
    
    Returns:
        Dictionary of average losses
    """
    generator.eval()  # Freeze generator
    discriminator.train()
    
    total_losses = {
        'total': 0.0,
        'real': 0.0,
        'fake': 0.0
    }
    
    num_batches = len(dataloader)
    pbar = tqdm(dataloader, desc=f"Disc Phase - Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        if isinstance(batch, dict):
            clean = batch['clean'].to(device)
            defected = batch['defected'].to(device)
            conditions = batch['condition'].to(device)
        else:
            images, _ = batch
            clean = images.to(device)
            defected = clean
            conditions = torch.rand(clean.size(0), 6).to(device)
        
        # Generate fake images
        with torch.no_grad():
            fake_images = generator(clean, conditions)
        
        # Discriminator predictions
        real_pred = discriminator(defected, conditions)
        fake_pred = discriminator(fake_images.detach(), conditions)
        
        # Compute loss
        d_loss, loss_dict = criterion.compute_discriminator_loss(real_pred, fake_pred)
        
        # Backward pass
        optimizer.zero_grad()
        d_loss.backward()
        optimizer.step()
        
        # Update losses
        for key, value in loss_dict.items():
            total_losses[key] += value
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{d_loss.item():.4f}",
            'real': f"{loss_dict['real']:.4f}",
            'fake': f"{loss_dict['fake']:.4f}"
        })
        
        # Log to TensorBoard
        if batch_idx % config['logging']['log_every_n_steps'] == 0:
            global_step = epoch * num_batches + batch_idx
            for key, value in loss_dict.items():
                writer.add_scalar(f'disc_phase/{key}_loss', value, global_step)
    
    # Compute averages
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}
    
    return avg_losses


def train_gan_phase(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    criterion: VintageGANLoss,
    g_optimizer: optim.Optimizer,
    d_optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: Dict
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Train both generator and discriminator (Phase 3).
    
    Args:
        generator: Generator
        discriminator: Discriminator
        dataloader: Training dataloader
        criterion: Loss function
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        device: Device
        epoch: Current epoch
        writer: TensorBoard writer
        config: Configuration
    
    Returns:
        Tuple of (generator_losses, discriminator_losses)
    """
    generator.train()
    discriminator.train()
    
    g_total_losses = {
        'total': 0.0,
        'pixel': 0.0,
        'perceptual': 0.0,
        'adversarial': 0.0,
        'consistency': 0.0
    }
    
    d_total_losses = {
        'total': 0.0,
        'real': 0.0,
        'fake': 0.0
    }
    
    num_batches = len(dataloader)
    pbar = tqdm(dataloader, desc=f"GAN Phase - Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        if isinstance(batch, dict):
            clean = batch['clean'].to(device)
            defected = batch['defected'].to(device)
            conditions = batch['condition'].to(device)
        else:
            images, _ = batch
            clean = images.to(device)
            defected = clean
            conditions = torch.rand(clean.size(0), 6).to(device)
        
        # =====================
        # Train Discriminator
        # =====================
        # Spec: Update discriminator 1 time per batch
        
        # Generate fake images
        with torch.no_grad():
            fake_images = generator(clean, conditions)
        
        # Discriminator predictions
        real_pred = discriminator(defected, conditions)
        fake_pred = discriminator(fake_images, conditions)
        
        # Discriminator loss
        d_loss, d_loss_dict = criterion.compute_discriminator_loss(real_pred, fake_pred)
        
        # Update discriminator
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # Update D losses
        for key, value in d_loss_dict.items():
            d_total_losses[key] += value
        
        # ==================
        # Train Generator
        # ==================
        # Spec: Update generator 1 time per batch
        
        # Generate fake images
        fake_images = generator(clean, conditions)
        
        # Discriminator prediction (for adversarial loss)
        fake_pred_for_g = discriminator(fake_images, conditions)
        
        # Generator loss (with all components)
        g_loss, g_loss_dict = criterion.compute_generator_loss(
            fake_images, defected, conditions,
            discriminator_pred=fake_pred_for_g,
            phase='gan'
        )
        
        # Update generator
        g_optimizer.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        g_optimizer.step()
        
        # Update G losses
        for key, value in g_loss_dict.items():
            g_total_losses[key] += value
        
        # Update progress bar
        pbar.set_postfix({
            'g_loss': f"{g_loss.item():.4f}",
            'd_loss': f"{d_loss.item():.4f}",
            'adv': f"{g_loss_dict.get('adversarial', 0):.4f}"
        })
        
        # Log to TensorBoard
        if batch_idx % config['logging']['log_every_n_steps'] == 0:
            global_step = epoch * num_batches + batch_idx
            for key, value in g_loss_dict.items():
                writer.add_scalar(f'gan_phase/generator_{key}', value, global_step)
            for key, value in d_loss_dict.items():
                writer.add_scalar(f'gan_phase/discriminator_{key}', value, global_step)
    
    # Compute averages
    g_avg_losses = {key: value / num_batches for key, value in g_total_losses.items()}
    d_avg_losses = {key: value / num_batches for key, value in d_total_losses.items()}
    
    return g_avg_losses, d_avg_losses


def main():
    """Main GAN training function."""
    parser = argparse.ArgumentParser(description='VintageGAN GAN Training')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--generator-checkpoint', type=str, required=True,
                        help='Path to pretrained generator checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-disc-pretrain', action='store_true',
                        help='Skip discriminator pretraining phase')
    parser.add_argument('--train-detectors', action='store_true',
                        help='Train defect detectors before GAN training')
    
    args = parser.parse_args()
    
    # Setup
    print("="*60)
    print("VINTAGEGAN GAN TRAINING")
    print("="*60)
    
    config = setup_training(args.config)
    device = torch.device(args.device)
    
    print(f"\nDevice: {device}")
    print(f"Config: {args.config}")
    print(f"Generator checkpoint: {args.generator_checkpoint}")
    
    # Set random seeds
    if config['reproducibility']['seed'] is not None:
        torch.manual_seed(config['reproducibility']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['reproducibility']['seed'])
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = create_dataloaders(
        args.config,
        defect_generator=apply_vintage_defects
    )
    
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")
    
    # Initialize models
    print("\nInitializing models...")
    
    generator = Generator(
        condition_dim=config['model']['generator']['condition_dim'],
        use_spectral_norm=config['model']['generator']['use_spectral_norm'],
        use_self_attention=config['model']['generator']['use_self_attention'],
        dropout_rate=config['model']['generator']['dropout_rate']
    ).to(device)
    
    discriminator = Discriminator(
        condition_dim=config['model']['discriminator']['in_channels'] - 3,
        base_filters=config['model']['discriminator']['base_filters'],
        use_spectral_norm=config['model']['discriminator']['use_spectral_norm'],
        use_instance_norm=config['model']['discriminator']['use_instance_norm']
    ).to(device)
    
    # Load pretrained generator
    print(f"\nLoading pretrained generator...")
    load_checkpoint(args.generator_checkpoint, generator)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Train detectors if requested
    detectors = None
    if args.train_detectors:
        print("\nTraining defect detectors...")
        detectors = create_detectors('multi', pretrained=True)
        detectors = train_detectors(
            detectors, dataloaders['train'], dataloaders['val'],
            device, num_epochs=5, lr=1e-3
        )
        
        # Save detectors
        detector_dir = Path(config['paths']['checkpoint_dir']) / 'detectors'
        detector_dir.mkdir(parents=True, exist_ok=True)
        torch.save(detectors['multi'].state_dict(), detector_dir / 'multi_detector.pth')
        print(f"Detectors saved to {detector_dir}")
    
    # Initialize loss function
    print("\nInitializing loss function...")
    criterion = VintageGANLoss(
        perceptual_weight=config['training']['phase3_gan']['perceptual_loss_weight'],
        pixel_weight=config['training']['phase3_gan']['pixel_loss_weight'],
        adversarial_weight=config['training']['phase3_gan']['adversarial_loss_weight'],
        consistency_weight=config['training']['phase3_gan']['consistency_loss_weight'],
        detectors=detectors
    )
    
    # Initialize optimizers
    print("\nInitializing optimizers...")
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=config['training']['phase3_gan']['learning_rate'],
        betas=tuple(config['training']['phase3_gan']['betas'])
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config['training']['phase3_gan']['learning_rate'],
        betas=tuple(config['training']['phase3_gan']['betas'])
    )
    
    # TensorBoard writers
    log_dir = Path(config['paths']['log_dir'])
    disc_writer = SummaryWriter(log_dir / 'disc_phase')
    gan_writer = SummaryWriter(log_dir / 'gan_phase')
    
    # =====================================
    # PHASE 2: Discriminator Pretraining
    # =====================================
    if not args.skip_disc_pretrain:
        print("\n" + "="*60)
        print("PHASE 2: DISCRIMINATOR PRETRAINING")
        print("="*60)
        
        num_disc_epochs = config['training']['phase2_critic']['epochs']
        
        for epoch in range(num_disc_epochs):
            epoch_start_time = time.time()
            
            losses = train_discriminator_phase(
                generator, discriminator, dataloaders['train'],
                criterion, d_optimizer, device, epoch, disc_writer, config
            )
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
            print(f"  D Loss: {losses['total']:.4f}, "
                  f"Real: {losses['real']:.4f}, "
                  f"Fake: {losses['fake']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_state_dict': d_optimizer.state_dict(),
                    'losses': losses,
                    'config': config
                }
                checkpoint_path = Path(config['paths']['checkpoint_dir']) / \
                                  f'discriminator_pretrain_epoch_{epoch:03d}.pth'
                torch.save(checkpoint, checkpoint_path)
        
        print(f"\nDiscriminator pretraining complete")
    
    # =============================
    # PHASE 3: GAN Fine-Tuning
    # =============================
    print("\n" + "="*60)
    print("PHASE 3: GAN FINE-TUNING")
    print("="*60)
    
    num_gan_epochs = config['training']['phase3_gan']['epochs']
    best_g_loss = float('inf')
    patience_counter = 0
    patience = config['training']['phase3_gan']['patience']
    
    for epoch in range(num_gan_epochs):
        epoch_start_time = time.time()
        
        g_losses, d_losses = train_gan_phase(
            generator, discriminator, dataloaders['train'],
            criterion, g_optimizer, d_optimizer,
            device, epoch, gan_writer, config
        )
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch} Summary ({epoch_time:.1f}s):")
        print(f"  G Loss: {g_losses['total']:.4f} "
              f"[Pixel: {g_losses['pixel']:.4f}, "
              f"Percept: {g_losses['perceptual']:.4f}, "
              f"Adv: {g_losses['adversarial']:.4f}, "
              f"Consist: {g_losses['consistency']:.4f}]")
        print(f"  D Loss: {d_losses['total']:.4f} "
              f"[Real: {d_losses['real']:.4f}, "
              f"Fake: {d_losses['fake']:.4f}]")
        
        # Save checkpoints
        if (epoch + 1) % 2 == 0:
            # Generator
            g_checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'optimizer_state_dict': g_optimizer.state_dict(),
                'losses': g_losses,
                'config': config
            }
            torch.save(g_checkpoint, 
                      Path(config['paths']['checkpoint_dir']) / f'generator_gan_epoch_{epoch:03d}.pth')
            
            # Discriminator
            d_checkpoint = {
                'epoch': epoch,
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': d_optimizer.state_dict(),
                'losses': d_losses,
                'config': config
            }
            torch.save(d_checkpoint,
                      Path(config['paths']['checkpoint_dir']) / f'discriminator_gan_epoch_{epoch:03d}.pth')
        
        # Save best generator
        if g_losses['total'] < best_g_loss:
            best_g_loss = g_losses['total']
            patience_counter = 0
            
            g_checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'optimizer_state_dict': g_optimizer.state_dict(),
                'losses': g_losses,
                'config': config
            }
            torch.save(g_checkpoint,
                      Path(config['paths']['checkpoint_dir']) / 'generator_gan_best.pth')
            print(f"  ✓ Best generator saved (loss: {best_g_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping (spec: patience=3)
        if patience_counter >= patience:
            print(f"\nEarly stopping: FID plateaued for {patience} epochs")
            break
    
    # Final checkpoints
    torch.save(generator.state_dict(),
              Path(config['paths']['checkpoint_dir']) / 'generator_final.pth')
    torch.save(discriminator.state_dict(),
              Path(config['paths']['checkpoint_dir']) / 'discriminator_final.pth')
    
    print("\n" + "="*60)
    print("GAN TRAINING COMPLETE")
    print(f"Best generator loss: {best_g_loss:.4f}")
    print(f"Models saved: {config['paths']['checkpoint_dir']}")
    print("="*60)
    
    disc_writer.close()
    gan_writer.close()


if __name__ == "__main__":
    main()
