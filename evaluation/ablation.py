"""
Ablation Study for VintageGAN
Specification Reference: Section 5.2

Tests importance of each model component by comparing variants.

Author: VintageGAN Project
Date: 2024
"""

import argparse
import sys
from pathlib import Path
from typing import Dict
import json

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Generator, Discriminator
from training.dataset import create_dataloaders
from evaluation.metrics import calculate_ssim, calculate_psnr, evaluate_model
from defects import apply_vintage_defects


def create_ablation_variants():
    """
    Create model variants for ablation study.
    
    Specification Reference: Section 5.2
    
    Variants:
    1. Baseline: U-Net without conditioning
    2. No Consistency Loss: Full model but trained without consistency
    3. No GAN Training: Pretraining only (NoGAN extreme)
    4. No Self-Attention: Generator without attention module
    5. Single Condition: Only grain control
    6. Full Model: Complete architecture
    
    Returns:
        Dictionary of variant names and model paths
    """
    variants = {
        'baseline': {
            'description': 'U-Net without conditioning',
            'checkpoint': 'checkpoints/ablation_baseline.pth',
            'features': {
                'conditioning': False,
                'self_attention': False,
                'consistency_loss': False,
                'gan_training': False
            }
        },
        'no_consistency': {
            'description': 'Full model without consistency loss',
            'checkpoint': 'checkpoints/ablation_no_consistency.pth',
            'features': {
                'conditioning': True,
                'self_attention': True,
                'consistency_loss': False,
                'gan_training': True
            }
        },
        'no_gan': {
            'description': 'Pretraining only (no GAN fine-tuning)',
            'checkpoint': 'checkpoints/generator_pretrain_best.pth',
            'features': {
                'conditioning': True,
                'self_attention': True,
                'consistency_loss': False,
                'gan_training': False
            }
        },
        'no_attention': {
            'description': 'Generator without self-attention',
            'checkpoint': 'checkpoints/ablation_no_attention.pth',
            'features': {
                'conditioning': True,
                'self_attention': False,
                'consistency_loss': True,
                'gan_training': True
            }
        },
        'single_condition': {
            'description': 'Only grain control (1D conditioning)',
            'checkpoint': 'checkpoints/ablation_single_condition.pth',
            'features': {
                'conditioning': True,  # But limited to 1D
                'self_attention': True,
                'consistency_loss': True,
                'gan_training': True
            }
        },
        'full_model': {
            'description': 'Complete architecture with all components',
            'checkpoint': 'checkpoints/generator_final.pth',
            'features': {
                'conditioning': True,
                'self_attention': True,
                'consistency_loss': True,
                'gan_training': True
            }
        }
    }
    
    return variants


def load_model_variant(checkpoint_path: str, device: str = 'cuda') -> Generator:
    """
    Load a model variant from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded generator model
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Initialize generator (adjust parameters based on variant if needed)
    generator = Generator().to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)
        
        generator.eval()
        print(f"✓ Loaded model from: {checkpoint_path}")
        return generator
    
    except FileNotFoundError:
        print(f"✗ Model not found: {checkpoint_path}")
        print(f"   Train this variant first!")
        return None


def evaluate_variant(
    generator: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    variant_name: str,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate a model variant.
    
    Args:
        generator: Generator model
        dataloader: Test dataloader
        variant_name: Name of variant
        device: Device
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {variant_name}")
    print(f"{'='*60}")
    
    generator.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    
    # Collect generated and target images
    import numpy as np
    generated_images = []
    target_images = []
    
    print("Generating test images...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            if isinstance(batch, dict):
                clean = batch['clean'].to(device)
                defected = batch['defected']
                conditions = batch['condition'].to(device)
            else:
                images, _ = batch
                clean = images.to(device)
                defected = images
                conditions = torch.rand(clean.size(0), 6).to(device)
            
            # Generate
            generated = generator(clean, conditions)
            
            # Convert to numpy uint8
            gen_np = ((generated + 1) / 2 * 255).clamp(0, 255).byte().cpu()
            gen_np = gen_np.permute(0, 2, 3, 1).numpy()
            
            tgt_np = ((defected + 1) / 2 * 255).clamp(0, 255).byte()
            tgt_np = tgt_np.permute(0, 2, 3, 1).numpy()
            
            generated_images.append(gen_np)
            target_images.append(tgt_np)
    
    generated_images = np.concatenate(generated_images, axis=0)
    target_images = np.concatenate(target_images, axis=0)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    ssim = calculate_ssim(generated_images, target_images)
    psnr = calculate_psnr(generated_images, target_images)
    
    metrics = {
        'variant': variant_name,
        'ssim': ssim,
        'psnr': psnr,
        'num_samples': len(generated_images)
    }
    
    print(f"Results:")
    print(f"  SSIM: {ssim:.4f} (target: >0.75)")
    print(f"  PSNR: {psnr:.2f} dB (target: >22 dB)")
    
    return metrics


def run_ablation_study(
    config_path: str,
    variants: list = None,
    device: str = 'cuda',
    output_file: str = 'ablation_results.json'
):
    """
    Run complete ablation study.
    
    Args:
        config_path: Path to config file
        variants: List of variant names to test (None = all)
        device: Device to use
        output_file: Path to save results
    """
    print("="*60)
    print("VINTAGEGAN ABLATION STUDY")
    print("="*60)
    
    # Get all variants
    all_variants = create_ablation_variants()
    
    # Filter variants if specified
    if variants is not None:
        all_variants = {k: v for k, v in all_variants.items() if k in variants}
    
    print(f"\nTesting {len(all_variants)} variants:")
    for name, info in all_variants.items():
        print(f"  • {name}: {info['description']}")
    
    # Create dataloader
    print(f"\nLoading test data from: {config_path}")
    dataloaders = create_dataloaders(
        config_path,
        defect_generator=apply_vintage_defects
    )
    test_loader = dataloaders['val']
    
    # Evaluate each variant
    results = []
    
    for variant_name, variant_info in all_variants.items():
        checkpoint_path = variant_info['checkpoint']
        
        # Load model
        generator = load_model_variant(checkpoint_path, device)
        
        if generator is None:
            print(f"  Skipping {variant_name} (model not found)\n")
            continue
        
        # Evaluate
        metrics = evaluate_variant(generator, test_loader, variant_name, device)
        
        # Add variant info
        metrics.update(variant_info['features'])
        results.append(metrics)
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print(f"{'Variant':<20} {'SSIM':<10} {'PSNR (dB)':<12} {'Description':<35}")
    print("="*80)
    
    # Sort by SSIM (descending)
    results_sorted = sorted(results, key=lambda x: x['ssim'], reverse=True)
    
    for result in results_sorted:
        variant_info = all_variants[result['variant']]
        print(f"{result['variant']:<20} {result['ssim']:<10.4f} "
              f"{result['psnr']:<12.2f} {variant_info['description']:<35}")
    
    print("="*80)
    
    # Identify best model
    best = results_sorted[0]
    print(f"\n🏆 Best Model: {best['variant']}")
    print(f"   SSIM: {best['ssim']:.4f}")
    print(f"   PSNR: {best['psnr']:.2f} dB")
    
    return results


def main():
    """Command-line interface for ablation study."""
    parser = argparse.ArgumentParser(description='VintageGAN Ablation Study')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to config file')
    parser.add_argument('--variants', type=str, nargs='+', default=None,
                        help='Specific variants to test (default: all)')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='ablation_results.json',
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    # Run ablation study
    run_ablation_study(
        args.config,
        variants=args.variants,
        device=args.device,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
