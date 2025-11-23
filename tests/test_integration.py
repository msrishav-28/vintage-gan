"""
VintageGAN Integration Tests
Tests complete pipeline from data to models

This validates Days 1-7 implementation by testing:
- Data loading with defect generation
- Generator forward pass
- Discriminator forward pass
- Training step simulation

Author: VintageGAN Project
Date: 2024
"""

import os
import sys
from pathlib import Path
import tempfile

import pytest
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_defect_integration():
    """Test that all defect functions work together."""
    print("\n" + "="*60)
    print("TEST: Defect Integration")
    print("="*60)
    
    try:
        from defects import (
            apply_vintage_defects,
            generate_random_conditions,
            create_preset_conditions
        )
        
        # Create test image
        test_image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
        
        # Test random conditions
        conditions = generate_random_conditions()
        defected = apply_vintage_defects(test_image, conditions)
        
        assert defected.shape == test_image.shape
        assert defected.dtype == np.uint8
        print("✓ Random conditions: OK")
        
        # Test all presets
        presets = ['light', 'medium', 'heavy', 'grain_only', 'faded', 'scratched']
        for preset in presets:
            cond = create_preset_conditions(preset)
            defected = apply_vintage_defects(test_image, cond)
            assert defected.shape == test_image.shape
        
        print(f"✓ All {len(presets)} presets: OK")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"✗ Defect integration failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_dataset_with_defects():
    """Test dataset integration with defect generator."""
    print("\n" + "="*60)
    print("TEST: Dataset with Defects")
    print("="*60)
    
    try:
        from training.dataset import ImageNetSubsetDataset, VintageGANDataset
        from training.download_data import create_dummy_dataset
        from defects import apply_vintage_defects
        
        # Create temporary dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "imagenet_subset"
            create_dummy_dataset(str(dataset_dir), num_train=5, num_val=2, image_size=512)
            
            # Create clean dataset
            clean_dataset = ImageNetSubsetDataset(
                root_dir=str(dataset_dir),
                split='train',
                image_size=512
            )
            
            print(f"✓ Clean dataset: {len(clean_dataset)} images")
            
            # Create VintageGAN dataset with defect generator
            vintage_dataset = VintageGANDataset(
                clean_dataset=clean_dataset,
                defect_generator=apply_vintage_defects,
                num_variants=2,
                return_clean=True
            )
            
            print(f"✓ VintageGAN dataset: {len(vintage_dataset)} pairs")
            
            # Load a sample
            sample = vintage_dataset[0]
            
            assert 'clean' in sample
            assert 'defected' in sample
            assert 'condition' in sample
            
            assert sample['clean'].shape == (3, 512, 512)
            assert sample['defected'].shape == (3, 512, 512)
            assert sample['condition'].shape == (6,)
            
            print(f"✓ Sample shapes: clean={sample['clean'].shape}, "
                  f"defected={sample['defected'].shape}, "
                  f"condition={sample['condition'].shape}")
            print("="*60 + "\n")
    
    except Exception as e:
        print(f"✗ Dataset integration failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_models_basic():
    """Test basic model functionality."""
    print("\n" + "="*60)
    print("TEST: Models Basic Functionality")
    print("="*60)
    
    try:
        import torch
        from models import Generator, Discriminator
        
        # Test Generator
        print("\nGenerator:")
        gen = Generator()
        img = torch.randn(1, 3, 512, 512)
        cond = torch.rand(1, 6)
        
        gen.eval()
        with torch.no_grad():
            out = gen(img, cond)
        
        assert out.shape == img.shape
        print(f"✓ Forward pass: {img.shape} -> {out.shape}")
        print(f"✓ Output range: [{out.min():.3f}, {out.max():.3f}]")
        
        # Test Discriminator
        print("\nDiscriminator:")
        disc = Discriminator()
        
        disc.eval()
        with torch.no_grad():
            pred = disc(img, cond)
        
        assert pred.shape == (1, 1, 32, 32)
        print(f"✓ Forward pass: {img.shape} -> {pred.shape}")
        print(f"✓ Patch predictions: 32×32 patches")
        
        print("="*60 + "\n")
    
    except Exception as e:
        print(f"✗ Model tests failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_full_pipeline():
    """Test complete pipeline from data to model."""
    print("\n" + "="*60)
    print("TEST: Full Pipeline Integration")
    print("="*60)
    
    try:
        import torch
        from torch.utils.data import DataLoader
        from training.dataset import ImageNetSubsetDataset, VintageGANDataset
        from training.download_data import create_dummy_dataset
        from defects import apply_vintage_defects
        from models import Generator, Discriminator
        
        # 1. Create dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "imagenet_subset"
            create_dummy_dataset(str(dataset_dir), num_train=4, num_val=2)
            
            clean_dataset = ImageNetSubsetDataset(str(dataset_dir), split='train')
            vintage_dataset = VintageGANDataset(
                clean_dataset=clean_dataset,
                defect_generator=apply_vintage_defects,
                num_variants=1,
                return_clean=True
            )
            
            dataloader = DataLoader(vintage_dataset, batch_size=2, shuffle=False)
            print(f"✓ Created dataloader: {len(dataloader)} batches")
            
            # 2. Initialize models
            gen = Generator()
            disc = Discriminator()
            gen.eval()
            disc.eval()
            print("✓ Initialized models")
            
            # 3. Process one batch
            batch = next(iter(dataloader))
            clean = batch['clean']
            defected = batch['defected']
            conditions = batch['condition']
            
            print(f"✓ Loaded batch: clean={clean.shape}, defected={defected.shape}, "
                  f"conditions={conditions.shape}")
            
            # 4. Generator forward
            with torch.no_grad():
                generated = gen(clean, conditions)
            
            assert generated.shape == clean.shape
            print(f"✓ Generator output: {generated.shape}")
            
            # 5. Discriminator on real and fake
            with torch.no_grad():
                real_pred = disc(defected, conditions)
                fake_pred = disc(generated, conditions)
            
            assert real_pred.shape == (2, 1, 32, 32)
            assert fake_pred.shape == (2, 1, 32, 32)
            print(f"✓ Discriminator predictions: real={real_pred.shape}, fake={fake_pred.shape}")
            
            # 6. Simulate training step
            gen.train()
            disc.train()
            
            # Forward passes
            generated = gen(clean, conditions)
            real_pred = disc(defected, conditions)
            fake_pred = disc(generated.detach(), conditions)
            
            # Calculate dummy losses
            real_labels = torch.ones_like(real_pred)
            fake_labels = torch.zeros_like(fake_pred)
            
            import torch.nn.functional as F
            d_loss_real = F.binary_cross_entropy_with_logits(real_pred, real_labels)
            d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            
            print(f"✓ Discriminator loss: {d_loss.item():.4f}")
            
            # Generator loss
            fake_pred_for_g = disc(generated, conditions)
            g_loss = F.binary_cross_entropy_with_logits(fake_pred_for_g, real_labels)
            
            print(f"✓ Generator loss: {g_loss.item():.4f}")
            
            # Test backward
            d_loss.backward()
            print("✓ Discriminator backward: OK")
            
            g_loss.backward()
            print("✓ Generator backward: OK")
            
            print("\n" + "="*60)
            print("✅ FULL PIPELINE TEST PASSED")
            print("="*60 + "\n")
    
    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_condition_control():
    """Test that conditioning actually controls defects."""
    print("\n" + "="*60)
    print("TEST: Condition Control")
    print("="*60)
    
    try:
        import torch
        from models import Generator
        
        gen = Generator()
        gen.eval()
        
        img = torch.randn(1, 3, 512, 512)
        
        # Test with different conditions
        cond_zero = torch.zeros(1, 6)
        cond_ones = torch.ones(1, 6)
        cond_half = torch.full((1, 6), 0.5)
        
        with torch.no_grad():
            out_zero = gen(img, cond_zero)
            out_ones = gen(img, cond_ones)
            out_half = gen(img, cond_half)
        
        # Outputs should be different
        diff_01 = (out_zero - out_ones).abs().mean().item()
        diff_0h = (out_zero - out_half).abs().mean().item()
        diff_1h = (out_ones - out_half).abs().mean().item()
        
        print(f"✓ Diff (zero vs ones): {diff_01:.6f}")
        print(f"✓ Diff (zero vs half): {diff_0h:.6f}")
        print(f"✓ Diff (ones vs half): {diff_1h:.6f}")
        
        assert diff_01 > 0.001, "Different conditions should produce different outputs"
        assert diff_0h > 0.001, "Different conditions should produce different outputs"
        
        print("✓ Conditions control output: VERIFIED")
        print("="*60 + "\n")
    
    except Exception as e:
        print(f"✗ Condition control test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_memory_efficiency():
    """Test model memory usage."""
    print("\n" + "="*60)
    print("TEST: Memory Efficiency")
    print("="*60)
    
    try:
        import torch
        from models import Generator, Discriminator
        
        gen = Generator()
        disc = Discriminator()
        
        # Count parameters
        gen_params = sum(p.numel() for p in gen.parameters())
        disc_params = sum(p.numel() for p in disc.parameters())
        
        # Estimate memory (FP32)
        gen_size_mb = gen_params * 4 / 1024 / 1024
        disc_size_mb = disc_params * 4 / 1024 / 1024
        total_size_mb = gen_size_mb + disc_size_mb
        
        print(f"✓ Generator parameters: {gen_params:,} ({gen_size_mb:.1f} MB)")
        print(f"✓ Discriminator parameters: {disc_params:,} ({disc_size_mb:.1f} MB)")
        print(f"✓ Total model size: {total_size_mb:.1f} MB")
        
        # Check if reasonable for 6GB GPU (with FP16 would be ~2-3GB)
        assert total_size_mb < 1000, f"Models too large: {total_size_mb:.1f} MB"
        print("✓ Memory footprint acceptable for RTX 3050")
        
        print("="*60 + "\n")
    
    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    """Run all integration tests."""
    print("\n" + "#"*60)
    print("# VINTAGEGAN INTEGRATION TESTS (DAYS 1-7)")
    print("#"*60 + "\n")
    
    tests = [
        ("Defect Integration", test_defect_integration),
        ("Dataset with Defects", test_dataset_with_defects),
        ("Models Basic", test_models_basic),
        ("Condition Control", test_condition_control),
        ("Memory Efficiency", test_memory_efficiency),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}\n")
            failed += 1
    
    print("\n" + "#"*60)
    print(f"# RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("# ✅ ALL INTEGRATION TESTS PASSED")
    else:
        print(f"# ❌ {failed} test(s) failed")
    print("#"*60 + "\n")
    
    sys.exit(0 if failed == 0 else 1)
