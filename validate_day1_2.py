"""
Day 1-2 Validation Script (Simplified)

This script validates the Day 1-2 implementation without requiring
all dependencies to be perfectly configured.

Validates:
1. Directory structure exists
2. Configuration file is valid
3. Dataset module can be imported
4. Dummy data can be created and loaded

Author: VintageGAN Project
Date: 2024
"""

import os
import sys
from pathlib import Path


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def check_directory_structure() -> bool:
    """Verify project directory structure exists."""
    print_header("DAY 1-2 VALIDATION: DIRECTORY STRUCTURE")
    
    required_dirs = [
        'data/imagenet_subset/train',
        'data/imagenet_subset/val',
        'data/filmset',
        'data/synthetic_pairs',
        'data/validation',
        'models',
        'training',
        'defects',
        'evaluation',
        'inference',
        'notebooks',
        'configs',
        'checkpoints',
        'logs',
        'outputs',
        'tests'
    ]
    
    project_root = Path(__file__).parent
    all_exist = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required directories exist!")
    else:
        print("\n⚠️  Some directories are missing (non-critical)")
    
    return all_exist


def check_files() -> bool:
    """Check if key files exist."""
    print_header("DAY 1-2 VALIDATION: KEY FILES")
    
    required_files = [
        'requirements.txt',
        'README.md',
        'plan.md',
        'configs/training_config.yaml',
        'training/__init__.py',
        'training/dataset.py',
        'training/download_data.py',
        'tests/__init__.py',
        'tests/test_dataset.py',
        'models/__init__.py',
        'defects/__init__.py',
        'setup_project.py',
    ]
    
    project_root = Path(__file__).parent
    all_exist = True
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            print(f"✓ {file_path} ({size_kb:.1f} KB)")
        else:
            print(f"✗ {file_path} (missing)")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required files exist!")
    else:
        print("\n❌ Some files are missing!")
    
    return all_exist


def check_config() -> bool:
    """Validate configuration file."""
    print_header("DAY 1-2 VALIDATION: CONFIGURATION FILE")
    
    try:
        import yaml
        
        project_root = Path(__file__).parent
        config_file = project_root / "configs" / "training_config.yaml"
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Configuration file loaded successfully")
        
        # Check key sections exist
        required_sections = ['dataset', 'model', 'training', 'hardware', 'evaluation']
        for section in required_sections:
            if section in config:
                print(f"✓ Section '{section}' exists")
            else:
                print(f"✗ Section '{section}' missing")
                return False
        
        # Display key settings
        print("\nKey Settings:")
        print(f"  - Image size: {config['dataset']['image_size']}")
        print(f"  - Batch size: {config['hardware']['batch_size']}")
        print(f"  - Mixed precision: {config['hardware']['mixed_precision']}")
        print(f"  - Generator base: {config['model']['generator']['base_model']}")
        print(f"  - Discriminator type: {config['model']['discriminator']['type']}")
        
        print("\n✅ Configuration file is valid!")
        return True
    
    except Exception as e:
        print(f"❌ Error loading configuration: {str(e)}")
        return False


def test_basic_imports() -> bool:
    """Test that basic Python modules can be imported."""
    print_header("DAY 1-2 VALIDATION: BASIC IMPORTS")
    
    basic_modules = [
        'os',
        'sys',
        'pathlib',
        'numpy',
        'yaml',
    ]
    
    all_ok = True
    for module_name in basic_modules:
        try:
            if module_name == 'pathlib':
                from pathlib import Path
            elif module_name == 'yaml':
                import yaml
            else:
                __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError:
            print(f"✗ {module_name} (not installed)")
            all_ok = False
    
    if all_ok:
        print("\n✅ All basic modules can be imported!")
    else:
        print("\n❌ Some modules are missing!")
    
    return all_ok


def create_test_dataset() -> bool:
    """Create a small test dataset."""
    print_header("DAY 1-2 VALIDATION: CREATING TEST DATASET")
    
    try:
        import numpy as np
        from PIL import Image
        from pathlib import Path
        
        project_root = Path(__file__).parent
        train_dir = project_root / "data" / "imagenet_subset" / "train"
        val_dir = project_root / "data" / "imagenet_subset" / "val"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 10 train images
        print("Creating 10 training images...")
        for i in range(10):
            img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='RGB')
            img.save(train_dir / f"train_{i:04d}.jpg", quality=95)
        
        # Create 5 val images
        print("Creating 5 validation images...")
        for i in range(5):
            img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='RGB')
            img.save(val_dir / f"val_{i:04d}.jpg", quality=95)
        
        print(f"\n✓ Created 10 training images in {train_dir}")
        print(f"✓ Created 5 validation images in {val_dir}")
        print("\n✅ Test dataset created successfully!")
        return True
    
    except Exception as e:
        print(f"\n❌ Error creating test dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_module() -> bool:
    """Test that dataset module works."""
    print_header("DAY 1-2 VALIDATION: DATASET MODULE")
    
    try:
        print("Importing dataset module...")
        sys.path.insert(0, str(Path(__file__).parent))
        from training.dataset import ImageNetSubsetDataset
        print("✓ ImageNetSubsetDataset imported")
        
        project_root = Path(__file__).parent
        dataset_path = project_root / "data" / "imagenet_subset"
        
        print("\nCreating train dataset...")
        train_dataset = ImageNetSubsetDataset(
            root_dir=str(dataset_path),
            split='train',
            image_size=512
        )
        print(f"✓ Train dataset created: {len(train_dataset)} images")
        
        print("\nCreating val dataset...")
        val_dataset = ImageNetSubsetDataset(
            root_dir=str(dataset_path),
            split='val',
            image_size=512
        )
        print(f"✓ Val dataset created: {len(val_dataset)} images")
        
        if len(train_dataset) > 0:
            print("\nLoading sample image...")
            image, path = train_dataset[0]
            print(f"✓ Image shape: {image.shape}")
            print(f"✓ Image dtype: {image.dtype}")
            print(f"✓ Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"✓ Image path: {Path(path).name}")
            
            # Validate shape
            assert image.shape == (3, 512, 512), f"Expected (3, 512, 512), got {image.shape}"
            print("✓ Image shape validation passed")
            
            # Validate range (should be approximately [-1, 1])
            assert image.min() >= -1.5, f"Min value {image.min()} too low"
            assert image.max() <= 1.5, f"Max value {image.max()} too high"
            print("✓ Image range validation passed")
        
        print("\n✅ Dataset module works correctly!")
        return True
    
    except Exception as e:
        print(f"\n❌ Dataset module test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results: dict) -> bool:
    """Print final summary."""
    print_header("DAY 1-2 IMPLEMENTATION SUMMARY")
    
    print("Validation Results:\n")
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    all_passed = all(results.values())
    critical_tests = ['Files', 'Configuration', 'Test Dataset', 'Dataset Module']
    critical_passed = all(results.get(test, False) for test in critical_tests)
    
    print(f"\n{'='*60}")
    
    if all_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("\nDay 1-2 Implementation: COMPLETE ✅")
        print("\nNext Steps:")
        print("  → Proceed to Day 3-4: Defect Synthesis Implementation")
        print("  → Implement grain, scratches, dust, vignette, color shift, blur")
    elif critical_passed:
        print("✅ CRITICAL TESTS PASSED!")
        print("\nDay 1-2 Implementation: FUNCTIONAL ✅")
        print("\nSome optional tests failed, but core functionality works.")
        print("\nNext Steps:")
        print("  → Proceed to Day 3-4: Defect Synthesis Implementation")
    else:
        print("❌ SOME CRITICAL TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
    
    print(f"{'='*60}\n")
    
    return critical_passed


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("  VINTAGEGAN DAY 1-2 VALIDATION")
    print("  Project Setup and Data Pipeline")
    print("="*60)
    
    results = {}
    
    # Run validation tests
    results['Directory Structure'] = check_directory_structure()
    results['Files'] = check_files()
    results['Configuration'] = check_config()
    results['Basic Imports'] = test_basic_imports()
    results['Test Dataset'] = create_test_dataset()
    results['Dataset Module'] = test_dataset_module()
    
    # Print summary
    success = print_summary(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
