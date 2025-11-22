"""
VintageGAN Project Setup Script

This script helps with initial project setup and validation.

Usage:
    python setup_project.py --mode [setup|validate|test]
    
Modes:
    setup: Create dummy dataset and verify installation
    validate: Validate existing setup
    test: Run unit tests

Author: VintageGAN Project
Date: 2024
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def check_dependencies() -> bool:
    """Check if all required packages are installed."""
    print_header("CHECKING DEPENDENCIES")
    
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'PIL',
        'cv2',
        'yaml',
        'tqdm',
        'pytest'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def check_directory_structure() -> bool:
    """Verify project directory structure exists."""
    print_header("CHECKING DIRECTORY STRUCTURE")
    
    required_dirs = [
        'data',
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
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (missing)")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some directories are missing!")
        print("Run this script to create them.")
        return False
    
    print("\n✅ All directories exist!")
    return True


def check_config_files() -> bool:
    """Check if configuration files exist."""
    print_header("CHECKING CONFIGURATION FILES")
    
    project_root = Path(__file__).parent
    config_file = project_root / "configs" / "training_config.yaml"
    
    if config_file.exists():
        print(f"✓ training_config.yaml")
        
        # Try to load it
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Configuration file is valid YAML")
            print(f"  - Batch size: {config['hardware']['batch_size']}")
            print(f"  - Image size: {config['dataset']['image_size']}")
        except Exception as e:
            print(f"✗ Error loading config: {str(e)}")
            return False
    else:
        print(f"✗ training_config.yaml (missing)")
        return False
    
    print("\n✅ Configuration files OK!")
    return True


def setup_dummy_dataset() -> bool:
    """Create dummy dataset for testing."""
    print_header("SETTING UP DUMMY DATASET")
    
    try:
        from training.download_data import create_dummy_dataset
        
        project_root = Path(__file__).parent
        dataset_dir = project_root / "data" / "imagenet_subset"
        
        print("Creating dummy dataset (100 train, 20 val images)...")
        create_dummy_dataset(
            str(dataset_dir),
            num_train=100,
            num_val=20,
            image_size=512
        )
        
        print("\n✅ Dummy dataset created successfully!")
        return True
    
    except Exception as e:
        print(f"\n❌ Error creating dummy dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader() -> bool:
    """Test that dataloader works correctly."""
    print_header("TESTING DATALOADER")
    
    try:
        from training.dataset import create_dataloaders
        import yaml
        
        project_root = Path(__file__).parent
        config_path = project_root / "configs" / "training_config.yaml"
        
        print("Loading configuration...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update config to point to correct dataset path
        dataset_path = project_root / "data" / "imagenet_subset"
        config['dataset']['imagenet_path'] = str(dataset_path)
        
        # Save temporary config
        temp_config_path = project_root / "configs" / "temp_config.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        print("Creating dataloaders (without defect generator)...")
        dataloaders = create_dataloaders(str(temp_config_path), defect_generator=None)
        
        print(f"✓ Train dataloader: {len(dataloaders['train'])} batches")
        print(f"✓ Val dataloader: {len(dataloaders['val'])} batches")
        
        # Test loading one batch
        print("\nLoading sample batch...")
        train_batch = next(iter(dataloaders['train']))
        images, paths = train_batch
        
        print(f"✓ Batch shape: {images.shape}")
        print(f"✓ Batch value range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Clean up temp config
        temp_config_path.unlink()
        
        print("\n✅ Dataloader test passed!")
        return True
    
    except Exception as e:
        print(f"\n❌ Dataloader test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_unit_tests() -> bool:
    """Run unit tests with pytest."""
    print_header("RUNNING UNIT TESTS")
    
    project_root = Path(__file__).parent
    test_file = project_root / "tests" / "test_dataset.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        print("Running pytest...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            return True
        else:
            print("\n❌ Some tests failed!")
            return False
    
    except Exception as e:
        print(f"\n❌ Error running tests: {str(e)}")
        return False


def setup_mode():
    """Run full setup process."""
    print_header("VINTAGEGAN PROJECT SETUP")
    
    steps = [
        ("Checking dependencies", check_dependencies),
        ("Checking directory structure", check_directory_structure),
        ("Checking configuration files", check_config_files),
        ("Setting up dummy dataset", setup_dummy_dataset),
        ("Testing dataloader", test_dataloader),
    ]
    
    results = {}
    for step_name, step_func in steps:
        try:
            success = step_func()
            results[step_name] = success
        except Exception as e:
            print(f"\n❌ Error in '{step_name}': {str(e)}")
            results[step_name] = False
    
    # Print summary
    print_header("SETUP SUMMARY")
    
    for step_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 Setup complete! You can now proceed to Day 3-4 implementation.")
        print("\nNext steps:")
        print("  1. Implement defect synthesis functions (Day 3-4)")
        print("  2. Run: python setup_project.py --mode test")
    else:
        print("\n⚠️  Some setup steps failed. Please fix the issues above.")
    
    return all_passed


def validate_mode():
    """Validate existing setup."""
    print_header("VINTAGEGAN SETUP VALIDATION")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Directory structure", check_directory_structure),
        ("Configuration files", check_config_files),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            success = check_func()
            results[check_name] = success
        except Exception as e:
            print(f"\n❌ Error in '{check_name}': {str(e)}")
            results[check_name] = False
    
    # Print summary
    print_header("VALIDATION SUMMARY")
    
    for check_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {check_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All validation checks passed!")
    else:
        print("\n❌ Some validation checks failed!")
    
    return all_passed


def test_mode():
    """Run unit tests."""
    return run_unit_tests()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VintageGAN Project Setup and Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        choices=['setup', 'validate', 'test'],
        default='setup',
        help='Operation mode: setup (full setup), validate (check setup), test (run tests)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'setup':
        success = setup_mode()
    elif args.mode == 'validate':
        success = validate_mode()
    elif args.mode == 'test':
        success = test_mode()
    else:
        print(f"Unknown mode: {args.mode}")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
