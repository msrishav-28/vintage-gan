#!/usr/bin/env python
"""
VintageGAN Installation Verification Script

Verifies that all components are correctly installed and integrated.

Usage:
    python verify_installation.py

Author: VintageGAN Project
Date: 2024
"""

import sys
import importlib
from pathlib import Path
from typing import List, Tuple


def check_python_version() -> bool:
    """Check Python version >= 3.9"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro} (requires >= 3.9)")
        return False


def check_dependencies() -> Tuple[List[str], List[str]]:
    """Check required dependencies."""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)
    
    required = [
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'PIL',
        'yaml',
        'tqdm',
    ]
    
    installed = []
    missing = []
    
    for pkg in required:
        # Handle package name variations
        import_name = pkg
        if pkg == 'opencv-python':
            import_name = 'cv2'
        elif pkg == 'PIL':
            import_name = 'PIL'
        elif pkg == 'yaml':
            import_name = 'yaml'
        
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {pkg:20s} (version: {version})")
            installed.append(pkg)
        except ImportError:
            print(f"❌ {pkg:20s} (NOT INSTALLED)")
            missing.append(pkg)
    
    return installed, missing


def check_project_structure() -> bool:
    """Check project directory structure."""
    print("\n" + "="*60)
    print("Checking Project Structure")
    print("="*60)
    
    project_root = Path(__file__).parent
    
    required_dirs = [
        'models',
        'training',
        'defects',
        'evaluation',
        'inference',
        'tests',
        'configs',
        'notebooks',
    ]
    
    all_exist = True
    for dirname in required_dirs:
        dirpath = project_root / dirname
        if dirpath.exists():
            print(f"✅ {dirname:20s} directory exists")
        else:
            print(f"❌ {dirname:20s} directory MISSING")
            all_exist = False
    
    return all_exist


def check_module_imports() -> bool:
    """Check that all modules can be imported."""
    print("\n" + "="*60)
    print("Checking Module Imports")
    print("="*60)
    
    modules_to_test = [
        ('models', ['Generator', 'Discriminator', 'SelfAttention', 'DefectDetector']),
        ('defects', ['apply_vintage_defects', 'generate_film_grain', 'generate_scratches']),
        ('training', ['VintageGANLoss', 'VGGPerceptualLoss', 'create_dataloaders']),
        ('evaluation', ['calculate_fid', 'calculate_ssim', 'calculate_psnr']),
        ('inference', ['VintageFilter']),
    ]
    
    all_ok = True
    
    for module_name, items in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            missing_items = []
            
            for item in items:
                if not hasattr(module, item):
                    missing_items.append(item)
            
            if missing_items:
                print(f"⚠️  {module_name:20s} - Missing: {', '.join(missing_items)}")
                all_ok = False
            else:
                print(f"✅ {module_name:20s} - All items present")
        
        except ImportError as e:
            print(f"❌ {module_name:20s} - Import failed: {e}")
            all_ok = False
    
    return all_ok


def check_config_files() -> bool:
    """Check that configuration files exist."""
    print("\n" + "="*60)
    print("Checking Configuration Files")
    print("="*60)
    
    project_root = Path(__file__).parent
    
    config_files = [
        'configs/training_config.yaml',
        'requirements.txt',
        'README.md',
        'LICENSE',
    ]
    
    all_exist = True
    for config_file in config_files:
        filepath = project_root / config_file
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✅ {config_file:35s} ({size:,} bytes)")
        else:
            print(f"❌ {config_file:35s} MISSING")
            all_exist = False
    
    return all_exist


def check_model_instantiation() -> bool:
    """Try to instantiate models."""
    print("\n" + "="*60)
    print("Checking Model Instantiation")
    print("="*60)
    
    try:
        import torch
        from models import Generator, Discriminator
        
        # Generator
        print("Testing Generator...")
        gen = Generator()
        test_img = torch.randn(1, 3, 512, 512)
        test_cond = torch.rand(1, 6)
        
        with torch.no_grad():
            output = gen(test_img, test_cond)
        
        if output.shape == test_img.shape:
            print(f"✅ Generator instantiation OK (output shape: {output.shape})")
        else:
            print(f"❌ Generator output shape incorrect: {output.shape}")
            return False
        
        # Discriminator
        print("Testing Discriminator...")
        disc = Discriminator()
        
        with torch.no_grad():
            pred = disc(output, test_cond)
        
        # Accept 31x31 or 32x32 depending on architecture
        if pred.shape[0] == 1 and pred.shape[1] == 1 and pred.shape[2] in [31, 32]:
            print(f"✅ Discriminator instantiation OK (output shape: {pred.shape})")
        else:
            print(f"❌ Discriminator output shape incorrect: {pred.shape}")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("="*60)
    print("VINTAGEGAN INSTALLATION VERIFICATION")
    print("="*60)
    
    checks = []
    
    # Check 1: Python version
    checks.append(("Python Version", check_python_version()))
    
    # Check 2: Dependencies
    installed, missing = check_dependencies()
    checks.append(("Dependencies", len(missing) == 0))
    
    # Check 3: Project structure
    checks.append(("Project Structure", check_project_structure()))
    
    # Check 4: Config files
    checks.append(("Config Files", check_config_files()))
    
    # Check 5: Module imports (only if PyTorch installed)
    if 'torch' in installed:
        checks.append(("Module Imports", check_module_imports()))
        checks.append(("Model Instantiation", check_model_instantiation()))
    else:
        print("\n⚠️  Skipping module and model tests (PyTorch not installed)")
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:30s} {status}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n✅ VintageGAN is correctly installed and integrated.")
        print("\nNext steps:")
        print("  1. Download ImageNet dataset (10k images)")
        print("  2. Run: python training/pretrain.py")
        print("  3. Run: python training/gan_train.py --generator-checkpoint checkpoints/generator_pretrain_best.pth")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed.")
        print("\nTo fix:")
        if 'torch' not in installed:
            print("  • Install PyTorch: pip install torch torchvision")
        if missing:
            print(f"  • Install missing packages: pip install {' '.join(missing)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
