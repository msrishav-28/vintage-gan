#!/usr/bin/env python
"""
Test Runner for VintageGAN

Runs all tests and generates a report.

Usage:
    python run_tests.py [--quick] [--module MODULE]

Author: VintageGAN Project
Date: 2024
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_module_tests(module_path: Path) -> bool:
    """
    Run self-tests for a module.
    
    Args:
        module_path: Path to Python module
    
    Returns:
        True if tests pass
    """
    print(f"\n{'='*60}")
    print(f"Testing: {module_path.name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(module_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ {module_path.name} - PASSED")
            return True
        else:
            print(f"❌ {module_path.name} - FAILED")
            print(result.stdout)
            print(result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        print(f"⏱️  {module_path.name} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {module_path.name} - ERROR: {e}")
        return False


def run_pytest() -> bool:
    """Run pytest on tests directory."""
    print(f"\n{'='*60}")
    print("Running pytest on tests/")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '-v'],
            timeout=300
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ pytest failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run VintageGAN tests')
    parser.add_argument('--quick', action='store_true',
                        help='Quick tests only (skip long-running tests)')
    parser.add_argument('--module', type=str, default=None,
                        help='Test specific module only')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VINTAGEGAN TEST SUITE")
    print("="*60)
    
    # Modules with self-tests
    test_modules = [
        'defects/grain.py',
        'defects/scratches.py',
        'defects/dust.py',
        'defects/vignette.py',
        'defects/color_shift.py',
        'defects/blur.py',
        'defects/combined.py',
    ]
    
    # Add model tests if not quick mode
    if not args.quick:
        test_modules.extend([
            'models/attention.py',
            'models/generator.py',
            'models/discriminator.py',
            'models/detectors.py',
            'training/losses.py',
        ])
    
    # Filter if specific module requested
    if args.module:
        test_modules = [m for m in test_modules if args.module in m]
    
    # Run module self-tests
    passed = 0
    failed = 0
    
    project_root = Path(__file__).parent
    
    for module_path in test_modules:
        full_path = project_root / module_path
        
        if not full_path.exists():
            print(f"⚠️  {module_path} - NOT FOUND")
            continue
        
        if run_module_tests(full_path):
            passed += 1
        else:
            failed += 1
    
    # Run pytest if available and not quick mode
    if not args.quick:
        print(f"\n{'='*60}")
        print("Running integration tests...")
        print(f"{'='*60}")
        
        try:
            import pytest
            if run_pytest():
                print("✅ Integration tests - PASSED")
                passed += 1
            else:
                print("❌ Integration tests - FAILED")
                failed += 1
        except ImportError:
            print("⚠️  pytest not installed, skipping integration tests")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
