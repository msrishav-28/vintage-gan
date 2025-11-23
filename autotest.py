#!/usr/bin/env python
"""
Automated Testing System for VintageGAN

Continuously monitors file changes and runs tests automatically.
Can also be run in single-shot mode.

Usage:
    python autotest.py                    # Watch mode (continuous)
    python autotest.py --once             # Run once and exit
    python autotest.py --install-hooks    # Install git hooks

Author: VintageGAN Project
Date: 2024
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Set


def run_tests(test_type: str = 'quick') -> bool:
    """
    Run tests and return success status.
    
    Args:
        test_type: 'quick' or 'full'
    
    Returns:
        True if tests pass
    """
    print(f"\n{'='*60}")
    print(f"Running {test_type} tests at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        cmd = [sys.executable, 'run_tests.py']
        if test_type == 'quick':
            cmd.append('--quick')
        
        result = subprocess.run(cmd, timeout=300)
        
        if result.returncode == 0:
            print(f"\n✅ All {test_type} tests passed!")
            return True
        else:
            print(f"\n❌ Some {test_type} tests failed")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"\n⏱️  Tests timed out")
        return False
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return False


def check_code_quality() -> bool:
    """Run code quality checks."""
    print(f"\n{'='*60}")
    print("Checking code quality...")
    print(f"{'='*60}")
    
    checks_passed = True
    
    # Check with black
    print("\n1. Checking code formatting (black)...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'black', '--check', '.', '--line-length', '100',
             '--exclude', 'venv|build|dist|\\.eggs'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   ✅ Code formatting: OK")
        else:
            print("   ⚠️  Code formatting: needs formatting")
            print("   Run: black . --line-length 100")
            checks_passed = False
    except Exception as e:
        print(f"   ⚠️  Black not installed: {e}")
    
    # Check with flake8
    print("\n2. Checking linting (flake8)...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'flake8', '.', '--count',
             '--max-line-length=100', '--exclude=venv,build,dist,.eggs',
             '--ignore=E203,W503'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   ✅ Linting: OK")
        else:
            print(f"   ⚠️  Linting issues found:")
            print(f"   {result.stdout[:200]}")
            checks_passed = False
    except Exception as e:
        print(f"   ⚠️  Flake8 not installed: {e}")
    
    return checks_passed


def verify_installation() -> bool:
    """Verify installation is correct."""
    print(f"\n{'='*60}")
    print("Verifying installation...")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, 'verify_installation.py'],
            timeout=60
        )
        
        if result.returncode == 0:
            print("\n✅ Installation verification passed")
            return True
        else:
            print("\n❌ Installation verification failed")
            return False
    except Exception as e:
        print(f"\n❌ Error verifying installation: {e}")
        return False


def install_git_hooks():
    """Install git pre-commit hooks."""
    print(f"\n{'='*60}")
    print("Installing Git Hooks")
    print(f"{'='*60}")
    
    hooks_dir = Path('.git/hooks')
    if not hooks_dir.exists():
        print("❌ Not a git repository")
        return False
    
    # Pre-commit hook
    pre_commit_hook = hooks_dir / 'pre-commit'
    hook_content = """#!/bin/sh
# VintageGAN Pre-commit Hook
# Runs quick tests before allowing commit

echo "Running pre-commit checks..."
python verify_installation.py
if [ $? -ne 0 ]; then
    echo "❌ Installation verification failed"
    exit 1
fi

python run_tests.py --quick
if [ $? -ne 0 ]; then
    echo "❌ Tests failed - commit aborted"
    exit 1
fi

echo "✅ Pre-commit checks passed"
exit 0
"""
    
    try:
        pre_commit_hook.write_text(hook_content)
        pre_commit_hook.chmod(0o755)
        print(f"✅ Installed pre-commit hook: {pre_commit_hook}")
    except Exception as e:
        print(f"❌ Failed to install pre-commit hook: {e}")
        return False
    
    # Pre-push hook
    pre_push_hook = hooks_dir / 'pre-push'
    push_hook_content = """#!/bin/sh
# VintageGAN Pre-push Hook
# Runs full tests before allowing push

echo "Running pre-push checks..."
python run_tests.py
if [ $? -ne 0 ]; then
    echo "❌ Full tests failed - push aborted"
    exit 1
fi

echo "✅ Pre-push checks passed"
exit 0
"""
    
    try:
        pre_push_hook.write_text(push_hook_content)
        pre_push_hook.chmod(0o755)
        print(f"✅ Installed pre-push hook: {pre_push_hook}")
    except Exception as e:
        print(f"❌ Failed to install pre-push hook: {e}")
        return False
    
    print("\n✅ Git hooks installed successfully!")
    print("\nHooks will run automatically:")
    print("  • pre-commit: Quick tests before each commit")
    print("  • pre-push: Full tests before each push")
    
    return True


def watch_files(interval: int = 5):
    """
    Watch for file changes and run tests automatically.
    
    Args:
        interval: Check interval in seconds
    """
    print(f"\n{'='*60}")
    print("VINTAGEGAN AUTOMATED TESTING - WATCH MODE")
    print(f"{'='*60}")
    print(f"Monitoring file changes every {interval} seconds...")
    print("Press Ctrl+C to stop\n")
    
    project_root = Path(__file__).parent
    watched_dirs = ['models', 'training', 'defects', 'evaluation', 'inference', 'tests']
    watched_extensions = {'.py'}
    
    # Get initial file timestamps
    file_mtimes = {}
    for dir_name in watched_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            for file_path in dir_path.rglob('*'):
                if file_path.suffix in watched_extensions:
                    file_mtimes[file_path] = file_path.stat().st_mtime
    
    print(f"Watching {len(file_mtimes)} files in {len(watched_dirs)} directories")
    
    last_test_time = 0
    
    try:
        while True:
            time.sleep(interval)
            
            # Check for changes
            changed_files = []
            for file_path, old_mtime in list(file_mtimes.items()):
                if file_path.exists():
                    new_mtime = file_path.stat().st_mtime
                    if new_mtime > old_mtime:
                        changed_files.append(file_path)
                        file_mtimes[file_path] = new_mtime
            
            # Check for new files
            for dir_name in watched_dirs:
                dir_path = project_root / dir_name
                if dir_path.exists():
                    for file_path in dir_path.rglob('*'):
                        if file_path.suffix in watched_extensions:
                            if file_path not in file_mtimes:
                                changed_files.append(file_path)
                                file_mtimes[file_path] = file_path.stat().st_mtime
            
            # Run tests if files changed
            if changed_files and time.time() - last_test_time > 10:
                print(f"\n{'='*60}")
                print(f"Detected changes in {len(changed_files)} file(s):")
                for f in changed_files[:5]:
                    print(f"  • {f.name}")
                if len(changed_files) > 5:
                    print(f"  ... and {len(changed_files) - 5} more")
                
                run_tests('quick')
                last_test_time = time.time()
    
    except KeyboardInterrupt:
        print("\n\n✅ Watch mode stopped")


def main():
    parser = argparse.ArgumentParser(description='Automated testing for VintageGAN')
    parser.add_argument('--once', action='store_true',
                        help='Run tests once and exit (no watch mode)')
    parser.add_argument('--full', action='store_true',
                        help='Run full tests instead of quick tests')
    parser.add_argument('--install-hooks', action='store_true',
                        help='Install git pre-commit and pre-push hooks')
    parser.add_argument('--quality', action='store_true',
                        help='Run code quality checks')
    parser.add_argument('--all', action='store_true',
                        help='Run all checks (verification + tests + quality)')
    parser.add_argument('--watch-interval', type=int, default=5,
                        help='File watch interval in seconds (default: 5)')
    
    args = parser.parse_args()
    
    # Install git hooks if requested
    if args.install_hooks:
        install_git_hooks()
        return 0
    
    # Run comprehensive checks if --all
    if args.all:
        print("="*60)
        print("RUNNING COMPREHENSIVE CHECKS")
        print("="*60)
        
        success = True
        
        # 1. Verify installation
        if not verify_installation():
            success = False
        
        # 2. Code quality
        if not check_code_quality():
            success = False
        
        # 3. Full tests
        test_type = 'full' if args.full else 'quick'
        if not run_tests(test_type):
            success = False
        
        print("\n" + "="*60)
        if success:
            print("✅ ALL CHECKS PASSED!")
        else:
            print("❌ SOME CHECKS FAILED")
        print("="*60)
        
        return 0 if success else 1
    
    # Run once if requested
    if args.once:
        success = True
        
        if args.quality:
            if not check_code_quality():
                success = False
        
        test_type = 'full' if args.full else 'quick'
        if not run_tests(test_type):
            success = False
        
        return 0 if success else 1
    
    # Default: watch mode
    watch_files(interval=args.watch_interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
