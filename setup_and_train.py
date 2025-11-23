#!/usr/bin/env python
"""
VintageGAN: One-Click Setup and Training

Interactive script that handles everything after dataset download.
Guides user through the complete workflow with minimal input.

Usage:
    python setup_and_train.py

Author: VintageGAN Project
Date: 2024
"""

import subprocess
import sys
from pathlib import Path


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(step_num: int, total: int, description: str):
    """Print step information."""
    print(f"\n[Step {step_num}/{total}] {description}")
    print("-" * 60)


def ask_yes_no(question: str, default: bool = True) -> bool:
    """Ask a yes/no question."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{question} [{default_str}]: ").strip().lower()
    
    if not response:
        return default
    return response in ['y', 'yes']


def check_dataset() -> tuple:
    """Check dataset status."""
    train_dir = Path("data/imagenet_subset/train")
    val_dir = Path("data/imagenet_subset/val")
    
    train_exists = train_dir.exists()
    val_exists = val_dir.exists()
    
    if train_exists and val_exists:
        train_images = list(train_dir.glob("**/*.jpg")) + list(train_dir.glob("**/*.png"))
        val_images = list(val_dir.glob("**/*.jpg")) + list(val_dir.glob("**/*.png"))
        return True, len(train_images), len(val_images)
    
    return False, 0, 0


def main():
    print_header("VintageGAN: Automated Setup and Training")
    
    print("This script will guide you through:")
    print("  1. Dataset verification")
    print("  2. Installation verification")
    print("  3. Generator pretraining (~6 hours)")
    print("  4. GAN training (~4 hours)")
    print("  5. Evaluation and results generation")
    print("\nTotal time: ~10 hours (can run unattended)")
    
    if not ask_yes_no("\nReady to begin?"):
        print("\nExiting. Run this script again when ready.")
        return 0
    
    # Step 1: Check dataset
    print_step(1, 6, "Dataset Verification")
    
    dataset_ready, train_count, val_count = check_dataset()
    
    if dataset_ready:
        print(f"✅ Dataset found:")
        print(f"   Training images: {train_count}")
        print(f"   Validation images: {val_count}")
    else:
        print("❌ Dataset not found")
        print("\nPlease download the dataset first:")
        print("  Option 1: Real data (recommended)")
        print("    python training/download_data.py --mode full --num-train 10000 --num-val 1000")
        print("\n  Option 2: Dummy data (for testing)")
        print("    python training/download_data.py --mode dummy")
        
        if ask_yes_no("\nCreate dummy dataset now for testing?"):
            print("\nCreating dummy dataset...")
            subprocess.run([
                sys.executable, "-c",
                "from training.download_data import create_dummy_dataset; "
                "create_dummy_dataset('data/imagenet_subset', 100, 20)"
            ])
            print("✅ Dummy dataset created")
        else:
            print("\nPlease download dataset and run this script again.")
            return 1
    
    # Step 2: Verify installation
    print_step(2, 6, "Installation Verification")
    
    print("Checking installation...")
    result = subprocess.run([sys.executable, "verify_installation.py"])
    
    if result.returncode != 0:
        print("\n❌ Installation verification failed")
        print("Please fix issues and run this script again.")
        return 1
    
    print("✅ Installation verified")
    
    # Step 3: Check for existing checkpoints
    print_step(3, 6, "Checkpoint Status")
    
    pretrain_checkpoint = Path("checkpoints/generator_pretrain_best.pth")
    final_checkpoint = Path("checkpoints/generator_final.pth")
    
    skip_pretrain = False
    skip_gan = False
    
    if pretrain_checkpoint.exists():
        print(f"✅ Pretrain checkpoint found: {pretrain_checkpoint}")
        if ask_yes_no("Skip pretraining and use existing checkpoint?"):
            skip_pretrain = True
    else:
        print("❌ No pretrain checkpoint found")
    
    if final_checkpoint.exists():
        print(f"✅ Final checkpoint found: {final_checkpoint}")
        if ask_yes_no("Skip GAN training and use existing checkpoint?"):
            skip_gan = True
    else:
        print("❌ No final checkpoint found")
    
    # Step 4: Estimate time
    print_step(4, 6, "Time Estimation")
    
    estimated_hours = 0
    if not skip_pretrain:
        estimated_hours += 6
        print("  - Generator pretraining: ~6 hours")
    if not skip_gan:
        estimated_hours += 4
        print("  - GAN training: ~4 hours")
    
    estimated_hours += 1  # Evaluation
    print("  - Evaluation: ~1 hour")
    
    print(f"\n  Total estimated time: ~{estimated_hours} hours")
    
    if estimated_hours > 1:
        print("\n⚠️  This will take a while. The script can run unattended.")
        print("   Progress will be logged to logs/ directory.")
        print("   You can monitor with: tail -f logs/*.log")
    
    if not ask_yes_no(f"\nProceed with ~{estimated_hours} hour training?"):
        print("\nTraining cancelled.")
        return 0
    
    # Step 5: Run automated pipeline
    print_step(5, 6, "Running Automated Training Pipeline")
    
    print("\n🚀 Starting automated pipeline...")
    print("   This will run unattended. You can close this window.")
    print("   Check logs/ directory for progress.")
    
    cmd = [sys.executable, "run_full_pipeline.py"]
    
    if skip_pretrain:
        cmd.append("--skip-pretrain")
    if skip_gan:
        cmd.append("--skip-gan")
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("\n❌ Pipeline failed")
        print("Check logs/ directory for details.")
        return 1
    
    # Step 6: Summary
    print_step(6, 6, "Training Complete!")
    
    print("✅ All training completed successfully!")
    print("\n📁 Generated files:")
    print("   Checkpoints: checkpoints/")
    print("   Results: results/")
    print("   Logs: logs/")
    
    print("\n📊 Results for your paper:")
    print("   - results/quantitative_metrics.json")
    print("   - results/results_table.csv (Excel/Sheets)")
    print("   - results/results_table.tex (LaTeX)")
    print("   - results/sample_results.png (Figure)")
    print("   - results/training_curves.png (Figure)")
    
    print("\n🎉 VintageGAN training complete!")
    print("   Use the generated results in your research paper.")
    
    # Offer to run inference demo
    if ask_yes_no("\nWould you like to test inference on a sample image?"):
        print("\nLaunching interactive demo...")
        subprocess.run([sys.executable, "-m", "jupyter", "notebook", "notebooks/demo.ipynb"])
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("You can resume by running this script again.")
        sys.exit(1)
