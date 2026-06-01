#!/usr/bin/env python
"""
VintageGAN pipeline helper.

Runs workflow commands, but results are publishable only when backed by saved
config, metadata, checkpoints, samples, and metrics artifacts.

Usage:
    python run_full_pipeline.py                    # Full pipeline
    python run_full_pipeline.py --skip-pretrain    # Skip if already done
    python run_full_pipeline.py --quick-test       # Use dummy data

Author: VintageGAN Project
Date: 2024
"""

import argparse
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class VintageGANPipeline:
    """Automated pipeline for VintageGAN training and evaluation."""

    def __init__(self, config_path: str = "configs/training_config.yaml"):
        self.config_path = config_path
        self.checkpoints_dir = Path("checkpoints")
        self.results_dir = Path("results")
        self.logs_dir = Path("logs")

        # Create directories
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Pipeline state
        self.state_file = Path("pipeline_state.json")
        self.state = self.load_state()

    def load_state(self) -> Dict:
        """Load pipeline state from file."""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {
            "dataset_ready": False,
            "pretrain_done": False,
            "detectors_trained": False,
            "gan_trained": False,
            "evaluation_done": False,
            "results_generated": False,
        }

    def save_state(self):
        """Save pipeline state to file."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            "INFO": "📋",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️",
            "RUNNING": "🔄",
        }.get(level, "📋")

        print(f"{prefix} [{timestamp}] {message}")

    def run_command(self, cmd: list, description: str, timeout: int = 3600) -> bool:
        """
        Run a command and return success status.

        Args:
            cmd: Command to run as list
            description: Human-readable description
            timeout: Timeout in seconds

        Returns:
            True if successful
        """
        self.log(f"Starting: {description}", "RUNNING")

        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode == 0:
                self.log(f"Completed: {description}", "SUCCESS")
                return True
            else:
                self.log(f"Failed: {description}", "ERROR")
                self.log(f"Error output: {result.stderr[:500]}", "ERROR")
                return False

        except subprocess.TimeoutExpired:
            self.log(f"Timeout: {description}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Exception in {description}: {e}", "ERROR")
            return False

    def check_dataset(self) -> bool:
        """Check if dataset is ready."""
        train_dir = Path("data/imagenet_subset/train")
        val_dir = Path("data/imagenet_subset/val")

        if train_dir.exists() and val_dir.exists():
            train_images = list(train_dir.glob("**/*.jpg")) + list(
                train_dir.glob("**/*.png")
            )
            val_images = list(val_dir.glob("**/*.jpg")) + list(val_dir.glob("**/*.png"))

            if len(train_images) > 0 and len(val_images) > 0:
                self.log(
                    f"Dataset found: {len(train_images)} train, {len(val_images)} val images",
                    "SUCCESS",
                )
                return True

        return False

    def setup_dataset(self, use_dummy: bool = False) -> bool:
        """Setup dataset (download or create dummy)."""
        self.log("Step 1: Dataset Setup", "INFO")

        if self.state["dataset_ready"] and self.check_dataset():
            self.log("Dataset already ready, skipping", "SUCCESS")
            return True

        if use_dummy:
            self.log("Creating dummy dataset for testing...", "INFO")
            success = self.run_command(
                [
                    sys.executable,
                    "-c",
                    "from training.download_data import create_dummy_dataset; "
                    "create_dummy_dataset('data/imagenet_subset', 100, 20)",
                ],
                "Create dummy dataset",
                timeout=300,
            )
        else:
            self.log("Please download ImageNet dataset manually", "WARNING")
            self.log(
                "Run: python training/download_data.py --mode full --num-train 10000 --num-val 1000",
                "INFO",
            )
            return False

        if success and self.check_dataset():
            self.state["dataset_ready"] = True
            self.save_state()
            return True

        return False

    def pretrain_generator(self, skip: bool = False) -> bool:
        """Pretrain generator (Phase 1)."""
        self.log("Step 2: Generator Pretraining (Phase 1)", "INFO")

        if skip or self.state["pretrain_done"]:
            checkpoint = self.checkpoints_dir / "generator_pretrain_best.pth"
            if checkpoint.exists():
                self.log("Pretraining already done, skipping", "SUCCESS")
                return True

        self.log("Starting generator pretraining (40 epochs, ~6 hours)...", "INFO")
        self.log("This will take a while. You can monitor progress in logs/", "INFO")

        success = self.run_command(
            [sys.executable, "training/pretrain.py", "--config", self.config_path],
            "Generator pretraining",
            timeout=36000,  # long-run guardrail
        )

        if success:
            self.state["pretrain_done"] = True
            self.save_state()

        return success

    def train_detectors(self, skip: bool = False) -> bool:
        """Train defect detectors."""
        self.log("Step 3: Training Defect Detectors", "INFO")

        if skip or self.state["detectors_trained"]:
            self.log("Detectors already trained, skipping", "SUCCESS")
            return True

        # Check if pretrained generator exists
        pretrain_checkpoint = self.checkpoints_dir / "generator_pretrain_best.pth"
        if not pretrain_checkpoint.exists():
            self.log("Pretrained generator not found, cannot train detectors", "ERROR")
            return False

        self.log("Training defect detectors (~2 hours)...", "INFO")

        success = self.run_command(
            [
                sys.executable,
                "training/gan_train.py",
                "--generator-checkpoint",
                str(pretrain_checkpoint),
                "--train-detectors",
                "--config",
                self.config_path,
            ],
            "Detector training",
            timeout=14400,  # 4 hours max
        )

        if success:
            self.state["detectors_trained"] = True
            self.save_state()

        return success

    def train_gan(self, skip: bool = False) -> bool:
        """Train GAN (Phases 2 & 3)."""
        self.log("Step 4: GAN Training (Phases 2 & 3)", "INFO")

        if skip or self.state["gan_trained"]:
            checkpoint = self.checkpoints_dir / "generator_final.pth"
            if checkpoint.exists():
                self.log("GAN training already done, skipping", "SUCCESS")
                return True

        # Check if pretrained generator exists
        pretrain_checkpoint = self.checkpoints_dir / "generator_pretrain_best.pth"
        if not pretrain_checkpoint.exists():
            self.log("Pretrained generator not found, cannot train GAN", "ERROR")
            return False

        self.log("Starting GAN training (20 epochs, ~4 hours)...", "INFO")

        success = self.run_command(
            [
                sys.executable,
                "training/gan_train.py",
                "--generator-checkpoint",
                str(pretrain_checkpoint),
                "--config",
                self.config_path,
            ],
            "GAN training",
            timeout=21600,  # 6 hours max
        )

        if success:
            self.state["gan_trained"] = True
            self.save_state()

        return success

    def evaluate_model(self, skip: bool = False) -> bool:
        """Evaluate trained model."""
        self.log("Step 5: Model Evaluation", "INFO")

        if skip or self.state["evaluation_done"]:
            metrics_file = self.results_dir / "quantitative_metrics.json"
            if metrics_file.exists():
                self.log("Evaluation already done, skipping", "SUCCESS")
                return True

        # Check if final model exists
        final_checkpoint = self.checkpoints_dir / "generator_final.pth"
        if not final_checkpoint.exists():
            self.log("Final model not found, cannot evaluate", "ERROR")
            return False

        self.log("Running model evaluation...", "INFO")

        # Run evaluation through notebook compilation
        success = self.run_command(
            [sys.executable, "compile_notebooks.py", "--format", "html"],
            "Model evaluation",
            timeout=3600,  # 1 hour max
        )

        if success:
            self.state["evaluation_done"] = True
            self.save_state()

        return success

    def generate_results(self, skip: bool = False) -> bool:
        """Generate all results for paper."""
        self.log("Step 6: Generating Results for Paper", "INFO")

        if skip or self.state["results_generated"]:
            self.log("Results already generated, skipping", "SUCCESS")
            return True

        self.log("Compiling results notebooks...", "INFO")

        success = self.run_command(
            [sys.executable, "compile_notebooks.py", "--format", "html", "pdf"],
            "Results generation",
            timeout=1800,  # 30 minutes max
        )

        if success:
            self.state["results_generated"] = True
            self.save_state()

            self.log("Results generated in results/ directory:", "SUCCESS")
            self.log("  - quantitative_metrics.json", "INFO")
            self.log("  - results_table.csv", "INFO")
            self.log("  - results_table.tex", "INFO")
            self.log("  - sample_results.png", "INFO")
            self.log("  - training_curves.png", "INFO")

        return success

    def run_ablation_study(self) -> bool:
        """Run ablation study (optional)."""
        self.log("Step 7 (Optional): Ablation Study", "INFO")

        self.log("Running ablation study...", "INFO")

        success = self.run_command(
            [sys.executable, "evaluation/ablation.py", "--config", self.config_path],
            "Ablation study",
            timeout=7200,  # 2 hours max
        )

        return success

    def run_full_pipeline(
        self,
        use_dummy: bool = False,
        skip_pretrain: bool = False,
        skip_detectors: bool = False,
        skip_gan: bool = False,
        skip_evaluation: bool = False,
        skip_results: bool = False,
        run_ablation: bool = False,
    ) -> bool:
        """
        Run the complete pipeline.

        Args:
            use_dummy: Use dummy dataset for testing
            skip_pretrain: Skip pretraining if already done
            skip_detectors: Skip detector training
            skip_gan: Skip GAN training
            skip_evaluation: Skip evaluation
            skip_results: Skip results generation
            run_ablation: Run ablation study (optional)

        Returns:
            True if pipeline completed successfully
        """
        start_time = time.time()

        self.log("=" * 60, "INFO")
        self.log("VINTAGEGAN: AUTOMATED TRAINING PIPELINE", "INFO")
        self.log("=" * 60, "INFO")

        # Step 1: Dataset
        if not self.setup_dataset(use_dummy=use_dummy):
            self.log("Dataset setup failed", "ERROR")
            return False

        # Step 2: Pretraining
        if not self.pretrain_generator(skip=skip_pretrain):
            self.log("Generator pretraining failed", "ERROR")
            return False

        # Step 3: Detectors
        if not skip_detectors:
            if not self.train_detectors():
                self.log("Detector training failed (continuing anyway)", "WARNING")

        # Step 4: GAN Training
        if not self.train_gan(skip=skip_gan):
            self.log("GAN training failed", "ERROR")
            return False

        # Step 5: Evaluation
        if not skip_evaluation:
            if not self.evaluate_model():
                self.log("Evaluation failed (continuing anyway)", "WARNING")

        # Step 6: Results
        if not skip_results:
            if not self.generate_results():
                self.log("Results generation failed (continuing anyway)", "WARNING")

        # Step 7: Ablation (optional)
        if run_ablation:
            if not self.run_ablation_study():
                self.log("Ablation study failed (optional, continuing)", "WARNING")

        # Summary
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)

        self.log("=" * 60, "INFO")
        self.log("PIPELINE COMPLETE!", "SUCCESS")
        self.log("=" * 60, "INFO")
        self.log(f"Total time: {hours}h {minutes}m", "INFO")
        self.log(f"Checkpoints: {self.checkpoints_dir}/", "INFO")
        self.log(f"Results: {self.results_dir}/", "INFO")
        self.log(f"Logs: {self.logs_dir}/", "INFO")
        self.log("=" * 60, "INFO")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="VintageGAN Automated Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with real data
  python run_full_pipeline.py
  
  # Quick test with dummy data
  python run_full_pipeline.py --quick-test
  
  # Resume from pretraining checkpoint
  python run_full_pipeline.py --skip-pretrain
  
  # Full pipeline with ablation study
  python run_full_pipeline.py --run-ablation
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Use dummy dataset for quick testing"
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pretraining if checkpoint exists",
    )
    parser.add_argument(
        "--skip-detectors", action="store_true", help="Skip detector training"
    )
    parser.add_argument(
        "--skip-gan", action="store_true", help="Skip GAN training if checkpoint exists"
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true", help="Skip evaluation step"
    )
    parser.add_argument(
        "--skip-results", action="store_true", help="Skip results generation"
    )
    parser.add_argument(
        "--run-ablation", action="store_true", help="Run ablation study (optional)"
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Reset pipeline state and start fresh",
    )

    args = parser.parse_args()

    # Reset state if requested
    if args.reset_state:
        state_file = Path("pipeline_state.json")
        if state_file.exists():
            state_file.unlink()
        print("✅ Pipeline state reset")

    # Create and run pipeline
    pipeline = VintageGANPipeline(config_path=args.config)

    success = pipeline.run_full_pipeline(
        use_dummy=args.quick_test,
        skip_pretrain=args.skip_pretrain,
        skip_detectors=args.skip_detectors,
        skip_gan=args.skip_gan,
        skip_evaluation=args.skip_evaluation,
        skip_results=args.skip_results,
        run_ablation=args.run_ablation,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
