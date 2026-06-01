# Project Status

VintageGAN is currently an **academic research prototype**.

## Current State

- The target user experience is preset/slider-based vintage filtering.
- The technical goal is controllable conditional image-to-image synthesis.
- The default training profile is `local_256` for RTX 3050 4GB class GPUs.
- Cloud profiles are available for 384px and 512px experimentation.

## Ready

- Procedural target generation for six vintage defect controls.
- Conditional generator/discriminator scaffolding.
- Explicit preset vectors in `configs/presets.yaml`.
- Local/cloud config profiles in `configs/training_config.yaml`.
- Inference API/CLI contract with optional metadata JSON.

## Needs Real Validation

- Clean environment installation.
- End-to-end smoke training.
- Real trained checkpoints.
- Real metrics generated from saved experiment artifacts.
- Dataset source/license documentation.
- Detector-backed consistency loss experiments.

## Not Claimed

- Production readiness.
- Publication readiness.
- Academic-grade results.
- Real FID/SSIM/PSNR numbers.
- Bundled trained checkpoints.

This file should be updated only from verified runs, not from intended targets.
