---
license: mit
library_name: pytorch
pipeline_tag: image-to-image
tags:
  - computer-vision
  - image-to-image
  - gan
  - vintage
  - pytorch
  - academic-project
---

# VintageGAN: Controllable Vintage Image Synthesis

VintageGAN is an academic portfolio project for learning controllable vintage
image degradation. The user-facing experience is intentionally similar to a
photo editor: choose a preset or tune six sliders for grain, scratches, dust,
vignette, color shift, and blur. Under the hood, the project trains a
conditional image-to-image model to imitate procedural vintage targets.

This repository is a research prototype. It does not claim production quality,
published metrics, or trained checkpoints until those artifacts are generated
from reproducible runs.

## Current Goal

- Run locally first on an RTX 3050 4GB class GPU at 256x256.
- Keep the same code path extendable to 384x384 and 512x512 cloud GPU runs.
- Report only metrics produced by saved experiment artifacts.
- Keep presets and manual slider values explicit and reproducible.

## What Is Implemented

- Conditional U-Net style generator with a dynamic bottleneck condition projection.
- Conditional PatchGAN discriminator with a dynamic patch grid.
- Procedural vintage target generator for six independent defect dimensions.
- Explicit preset vectors in `configs/presets.yaml`.
- Local/cloud training profiles in `configs/training_config.yaml`.
- Mixed precision and gradient accumulation hooks for memory-constrained GPUs.
- Inference CLI/API that accepts presets or manual slider values and can write metadata JSON.
- Dataset loading with explicit train/val split folders to avoid leakage.

## What Is Not Yet Claimed

- No real FID/SSIM/PSNR numbers are claimed in this README.
- No “production ready” status is claimed.
- No trained checkpoint is bundled.
- ImageNet/FilmSet use is not assumed unless access and license are verified.

## Installation

Use Python 3.10 or 3.11. For the local RTX 3050 profile:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-local-cu118.txt
pip install -e .
```

For CUDA 12.1 cloud images:

```bash
pip install -r requirements-cloud-cu121.txt
pip install -e .
```

## Dataset Layout

Use a legally allowed clean-image dataset and split it explicitly:

```text
data/public_images/
  train/
    image_000001.jpg
  val/
    image_000001.jpg
```

Before reporting results, record the dataset source and license in
`configs/training_config.yaml` under `dataset.dataset_name` and
`dataset.dataset_license`.

## Training Profiles

The default profile is `local_256`, intended for a 4GB RTX 3050:

```bash
python training/pretrain.py --config configs/training_config.yaml --profile local_256
python training/gan_train.py ^
  --config configs/training_config.yaml ^
  --profile local_256 ^
  --generator-checkpoint checkpoints/generator_pretrain_best.pth
```

Cloud profiles use the same scripts:

```bash
python training/pretrain.py --config configs/training_config.yaml --profile cloud_384
python training/pretrain.py --config configs/training_config.yaml --profile cloud_512
```

Each training run writes a metadata manifest under `outputs/experiments/`.

## Inference

Preset mode:

```bash
python inference/apply_filter.py input.jpg outputs/input_vintage.jpg ^
  --checkpoint checkpoints/generator_final.pth ^
  --preset warm_film ^
  --image-size 256 ^
  --metadata-output outputs/input_vintage.json
```

Manual slider mode:

```bash
python inference/apply_filter.py input.jpg outputs/custom.jpg ^
  --checkpoint checkpoints/generator_final.pth ^
  --grain 0.6 --scratch 0.2 --dust 0.4 --vignette 0.5 --color-shift 0.7 --blur 0.1
```

Python API:

```python
from inference import VintageFilter

vintage = VintageFilter(checkpoint="checkpoints/generator_final.pth", image_size=256)
output = vintage.apply("input.jpg", conditions="dusty_archive")
output.save("outputs/dusty_archive.jpg")
```

## Presets

Preset vectors live in `configs/presets.yaml`.

The canonical condition order is:

```text
grain, scratch, dust, vignette, color_shift, blur
```

Current academic/demo presets:

- `soft_fade`
- `warm_film`
- `dusty_archive`
- `scratched_negative`
- `heavy_vintage`

Backward-compatible aliases such as `light`, `medium`, and `heavy` are also
kept for older scripts.

## Evaluation Plan

Metrics should be generated only after training:

- SSIM/PSNR against synthetic procedural targets, with clear caveats.
- FID only when a real, licensed vintage reference set is available.
- Condition-control accuracy only when defect detectors are trained and validated.
- Ablations for no adversarial loss, no perceptual loss, and no consistency loss.

Recommended portfolio evidence:

- Config file.
- Git commit.
- Dataset source/license.
- Hardware notes.
- Training duration.
- Saved sample grids.
- Metrics JSON produced by evaluation scripts.

## Tests

```bash
pytest tests -v
```

Minimum checks expected before publishing results:

- `import models`, `import training`, and `import inference` pass.
- Generator and discriminator shape tests pass at 256x256.
- Dataset determinism and train/val split checks pass.
- One tiny smoke training run completes.
- A checkpoint can be loaded for inference.

## Repository Structure

```text
configs/      training profiles and preset vectors
defects/      procedural target generation
models/       generator, discriminator, attention, detector modules
training/     dataloaders, losses, pretraining, GAN training
evaluation/   metrics and ablation scaffolding
inference/    single-image and batch filter application
tests/        unit and integration tests
```

## Citation

This project is not a publication. If you use it in coursework or a portfolio,
cite the repository and include the exact commit hash used for results.

## License

MIT. Dataset licenses are separate and must be checked for the images you train
or evaluate on.
