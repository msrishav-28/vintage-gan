# Technical Documentation

## Product Shape

VintageGAN is a controllable vintage image synthesis prototype. It should feel
like a filter tool at inference time, but the academic value comes from proving
that a conditional model can learn the mapping from a clean image and six
control values to a vintage target image.

## Condition Vector

Canonical order:

```text
grain, scratch, dust, vignette, color_shift, blur
```

Each value is normalized to `[0, 1]`. Presets are stored in
`configs/presets.yaml` and are not hidden inside notebooks.

## Architecture

- Generator: custom conditioned U-Net style encoder/decoder.
- Conditioning: spatial replication at input plus MLP projection at bottleneck.
- Discriminator: conditional PatchGAN with dynamic patch-grid output.
- Defects: procedural targets applied in a fixed order.

The current generator is not a torchvision ResNet34 model.

## Training Profiles

- `local_256`: RTX 3050 4GB class local experimentation.
- `cloud_384`: medium cloud GPU experiments.
- `cloud_512`: larger cloud GPU experiments.

All profiles live in `configs/training_config.yaml`.

## Losses

- Pixel reconstruction loss.
- Optional VGG perceptual loss.
- Adversarial loss for GAN phase.
- Optional detector-backed consistency loss.

Consistency loss should only be treated as academically meaningful after
detectors are trained and validated.

## Evaluation

No metric in this project should be hand-entered. Valid metrics must come from
evaluation code and saved artifacts.

Recommended reporting:

- SSIM/PSNR against synthetic procedural targets, with limitations stated.
- FID only against a real, licensed vintage reference set.
- Condition MAE only with validated detectors.
- Ablations for key losses and condition strengths.

## Reproducibility

Each result should include:

- git commit,
- config profile,
- dataset source and license,
- split seed,
- hardware,
- training duration,
- checkpoint path,
- generated samples,
- metrics artifact.
