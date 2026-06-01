# VintageGAN Summary

VintageGAN is being rebuilt into an honest academic portfolio project for
controllable vintage image synthesis.

The project should be understood as:

- A photo-editor-like interface idea: presets and sliders.
- A machine-learning project underneath: a conditional generator learns to
  reproduce procedural vintage targets.
- A local-first training setup: 256px on RTX 3050 4GB, with cloud profiles for
  larger experiments.

## Current Implementation Focus

- Make imports and environment setup reliable.
- Make 256px training feasible.
- Keep presets explicit and reproducible.
- Save experiment metadata for every training run.
- Report only metrics produced by real evaluation scripts.

## Publication Rule

No result table, metric, figure, or “production ready” claim should be added
unless it is backed by:

- the config used,
- the git commit,
- the dataset source/license,
- hardware notes,
- the checkpoint,
- and the generated metrics artifact.
