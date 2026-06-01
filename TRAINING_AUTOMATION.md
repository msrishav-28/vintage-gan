# Training Workflow

This project no longer claims zero-click full automation or pre-existing
academic results. The training workflow is intentionally explicit so that
portfolio claims can be traced back to reproducible artifacts.

## Local RTX 3050 Workflow

```bash
pip install -r requirements-local-cu118.txt
pip install -e .
python training/pretrain.py --config configs/training_config.yaml --profile local_256
python training/gan_train.py --config configs/training_config.yaml --profile local_256 --generator-checkpoint checkpoints/generator_pretrain_best.pth
```

The local profile uses:

- 256px images,
- batch size 1,
- mixed precision on CUDA,
- gradient accumulation,
- small epoch counts for iteration.

## Cloud Workflow

Use the same scripts with `cloud_384` or `cloud_512`:

```bash
python training/pretrain.py --config configs/training_config.yaml --profile cloud_384
```

## Required Run Artifacts

Before a result is documented, keep:

- checkpoint files,
- `outputs/experiments/*_metadata.json`,
- sample grids,
- metrics JSON,
- dataset source/license notes.

## Smoke Test Expectation

Before long training, run a tiny dataset smoke path and confirm:

- model forward passes work,
- losses run on the chosen device,
- checkpoints save/load,
- inference produces an image.
