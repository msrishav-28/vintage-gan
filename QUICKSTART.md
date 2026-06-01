# Quickstart

This quickstart is for the honest academic prototype path: local 256px training
first, then optional cloud scaling.

## 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-local-cu118.txt
pip install -e .
```

## 2. Prepare Data

Use a legally allowed clean-image dataset:

```text
data/public_images/
  train/
  val/
```

Record the dataset source and license in `configs/training_config.yaml` before
publishing any result.

## 3. Smoke Check

```bash
pytest tests -v
```

## 4. Train Locally

```bash
python training/pretrain.py --config configs/training_config.yaml --profile local_256
python training/gan_train.py --config configs/training_config.yaml --profile local_256 --generator-checkpoint checkpoints/generator_pretrain_best.pth
```

## 5. Run Inference

```bash
python inference/apply_filter.py input.jpg outputs/warm_film.jpg --checkpoint checkpoints/generator_final.pth --preset warm_film --metadata-output outputs/warm_film.json
```

## Notes

- Local profile targets RTX 3050 4GB class hardware.
- Use `cloud_384` or `cloud_512` only when larger GPU memory is available.
- Do not add metrics to the README unless they were generated from a real run.
