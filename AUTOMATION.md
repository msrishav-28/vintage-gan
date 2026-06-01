# Automation Notes

VintageGAN automation is intentionally conservative. Scripts may help with
training, testing, and evaluation, but the project does not claim zero-click
academic results.

Use automation only after:

- dependencies install cleanly,
- train/val data exists,
- the dataset license is recorded,
- tests pass,
- and a smoke run is successful.

Primary commands:

```bash
pytest tests -v
python training/pretrain.py --config configs/training_config.yaml --profile local_256
python training/gan_train.py --config configs/training_config.yaml --profile local_256 --generator-checkpoint checkpoints/generator_pretrain_best.pth
```

Automation outputs should be treated as valid only when the matching config,
metadata, checkpoint, and metrics artifact are saved.
