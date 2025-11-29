# VintageGAN Quick Start Guide

Get started with VintageGAN in 5 minutes!

---

## 1. Installation (2 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/VintageGAN.git
cd VintageGAN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with automation tools
pip install -e .
pip install pre-commit black flake8 pytest

# Setup automated testing (optional but recommended)
python autotest.py --install-hooks
# Tests now run automatically on commits and pushes!

# Verify
python verify_installation.py
```

Expected output: ✅ All checks pass

**Bonus - Automated Testing:**
```bash
# Option A: Watch mode (tests run on file changes)
python autotest.py

# Option B: Windows easy commands
automate install      # Install everything
automate hooks        # Setup git hooks
automate watch        # Start watch mode

# Option C: Linux/Mac Makefile
make install          # Install everything
make hooks            # Setup git hooks
make watch            # Start watch mode
```

---

## 2. Test with Dummy Data (1 minute)

```bash
# Create test dataset
python -c "from training.download_data import create_dummy_dataset; create_dummy_dataset('data/imagenet_subset', 10, 5)"

# Run quick tests
python run_tests.py --quick
```

Expected output: ✅ All tests pass

---

## 3. Try the Demo (2 minutes)

```bash
# Install Jupyter
pip install jupyter ipywidgets

# Launch interactive demo
jupyter notebook notebooks/demo.ipynb
```

Use the sliders to control defect parameters in real-time!

---

## 4. Train Your Model (FULLY AUTOMATED!)

### Option A: Full Automation (Recommended)

```bash
# Step 1: Download dataset
python training/download_data.py --mode full --num-train 10000 --num-val 1000

# Step 2: Run full automated pipeline
python run_full_pipeline.py

# Everything runs automatically:
# - Pretraining (6 hours)
# - GAN training (4 hours)
# - Evaluation (1 hour)
# - Results generation (automatic)
```

### Option B: Quick Test (30 minutes)

```bash
# Use dummy data for quick testing
python run_full_pipeline.py --quick-test

# Tests entire pipeline in ~30 minutes
```

### Option C: Manual Step-by-Step

```bash
# If you prefer manual control
# Step 1: Get training data
python training/download_data.py --mode full --num-train 10000 --num-val 1000

# Step 2: Train
python training/pretrain.py --config configs/training_config.yaml
python training/gan_train.py --generator-checkpoint checkpoints/generator_pretrain_best.pth
```

---

## 5. Apply Vintage Effects (Inference)

After training completes:

```bash
# Apply to your image
vintagegan-filter input.jpg output.jpg \
    --checkpoint checkpoints/generator_final.pth \
    --preset medium

# Or use Python API
python -c "
from inference import VintageFilter
filter = VintageFilter('checkpoints/generator_final.pth')
output = filter.apply('input.jpg', conditions='medium')
output.save('output.jpg')
"
```

---

## 6. Generate Results for Paper

```bash
# Run the full pipeline which includes results generation
python run_full_pipeline.py

# Or run evaluation notebook manually
jupyter notebook notebooks/results_analysis.ipynb

# Results saved to: results/
# - Metrics (JSON, CSV)
# - Figures (PNG at 300 DPI)
# - Training curves
```

---

## Quick Examples

### Python API

```python
from inference import VintageFilter

# Load model
filter = VintageFilter('checkpoints/generator_final.pth')

# Apply effect
output = filter.apply('input.jpg', conditions='medium')
output.save('output.jpg')

# Custom conditions
output = filter.apply('input.jpg', conditions={
    'grain': 0.7,
    'vignette': 0.5,
    'color_shift': 0.6
})
```

### Batch Processing

```bash
# Process entire folder
vintagegan-batch photos/ vintage_photos/ \
    --checkpoint checkpoints/generator_final.pth \
    --preset heavy
```

### Available Presets

- `light` - Subtle vintage look
- `medium` - Balanced aging (default)
- `heavy` - Strong aged appearance
- `grain_only` - Only film grain
- `faded` - Color fading focus
- `scratched` - Emphasis on scratches

---

## Common Issues

### CUDA Out of Memory

```bash
# Reduce batch size in configs/training_config.yaml
# Change: batch_size: 8 → batch_size: 4
```

### ImportError

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Model Not Found

```bash
# Train the model first OR
# Download pre-trained weights (when available)
```

---

## Next Steps

1. **Read full documentation**: See [DOCUMENTATION.md](DOCUMENTATION.md)
2. **Customize training**: Edit `configs/training_config.yaml`
3. **Run ablation study**: `vintagegan-ablation`
4. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Getting Help

- **Documentation**: [DOCUMENTATION.md](DOCUMENTATION.md)
- **Issues**: https://github.com/yourusername/VintageGAN/issues
- **Examples**: Check `notebooks/` directory

---

**Ready to create vintage magic!** 🎞️✨

For detailed information, see the main [README.md](README.md).
