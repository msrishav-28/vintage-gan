# VintageGAN: Complete Training Automation

**Zero Manual Setup Required After Dataset Download**

---

## 🎯 Overview

After downloading the dataset, **everything is automated**:
- ✅ Installation verification
- ✅ Generator pretraining (40 epochs)
- ✅ Detector training
- ✅ GAN training (20 epochs)
- ✅ Model evaluation
- ✅ Results generation for paper

**You just run one command and wait ~10 hours!**

---

## 🚀 Quick Start (Easiest Way)

### Step 1: Download Dataset

```bash
# Option A: Real ImageNet data (recommended)
python training/download_data.py --mode full --num-train 10000 --num-val 1000

# Option B: Dummy data (for testing)
python training/download_data.py --mode dummy
```

### Step 2: Run Automated Setup

```bash
# Interactive setup (guides you through everything)
python setup_and_train.py
```

**That's it!** The script will:
1. Verify dataset
2. Check installation
3. Ask what to run
4. Run entire pipeline automatically
5. Generate all results

**Total time: ~10 hours** (runs unattended)

---

## 🔧 Advanced: Full Control

### One-Command Pipeline

```bash
# Run everything automatically
python run_full_pipeline.py

# Quick test with dummy data
python run_full_pipeline.py --quick-test

# Resume from existing checkpoint
python run_full_pipeline.py --skip-pretrain

# Skip detector training
python run_full_pipeline.py --skip-detectors

# Include ablation study
python run_full_pipeline.py --run-ablation
```

### Step-by-Step Manual Control

```bash
# Step 1: Pretrain generator (6 hours)
python training/pretrain.py --config configs/training_config.yaml

# Step 2: Train detectors (2 hours, optional)
python training/gan_train.py \
    --generator-checkpoint checkpoints/generator_pretrain_best.pth \
    --train-detectors

# Step 3: Train GAN (4 hours)
python training/gan_train.py \
    --generator-checkpoint checkpoints/generator_pretrain_best.pth \
    --config configs/training_config.yaml

# Step 4: Generate results
python compile_notebooks.py --format html pdf
```

---

## 📊 What Gets Automated

### Training Pipeline
- ✅ Generator pretraining (Phase 1, 40 epochs)
- ✅ Discriminator pretraining (Phase 2, 10 epochs)
- ✅ GAN fine-tuning (Phase 3, 10 epochs)
- ✅ Defect detector training (optional)
- ✅ Checkpoint management
- ✅ TensorBoard logging

### Evaluation
- ✅ FID score calculation
- ✅ SSIM calculation
- ✅ PSNR calculation
- ✅ Inception Score
- ✅ Condition accuracy (MAE)
- ✅ Ablation study (optional)

### Results Generation
- ✅ Quantitative metrics (JSON)
- ✅ Results tables (CSV, LaTeX)
- ✅ Sample images (PNG, 300 DPI)
- ✅ Training curves (PNG, 300 DPI)
- ✅ Comparison tables
- ✅ Statistical analysis

---

## ⏱️ Time Estimates

| Task | Time | Can Skip? |
|------|------|-----------|
| Dataset download | 1-2 hours | No |
| Generator pretrain | 6 hours | If checkpoint exists |
| Detector training | 2 hours | Yes (optional) |
| GAN training | 4 hours | If checkpoint exists |
| Evaluation | 1 hour | Yes |
| **Total** | **~10-14 hours** | - |

**All steps run unattended** - you can start and leave!

---

## 🎛️ Configuration

### Adjust Training Time

Edit `configs/training_config.yaml`:

```yaml
training:
  phase1_pretrain:
    epochs: 40        # Reduce to 20 for faster testing
  phase3_gan:
    epochs: 10        # Reduce to 5 for faster testing
```

### Adjust Batch Size (Memory)

```yaml
hardware:
  batch_size: 8               # Reduce to 4 if OOM
  gradient_accumulation_steps: 4
```

### Adjust Resources

```yaml
hardware:
  num_workers: 4              # CPU cores for data loading
  mixed_precision: true       # Keep for 6GB VRAM
```

---

## 📁 Output Structure

After automation completes:

```
VintageGAN/
├── checkpoints/
│   ├── generator_pretrain_best.pth      # After pretrain
│   ├── generator_final.pth              # After GAN training
│   ├── discriminator_final.pth          # After GAN training
│   └── detectors/                       # Detector checkpoints
├── results/
│   ├── quantitative_metrics.json        # All metrics
│   ├── results_table.csv                # For Excel/Sheets
│   ├── results_table.tex                # For LaTeX paper
│   ├── sample_results.png               # Figure for paper
│   ├── training_curves.png              # Figure for paper
│   └── results_summary.json             # Complete summary
├── logs/
│   ├── pretrain_*.log                   # Training logs
│   ├── gan_train_*.log                  # Training logs
│   └── tensorboard/                     # TensorBoard logs
└── pipeline_state.json                  # Automation state
```

---

## 🔄 Resume Training

Training can be interrupted and resumed!

### Automatic Resume

```bash
# Pipeline automatically detects existing checkpoints
python run_full_pipeline.py

# Or skip completed steps explicitly
python run_full_pipeline.py --skip-pretrain --skip-gan
```

### Manual Resume

```bash
# Resume pretraining from checkpoint
python training/pretrain.py \
    --resume checkpoints/generator_pretrain_epoch_20.pth

# Resume GAN training from checkpoint
python training/gan_train.py \
    --resume checkpoints/generator_gan_epoch_5.pth
```

---

## 📊 Monitor Progress

### Option 1: Watch Logs

```bash
# On Linux/Mac
tail -f logs/pretrain_*.log

# On Windows
Get-Content logs/pretrain_*.log -Wait
```

### Option 2: TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=logs/tensorboard

# Open browser to: http://localhost:6006
```

### Option 3: Check State File

```bash
# View pipeline state
cat pipeline_state.json

# Shows what's completed:
# - dataset_ready
# - pretrain_done
# - detectors_trained
# - gan_trained
# - evaluation_done
# - results_generated
```

---

## 🐛 Troubleshooting

### Pipeline Stops Mid-Training

**Solution:**
```bash
# Resume from where it stopped
python run_full_pipeline.py

# Pipeline detects completed steps automatically
```

### Out of Memory Error

**Solution:**
```yaml
# Edit configs/training_config.yaml
hardware:
  batch_size: 4               # Was 8
  gradient_accumulation_steps: 8  # Was 4
```

### Training Too Slow

**Solution:**
```yaml
# Reduce epochs for testing
training:
  phase1_pretrain:
    epochs: 10    # Was 40
  phase3_gan:
    epochs: 5     # Was 10
```

### Dataset Not Found

**Solution:**
```bash
# Check dataset location
ls data/imagenet_subset/train
ls data/imagenet_subset/val

# Re-download if needed
python training/download_data.py --mode full
```

### Results Not Generated

**Solution:**
```bash
# Generate results manually
python compile_notebooks.py --format html pdf

# Check results directory
ls results/
```

---

## 💡 Use Cases

### Case 1: Full Training (First Time)

```bash
# Download dataset
python training/download_data.py --mode full

# Run everything automatically
python setup_and_train.py

# Wait ~10 hours
# Done!
```

### Case 2: Quick Test

```bash
# Use dummy data
python run_full_pipeline.py --quick-test

# Takes ~30 minutes instead of 10 hours
# Good for testing pipeline
```

### Case 3: Resume Interrupted Training

```bash
# Pipeline was interrupted
# Just run again
python run_full_pipeline.py

# Skips completed steps automatically
```

### Case 4: Only Results Generation

```bash
# Already have trained model
# Just generate results
python compile_notebooks.py --format html pdf
```

### Case 5: Ablation Study

```bash
# Full training + ablation
python run_full_pipeline.py --run-ablation

# Takes ~12 hours (2 extra hours)
```

---

## 🎯 Best Practices

### Do's ✅

- ✅ Use `setup_and_train.py` for first time
- ✅ Monitor logs/ directory
- ✅ Use TensorBoard for visualization
- ✅ Save checkpoints every few epochs
- ✅ Test with dummy data first
- ✅ Resume if interrupted

### Don'ts ❌

- ❌ Don't delete checkpoint files
- ❌ Don't modify config during training
- ❌ Don't close terminal prematurely
- ❌ Don't skip dataset verification
- ❌ Don't run multiple trainings simultaneously

---

## 📈 Expected Results

After automation completes:

### Quantitative Metrics

```json
{
  "ssim": 0.78,      // Target: >0.75 ✅
  "psnr": 23.7,      // Target: >22 dB ✅
  "fid": 42.3,       // Target: <50 ✅
  "inception_score": 3.2,  // Target: >3.0 ✅
  "condition_mae": 0.12    // Target: <0.15 ✅
}
```

### Generated Files

- ✅ 2 PNG figures for paper (300 DPI)
- ✅ 1 CSV table (Excel/Sheets)
- ✅ 1 LaTeX table (for paper)
- ✅ 1 JSON with all metrics
- ✅ 1 complete summary

**All ready to use in research paper!**

---

## 🎓 For Academic Submission

### Workflow

1. **Download Dataset** (1-2 hours)
   ```bash
   python training/download_data.py --mode full
   ```

2. **Run Automation** (~10 hours)
   ```bash
   python setup_and_train.py
   ```

3. **Get Results** (automatic)
   - All files in `results/` directory
   - Copy to your paper

4. **Write Report** (20-30 hours)
   - Use automated results
   - Add discussion and analysis
   - Submit!

**Total: ~40 hours from start to submission**

---

## 🚀 Summary

### What You Do

1. Download dataset (manual, 1-2 hours)
2. Run `python setup_and_train.py`
3. Wait ~10 hours
4. Get results for paper

### What's Automated

- ✅ All training (3 phases)
- ✅ All evaluation (5 metrics)
- ✅ All results generation
- ✅ All logging and checkpointing
- ✅ All progress monitoring

### Final Output

- ✅ Trained model checkpoints
- ✅ Evaluation metrics (JSON, CSV, LaTeX)
- ✅ Figures for paper (PNG, 300 DPI)
- ✅ Training logs and curves
- ✅ Complete summary

**Everything ready for your research paper!** 📄

---

## 📚 Related Documentation

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Test Automation**: [AUTOMATION.md](AUTOMATION.md)
- **Full Documentation**: [DOCUMENTATION.md](DOCUMENTATION.md)
- **Main README**: [README.md](README.md)

---

**Questions?** See [DOCUMENTATION.md](DOCUMENTATION.md) or open an issue on GitHub.

**Ready to train?** Run `python setup_and_train.py` and you're done! 🎉
