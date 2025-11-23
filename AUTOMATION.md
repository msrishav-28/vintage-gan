# VintageGAN: Automation Guide

Complete guide to automated testing, CI/CD, and development workflows.

---

## 🚀 Quick Start

### One Command Setup

```bash
# Windows
automate install
automate hooks

# Linux/Mac
make install
make hooks
```

Now tests run automatically on every commit and push!

---

## 📋 Automation Features

### 1. Git Hooks (Automatic on Commit/Push)

**Install once:**
```bash
python autotest.py --install-hooks
```

**What happens:**
- **Pre-commit**: Runs quick tests before allowing commit
- **Pre-push**: Runs full tests before allowing push
- Prevents committing broken code
- No manual testing needed

**Skip if needed:**
```bash
git commit --no-verify    # Skip pre-commit hook
git push --no-verify      # Skip pre-push hook
```

---

### 2. Watch Mode (Automatic on File Changes)

**Start watching:**
```bash
python autotest.py
```

**What happens:**
- Monitors all Python files in project
- Detects changes automatically
- Runs quick tests when files change
- Runs every 5 seconds (configurable)
- Press Ctrl+C to stop

**Custom interval:**
```bash
python autotest.py --watch-interval 10    # Check every 10 seconds
```

---

### 3. GitHub Actions CI/CD

**Automatically runs on:**
- Every push to main/develop
- Every pull request
- Manual workflow trigger

**What it tests:**
- ✅ Python 3.9, 3.10, 3.11
- ✅ Ubuntu and Windows
- ✅ Code formatting (Black)
- ✅ Linting (flake8)
- ✅ Installation verification
- ✅ Quick tests
- ✅ Full test suite
- ✅ Package building
- ✅ Code quality checks

**View status:**
- Check the "Actions" tab on GitHub
- Badge in README shows build status

---

### 4. Pre-commit Framework

**Install pre-commit:**
```bash
pip install pre-commit
pre-commit install
```

**What it checks:**
- Trailing whitespace
- File formatting
- YAML/JSON syntax
- Large files
- Merge conflicts
- Code formatting (Black)
- Linting (flake8)
- Import sorting (isort)

**Run manually:**
```bash
pre-commit run --all-files
```

---

## 🎯 Automation Commands

### Windows (automate.bat)

```bash
automate help           # Show all commands
automate install        # Install package + dev tools
automate hooks          # Install git hooks
automate test           # Run quick tests
automate test-full      # Run all tests
automate verify         # Verify installation
automate watch          # Start watch mode
automate quality        # Check code quality
automate format         # Format code with black
automate lint           # Lint with flake8
automate ci             # Run full CI pipeline locally
automate all            # Run everything
automate clean          # Remove generated files
automate results        # Generate paper results
```

### Linux/Mac (Makefile)

```bash
make help               # Show all commands
make install            # Install package + dev tools
make hooks              # Install git hooks
make test               # Run quick tests
make test-full          # Run all tests
make verify             # Verify installation
make watch              # Start watch mode
make quality            # Check code quality
make format             # Format code with black
make lint               # Lint with flake8
make ci                 # Run full CI pipeline locally
make all                # Run everything
make clean              # Remove generated files
make results            # Generate paper results
```

### Python Script (autotest.py)

```bash
# Watch mode (continuous testing)
python autotest.py

# Run once and exit
python autotest.py --once

# Run full tests (not quick)
python autotest.py --once --full

# Check code quality
python autotest.py --once --quality

# Run all checks
python autotest.py --once --all

# Install git hooks
python autotest.py --install-hooks

# Custom watch interval
python autotest.py --watch-interval 10
```

---

## 🔧 Customization

### Modify Git Hooks

Edit `.git/hooks/pre-commit` or `.git/hooks/pre-push`:

```bash
#!/bin/sh
# Custom pre-commit hook

# Only run quick tests
python run_tests.py --quick

# Or skip verification
# python run_tests.py --quick --skip-verify
```

### Modify Watch Mode

Edit `autotest.py`:

```python
# Line ~280
watched_dirs = ['models', 'training', 'defects', 'evaluation', 'inference', 'tests']

# Add more directories
watched_dirs.append('notebooks')

# Change watch interval (default 5 seconds)
watch_files(interval=10)
```

### Modify CI/CD

Edit `.github/workflows/ci.yml`:

```yaml
# Change Python versions
matrix:
  python-version: ['3.9', '3.10', '3.11', '3.12']

# Change OS
matrix:
  os: [ubuntu-latest, windows-latest, macos-latest]

# Add more test steps
- name: Custom test
  run: |
    python custom_test.py
```

---

## 📊 What Gets Tested

### Installation Verification
- Python version check (>= 3.9)
- All dependencies installed
- Project structure exists
- Config files present
- Module imports work
- Models instantiate correctly

### Code Quality
- **Black**: Code formatting (line length 100)
- **flake8**: Linting (PEP 8 compliance)
- **isort**: Import sorting
- **mypy**: Type checking (optional)
- **pylint**: Static analysis (optional)

### Unit Tests
- Defect synthesis functions
- Model forward passes
- Loss calculations
- Data loading
- Inference API

### Integration Tests
- Full pipeline (data → models → training)
- Defect + dataset integration
- Model memory efficiency
- Condition control

---

## 🎨 Development Workflow

### Recommended Workflow

1. **Initial Setup:**
   ```bash
   automate install
   automate hooks
   ```

2. **Development:**
   - Option A: Start watch mode: `automate watch`
   - Option B: Let git hooks handle testing
   - Tests run automatically as you code

3. **Before Commit:**
   - Pre-commit hook runs automatically
   - Fix any issues
   - Commit succeeds when tests pass

4. **Before Push:**
   - Pre-push hook runs full tests
   - Fix any issues
   - Push succeeds when all tests pass

5. **On GitHub:**
   - CI/CD runs automatically
   - Multiple Python versions tested
   - Multiple OS tested
   - Build artifacts created

---

## 🐛 Troubleshooting

### Hook Not Running

```bash
# Reinstall hooks
python autotest.py --install-hooks

# Check hook exists
ls .git/hooks/pre-commit

# Check hook is executable (Linux/Mac)
chmod +x .git/hooks/pre-commit
```

### Watch Mode Not Detecting Changes

```bash
# Check which files are watched
python autotest.py --watch-interval 1

# Manually save file
# Watch output should show detection
```

### CI/CD Failing

```bash
# Run CI locally first
automate ci      # Windows
make ci          # Linux/Mac

# Fix issues before pushing
```

### Tests Taking Too Long

```bash
# Use quick tests
python run_tests.py --quick

# Or adjust what runs in hooks
# Edit .git/hooks/pre-commit
```

---

## ⚡ Performance Tips

### Speed Up Tests

1. **Use Quick Tests for Development:**
   ```bash
   python run_tests.py --quick    # ~30 seconds
   ```

2. **Save Full Tests for Push:**
   - Quick tests on pre-commit
   - Full tests on pre-push

3. **Parallel Testing:**
   ```bash
   pytest tests/ -n auto    # Run tests in parallel
   ```

4. **Skip Slow Tests During Development:**
   ```bash
   pytest tests/ -m "not slow"
   ```

---

## 📈 Monitoring

### Local Monitoring

```bash
# Watch mode shows:
# - Files changed
# - Tests run
# - Results (pass/fail)
# - Timestamps
```

### GitHub Monitoring

```bash
# Check Actions tab:
# - Build status
# - Test results
# - Code coverage (if enabled)
# - Build artifacts
```

### Pre-commit Monitoring

```bash
# Shows in terminal:
# - Which hooks ran
# - Which passed/failed
# - Error messages
```

---

## 🔐 Best Practices

### Do's ✅

- ✅ Install git hooks on every project clone
- ✅ Use watch mode during active development
- ✅ Run full tests before important commits
- ✅ Keep tests fast (< 1 minute for quick tests)
- ✅ Fix failing tests immediately
- ✅ Check CI status before merging PRs

### Don'ts ❌

- ❌ Don't skip hooks without good reason
- ❌ Don't commit broken tests
- ❌ Don't push without running full tests
- ❌ Don't ignore CI failures
- ❌ Don't disable hooks permanently

---

## 🎯 Examples

### Example 1: First Time Setup

```bash
# Clone repo
git clone https://github.com/yourusername/VintageGAN.git
cd VintageGAN

# Setup automation (one time)
automate install
automate hooks

# Start developing with watch mode
automate watch

# Code and see tests run automatically!
```

### Example 2: Before Important Commit

```bash
# Run everything manually
automate all              # Windows
make all                  # Linux/Mac

# Commit if all passed
git add .
git commit -m "Add feature"

# Push (hooks run automatically)
git push
```

### Example 3: Debugging Failed Tests

```bash
# Run tests manually to see output
python run_tests.py -v

# Run specific test
python run_tests.py --module generator

# Fix issues

# Try again
automate test
```

---

## 📝 Summary

**Automated Testing Options:**

| Method | When It Runs | Speed | Best For |
|--------|-------------|-------|----------|
| Watch Mode | On file save | Fast | Active development |
| Git Hooks | On commit/push | Medium | Quality control |
| GitHub Actions | On push to GitHub | Full | CI/CD pipeline |
| Manual | On command | Variable | Debugging |

**Recommendation:** Use all four!
- Watch mode while coding
- Git hooks for local quality
- GitHub Actions for team/CI
- Manual for debugging

---

## 🚀 Getting Started Now

**Quick setup (2 minutes):**

```bash
# 1. Install
pip install -e .
pip install pre-commit black flake8 pytest

# 2. Setup hooks
python autotest.py --install-hooks

# 3. Start watch mode
python autotest.py
```

**That's it!** Tests now run automatically. 🎉

---

For more details, see:
- [README.md](README.md) - Main documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [DOCUMENTATION.md](DOCUMENTATION.md) - Technical reference
