# Contributing to VintageGAN

Thank you for your interest in contributing to VintageGAN! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of professional conduct. By participating, you are expected to uphold this standard.

## How to Contribute

### Reporting Bugs

Before submitting a bug report:
- Check existing issues to avoid duplicates
- Verify the issue with the latest version
- Collect relevant information (OS, Python version, PyTorch version, error logs)

**Bug Report Format:**
```markdown
**Description:** Brief description of the bug

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.9.0]
- PyTorch: [e.g., 2.0.1]
- CUDA: [e.g., 11.8]

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. ...

**Expected Behavior:** What should happen

**Actual Behavior:** What actually happens

**Error Logs:** [Paste error messages]
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Use a clear and descriptive title
- Provide detailed description of the enhancement
- Explain why it would be useful
- Include code examples if applicable

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
   - Follow code style guidelines (below)
   - Add tests for new functionality
   - Update documentation
4. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

## Code Style Guidelines

### Python Code

- **PEP 8 Compliance:** Follow PEP 8 style guide
- **Line Length:** Maximum 100 characters
- **Formatting:** Use `black` formatter
  ```bash
  black . --line-length 100
  ```

### Type Hints

All functions must have type hints:

```python
def process_image(
    image: np.ndarray,
    conditions: np.ndarray,
    intensity: float = 0.5
) -> np.ndarray:
    """Process image with vintage effects."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def apply_defects(image: np.ndarray, conditions: np.ndarray) -> np.ndarray:
    """
    Apply vintage defects to an image.
    
    Args:
        image: Input image as numpy array (H, W, 3)
        conditions: Defect intensity vector (6,) in range [0, 1]
    
    Returns:
        Defected image as numpy array (H, W, 3)
    
    Raises:
        ValueError: If conditions are out of range
    
    Example:
        >>> img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        >>> cond = np.array([0.5, 0.3, 0.2, 0.4, 0.5, 0.1])
        >>> result = apply_defects(img, cond)
    """
    ...
```

### Import Order

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from models import Generator
from defects import apply_vintage_defects
```

## Testing

### Running Tests

```bash
# Run all tests
python run_tests.py

# Quick tests only
python run_tests.py --quick

# Specific module
python run_tests.py --module generator
```

### Writing Tests

- Add tests for all new functionality
- Maintain test coverage above 80%
- Use descriptive test names
- Include edge cases

```python
def test_defect_intensity_bounds():
    """Test that defect functions handle boundary conditions."""
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Test zero intensity
    result_zero = generate_film_grain(image, 0.0)
    assert np.array_equal(result_zero, image)
    
    # Test maximum intensity
    result_max = generate_film_grain(image, 1.0)
    assert not np.array_equal(result_max, image)
```

## Documentation

- Update README.md for user-facing changes
- Update DOCUMENTATION.md for API changes
- Add inline comments for complex logic
- Update docstrings when changing function signatures

## Commit Messages

Use clear, descriptive commit messages:

```
# Good
Add motion blur variant to blur defect module
Fix memory leak in discriminator training loop
Update documentation for new API endpoints

# Bad
Fixed stuff
Update
Changes
```

### Commit Message Format

```
<type>: <subject>

<body (optional)>

<footer (optional)>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Project Structure

When adding new files, follow the existing structure:

```
VintageGAN/
├── models/          # Neural network architectures
├── training/        # Training scripts and utilities
├── defects/         # Defect generation algorithms
├── evaluation/      # Metrics and evaluation tools
├── inference/       # Inference and deployment
├── tests/           # Unit and integration tests
├── notebooks/       # Jupyter notebooks
└── configs/         # Configuration files
```

## Performance Considerations

- Optimize for both CPU and GPU execution
- Profile code for bottlenecks before optimizing
- Document any hardware-specific optimizations
- Maintain backward compatibility when possible

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

---

**Thank you for contributing to VintageGAN!** 🎉
