"""
Combined Defect Generator
Specification Reference: Section 3.3 - Training Pair Generation

This module combines all individual defect types and applies them in the
correct order to generate realistic vintage film effects.

Order of application (IMPORTANT - specified in Section 3.3):
1. Film grain
2. Scratches
3. Dust
4. Vignetting
5. Color shift
6. Blur

Author: VintageGAN Project
Date: 2024
"""

from typing import Dict, Tuple
import numpy as np

from .grain import generate_film_grain, generate_colored_grain
from .scratches import generate_scratches
from .dust import generate_dust
from .vignette import generate_vignette
from .color_shift import generate_color_shift
from .blur import generate_blur
from .presets import create_preset_conditions


def apply_vintage_defects(
    image: np.ndarray, condition_vector: np.ndarray, use_variants: bool = False
) -> np.ndarray:
    """
    Apply all vintage film defects to image based on condition vector.

    Specification Reference: Section 3.3 - create_training_pair()

    This is the main function used by VintageGANDataset to generate
    training pairs. It applies defects in the exact order specified.

    Args:
        image: Input image (H, W, 3) uint8 in range [0, 255]
        condition_vector: 6D numpy array with values in [0, 1]:
            [grain_intensity, scratch_density, dust_count,
             vignette_strength, color_shift, blur_amount]
        use_variants: If True, use additional defect variations

    Returns:
        Defected image (H, W, 3) uint8

    Example:
        >>> img = np.ones((512, 512, 3), dtype=np.uint8) * 128
        >>> conditions = np.array([0.7, 0.3, 0.2, 0.5, 0.4, 0.1])
        >>> defected = apply_vintage_defects(img, conditions)
        >>> defected.shape
        (512, 512, 3)
    """
    # Validate condition vector
    if condition_vector.shape != (6,):
        raise ValueError(
            f"Condition vector must be shape (6,), got {condition_vector.shape}"
        )

    if not np.all((condition_vector >= 0) & (condition_vector <= 1)):
        raise ValueError("All condition values must be in [0, 1]")

    # Extract individual condition values
    grain_intensity = condition_vector[0]
    scratch_density = condition_vector[1]
    dust_count = condition_vector[2]
    vignette_strength = condition_vector[3]
    color_shift = condition_vector[4]
    blur_amount = condition_vector[5]

    # Start with original image
    defected = image.copy()

    # Apply defects in order (CRITICAL - per specification)
    # Order matters because each effect builds on the previous

    # 1. Film Grain
    if grain_intensity > 0.01:
        if use_variants and np.random.rand() < 0.3:
            # 30% chance of colored grain
            defected = generate_colored_grain(defected, grain_intensity)
        else:
            defected = generate_film_grain(defected, grain_intensity)

    # 2. Scratches
    if scratch_density > 0.01:
        defected = generate_scratches(defected, scratch_density)

    # 3. Dust
    if dust_count > 0.01:
        defected = generate_dust(defected, dust_count)

    # 4. Vignetting
    if vignette_strength > 0.01:
        defected = generate_vignette(defected, vignette_strength)

    # 5. Color Shift
    if color_shift > 0.01:
        defected = generate_color_shift(defected, color_shift)

    # 6. Blur (applied last)
    if blur_amount > 0.01:
        defected = generate_blur(defected, blur_amount)

    return defected


def generate_random_conditions() -> np.ndarray:
    """
    Generate random condition vector.

    Specification Reference: Section 3.3 - Training pair generation

    Returns:
        6D numpy array with random values in [0, 1]
    """
    return np.random.uniform(0.0, 1.0, size=6).astype(np.float32)


def generate_progressive_conditions(
    num_steps: int = 10, defect_index: int = 0
) -> np.ndarray:
    """
    Generate conditions for progressive visualization.

    Useful for creating interpolation sequences showing gradual
    increase in defect intensity.

    Args:
        num_steps: Number of steps in progression
        defect_index: Which defect to vary (0-5)
            0 = grain, 1 = scratch, 2 = dust, 3 = vignette,
            4 = color_shift, 5 = blur

    Returns:
        Array of shape (num_steps, 6) with progressive conditions

    Example:
        >>> # Generate 10 steps of increasing grain
        >>> conditions = generate_progressive_conditions(10, defect_index=0)
        >>> conditions.shape
        (10, 6)
        >>> conditions[:, 0]  # Grain increases from 0 to 1
        array([0. , 0.11, 0.22, 0.33, 0.44, 0.56, 0.67, 0.78, 0.89, 1. ])
    """
    conditions = np.zeros((num_steps, 6), dtype=np.float32)

    # Vary the specified defect from 0 to 1
    conditions[:, defect_index] = np.linspace(0, 1, num_steps)

    return conditions


def batch_apply_defects(
    images: np.ndarray, condition_vectors: np.ndarray
) -> np.ndarray:
    """
    Apply defects to batch of images.

    Args:
        images: Batch of images (B, H, W, 3) uint8
        condition_vectors: Batch of conditions (B, 6) float32

    Returns:
        Batch of defected images (B, H, W, 3) uint8
    """
    batch_size = images.shape[0]
    defected_batch = np.zeros_like(images)

    for i in range(batch_size):
        defected_batch[i] = apply_vintage_defects(images[i], condition_vectors[i])

    return defected_batch


def visualize_defect_progression(
    image: np.ndarray, defect_names: list = None
) -> Dict[str, np.ndarray]:
    """
    Create visualization showing each defect applied separately.

    Useful for understanding what each defect does.

    Args:
        image: Input image (H, W, 3) uint8
        defect_names: List of defects to visualize, or None for all

    Returns:
        Dictionary mapping defect names to defected images
    """
    if defect_names is None:
        defect_names = ["grain", "scratch", "dust", "vignette", "color_shift", "blur"]

    results = {"original": image.copy()}

    # Create condition vector with medium intensity for each defect
    defect_intensity = 0.6

    defect_map = {
        "grain": 0,
        "scratch": 1,
        "dust": 2,
        "vignette": 3,
        "color_shift": 4,
        "blur": 5,
    }

    for defect_name in defect_names:
        if defect_name not in defect_map:
            continue

        # Create condition with only this defect active
        conditions = np.zeros(6, dtype=np.float32)
        conditions[defect_map[defect_name]] = defect_intensity

        # Apply defect
        defected = apply_vintage_defects(image, conditions)
        results[defect_name] = defected

    # Also create combined version
    conditions = np.full(6, defect_intensity, dtype=np.float32)
    results["all_combined"] = apply_vintage_defects(image, conditions)

    return results


if __name__ == "__main__":
    """
    Test script for combined defect generation.

    Usage:
        python defects/combined.py
    """
    print("Combined Defect Generator - Test Script")
    print("=" * 60)

    # Create test image
    test_image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)

    print("\n1. Testing combined defect application...")
    try:
        # Test with random conditions
        conditions = generate_random_conditions()
        print(f"   Random conditions: {conditions}")

        defected = apply_vintage_defects(test_image, conditions)
        print(f"   ✓ Defected image shape: {defected.shape}")
        print(f"   ✓ Value range: [{defected.min()}, {defected.max()}]")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n2. Testing preset conditions...")
    try:
        presets = ["light", "medium", "heavy", "grain_only", "faded", "scratched"]
        for preset in presets:
            cond = create_preset_conditions(preset)
            defected = apply_vintage_defects(test_image, cond)
            print(f"   ✓ Preset '{preset}': {cond}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n3. Testing progressive conditions...")
    try:
        prog_conditions = generate_progressive_conditions(num_steps=5, defect_index=0)
        print(f"   ✓ Progressive conditions shape: {prog_conditions.shape}")
        print(f"   ✓ Grain progression: {prog_conditions[:, 0]}")

        # Apply to image
        for i, cond in enumerate(prog_conditions):
            defected = apply_vintage_defects(test_image, cond)
            print(f"      Step {i}: grain={cond[0]:.2f}, shape={defected.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n4. Testing batch processing...")
    try:
        batch_images = np.stack([test_image] * 4, axis=0)
        batch_conditions = np.array([generate_random_conditions() for _ in range(4)])

        batch_defected = batch_apply_defects(batch_images, batch_conditions)
        print(f"   ✓ Batch defected shape: {batch_defected.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n5. Testing defect progression visualization...")
    try:
        results = visualize_defect_progression(test_image)
        print(f"   ✓ Generated {len(results)} visualization images")
        for name in results.keys():
            print(f"      - {name}: shape {results[name].shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n6. Testing edge cases...")
    try:
        # Zero conditions should return near-original
        zero_cond = np.zeros(6, dtype=np.float32)
        no_defects = apply_vintage_defects(test_image, zero_cond)
        diff = np.sum(np.abs(no_defects.astype(int) - test_image.astype(int)))
        print(f"   ✓ Zero conditions: diff={diff} pixels")

        # All max conditions
        max_cond = np.ones(6, dtype=np.float32)
        max_defects = apply_vintage_defects(test_image, max_cond)
        print(f"   ✓ Max conditions: shape {max_defects.shape}")

        # Invalid condition vector should raise error
        try:
            bad_cond = np.array([1.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            apply_vintage_defects(test_image, bad_cond)
            print("   ✗ Should have raised ValueError for invalid conditions")
        except ValueError:
            print("   ✓ ValueError raised for invalid conditions")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n✅ Combined defect generator tests complete!")
    print("\n" + "=" * 60)
    print("All 6 defect types successfully integrated:")
    print("  1. Film Grain")
    print("  2. Scratches")
    print("  3. Dust Particles")
    print("  4. Vignetting")
    print("  5. Color Shift")
    print("  6. Blur")
    print("=" * 60)
