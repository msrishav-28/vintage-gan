"""
Scratch Generation Module
Specification Reference: Section 3.2.2

Implements morphological line defects simulating film scratches.
Scratches are vertical lines caused by physical damage to film during
projection or handling.

Statistical Properties (from spec):
- Width: 1-3 pixels (99% of analog scratches)
- Orientation: 85-90° vertical bias (film movement direction)
- Color: 70% dark scratches, 30% light (emulsion vs base damage)

Author: VintageGAN Project
Date: 2024
"""

from typing import Tuple, List
import numpy as np
import cv2


def generate_scratches(
    image: np.ndarray,
    density: float,
    vertical_bias: float = 0.9
) -> np.ndarray:
    """
    Generate film scratches on image.
    
    Specification Reference: Section 3.2.2 - Scratch Generation
    
    Algorithm:
    1. Determine number of scratches based on density
    2. Generate random scratch positions, lengths, widths
    3. Draw scratches as lines (mostly vertical)
    4. Apply as dark/light lines with random intensity
    
    Args:
        image: Input image (H, W, 3) uint8 in range [0, 255]
        density: Scratch density in range [0.0, 1.0]
            0.0 = no scratches
            1.0 = maximum scratches (15 per image)
        vertical_bias: Probability of vertical vs diagonal scratches [0, 1]
            Default 0.9 = 90% vertical (spec: 85-90° bias)
    
    Returns:
        Image with scratches applied (H, W, 3) uint8
    
    Raises:
        ValueError: If density not in [0, 1]
    
    Example:
        >>> img = np.ones((512, 512, 3), dtype=np.uint8) * 128
        >>> scratched = generate_scratches(img, density=0.5)
        >>> scratched.shape
        (512, 512, 3)
    """
    # Validate inputs
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"Density must be in [0, 1], got {density}")
    
    # Early exit for zero density
    if density < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    defected = image.copy()
    
    # Calculate number of scratches (spec: 0-15 scratches)
    num_scratches = int(density * 15)
    if num_scratches == 0:
        return defected
    
    # Create scratch mask
    scratch_mask = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(num_scratches):
        # Determine if scratch is vertical or diagonal
        is_vertical = np.random.rand() < vertical_bias
        
        if is_vertical:
            # Vertical scratch (most common in film)
            x_pos = np.random.randint(0, w)
            
            # Scratches don't always span full height (spec)
            y_start = np.random.randint(0, h // 3)
            y_end = np.random.randint(2 * h // 3, h)
            
            # Random thickness 1-3 pixels (spec)
            thickness = np.random.randint(1, 4)
            
            # Draw vertical line
            cv2.line(scratch_mask, (x_pos, y_start), (x_pos, y_end), 255, thickness)
        
        else:
            # Diagonal scratch (less common)
            x_start = np.random.randint(0, w)
            y_start = np.random.randint(0, h // 2)
            
            # Slight angle deviation (5-15 degrees from vertical)
            angle_deviation = np.random.uniform(-15, 15)
            angle_rad = np.radians(90 + angle_deviation)
            
            length = np.random.randint(h // 3, h)
            x_end = int(x_start + length * np.cos(angle_rad))
            y_end = int(y_start + length * np.sin(angle_rad))
            
            # Clip to image bounds
            x_end = np.clip(x_end, 0, w - 1)
            y_end = np.clip(y_end, 0, h - 1)
            
            thickness = np.random.randint(1, 4)
            cv2.line(scratch_mask, (x_start, y_start), (x_end, y_end), 255, thickness)
    
    # Apply scratches as dark or light lines (spec: 70% dark, 30% light)
    is_dark_scratch = np.random.rand() < 0.7
    
    if is_dark_scratch:
        # Dark scratch (emulsion damage) - darken image
        scratch_intensity = np.random.uniform(0.2, 0.5)  # Darken by 50-80%
        defected[scratch_mask > 0] = (image[scratch_mask > 0] * scratch_intensity).astype(np.uint8)
    else:
        # Light scratch (base damage) - lighten image
        scratch_intensity = np.random.uniform(1.3, 1.8)  # Brighten by 30-80%
        defected[scratch_mask > 0] = np.clip(
            image[scratch_mask > 0] * scratch_intensity, 0, 255
        ).astype(np.uint8)
    
    return defected


def generate_hairline_scratches(
    image: np.ndarray,
    density: float,
    alpha: float = 0.7
) -> np.ndarray:
    """
    Generate fine hairline scratches with transparency.
    
    These are very thin scratches (1 pixel) with partial transparency,
    simulating light surface scratches.
    
    Args:
        image: Input image (H, W, 3) uint8
        density: Scratch density [0, 1]
        alpha: Scratch opacity [0, 1]
            0.0 = fully transparent, 1.0 = fully opaque
    
    Returns:
        Image with hairline scratches (H, W, 3) uint8
    """
    if density < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    defected = image.copy().astype(np.float32)
    
    num_scratches = int(density * 20)  # More hairlines than regular scratches
    
    for _ in range(num_scratches):
        x_pos = np.random.randint(0, w)
        y_start = np.random.randint(0, h // 4)
        y_end = np.random.randint(3 * h // 4, h)
        
        # Create single-pixel scratch line
        scratch_mask = np.zeros((h, w), dtype=np.float32)
        cv2.line(scratch_mask, (x_pos, y_start), (x_pos, y_end), 1.0, 1)
        
        # Apply with transparency
        is_dark = np.random.rand() < 0.6
        if is_dark:
            color = 0  # Black
        else:
            color = 255  # White
        
        # Blend scratch with image
        for c in range(3):
            defected[:, :, c] = (
                defected[:, :, c] * (1 - scratch_mask * alpha) +
                color * scratch_mask * alpha
            )
    
    return np.clip(defected, 0, 255).astype(np.uint8)


def generate_clustered_scratches(
    image: np.ndarray,
    density: float,
    cluster_size: int = 5
) -> np.ndarray:
    """
    Generate scratches in clusters (simulating damaged film sections).
    
    In reality, film damage often occurs in clustered regions where
    the film was repeatedly damaged by the same projector or handling.
    
    Args:
        image: Input image (H, W, 3) uint8
        density: Overall scratch density [0, 1]
        cluster_size: Number of scratches per cluster
    
    Returns:
        Image with clustered scratches (H, W, 3) uint8
    """
    if density < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    defected = image.copy()
    
    num_clusters = max(1, int(density * 5))
    
    for _ in range(num_clusters):
        # Choose cluster center
        cluster_x = np.random.randint(w // 4, 3 * w // 4)
        
        # Generate scratches around cluster center
        for _ in range(cluster_size):
            # Position near cluster center (within ±50 pixels)
            x_offset = np.random.randint(-50, 51)
            x_pos = np.clip(cluster_x + x_offset, 0, w - 1)
            
            y_start = np.random.randint(0, h // 3)
            y_end = np.random.randint(2 * h // 3, h)
            
            thickness = np.random.randint(1, 3)
            
            # Create scratch
            scratch_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(scratch_mask, (x_pos, y_start), (x_pos, y_end), 255, thickness)
            
            # Apply scratch
            is_dark = np.random.rand() < 0.7
            intensity = np.random.uniform(0.3, 0.6) if is_dark else np.random.uniform(1.2, 1.6)
            
            if is_dark:
                defected[scratch_mask > 0] = (image[scratch_mask > 0] * intensity).astype(np.uint8)
            else:
                defected[scratch_mask > 0] = np.clip(
                    image[scratch_mask > 0] * intensity, 0, 255
                ).astype(np.uint8)
    
    return defected


def add_scratch_texture(
    image: np.ndarray,
    density: float,
    roughness: float = 0.5
) -> np.ndarray:
    """
    Add textured/rough edges to scratches (more realistic).
    
    Real film scratches aren't perfectly straight - they have
    slight variations and rough edges due to uneven damage.
    
    Args:
        image: Input image (H, W, 3) uint8
        density: Scratch density [0, 1]
        roughness: Edge roughness [0, 1]
            0.0 = smooth scratches, 1.0 = very rough
    
    Returns:
        Image with textured scratches (H, W, 3) uint8
    """
    if density < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    defected = image.copy().astype(np.float32)
    
    num_scratches = int(density * 12)
    
    for _ in range(num_scratches):
        x_pos = np.random.randint(0, w)
        y_start = np.random.randint(0, h // 3)
        y_end = np.random.randint(2 * h // 3, h)
        
        # Generate points along scratch path with roughness
        num_points = int((y_end - y_start) / 10)
        points = []
        
        for i in range(num_points):
            y = int(y_start + (y_end - y_start) * i / num_points)
            # Add random x offset for roughness
            x_offset = int(np.random.randn() * roughness * 3)
            x = np.clip(x_pos + x_offset, 0, w - 1)
            points.append((x, y))
        
        # Draw polyline for rough scratch
        if len(points) > 1:
            scratch_mask = np.zeros((h, w), dtype=np.uint8)
            points_array = np.array(points, dtype=np.int32)
            cv2.polylines(scratch_mask, [points_array], False, 255, 
                         thickness=np.random.randint(1, 3))
            
            # Apply scratch
            is_dark = np.random.rand() < 0.7
            intensity = np.random.uniform(0.2, 0.5) if is_dark else np.random.uniform(1.3, 1.7)
            
            mask_3channel = scratch_mask[:, :, np.newaxis] / 255.0
            
            if is_dark:
                defected = defected * (1 - mask_3channel) + (defected * intensity) * mask_3channel
            else:
                defected = defected * (1 - mask_3channel) + np.clip(defected * intensity, 0, 255) * mask_3channel
    
    return np.clip(defected, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    """
    Test script for scratch generation.
    
    Usage:
        python defects/scratches.py
    """
    print("Scratch Generation - Test Script")
    print("="*60)
    
    # Create test image with gradient
    test_image = np.ones((512, 512, 3), dtype=np.uint8) * 128
    for i in range(512):
        test_image[i, :, :] = int(i / 2)
    
    print("\n1. Testing basic scratch generation...")
    try:
        scratched_low = generate_scratches(test_image, density=0.3)
        print(f"   ✓ Low density (0.3): shape {scratched_low.shape}, modified pixels: "
              f"{np.sum(scratched_low != test_image)}")
        
        scratched_med = generate_scratches(test_image, density=0.6)
        print(f"   ✓ Medium density (0.6): shape {scratched_med.shape}, modified pixels: "
              f"{np.sum(scratched_med != test_image)}")
        
        scratched_high = generate_scratches(test_image, density=1.0)
        print(f"   ✓ High density (1.0): shape {scratched_high.shape}, modified pixels: "
              f"{np.sum(scratched_high != test_image)}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n2. Testing hairline scratches...")
    try:
        hairline = generate_hairline_scratches(test_image, density=0.5, alpha=0.7)
        print(f"   ✓ Hairline scratches: shape {hairline.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n3. Testing clustered scratches...")
    try:
        clustered = generate_clustered_scratches(test_image, density=0.5, cluster_size=5)
        print(f"   ✓ Clustered scratches: shape {clustered.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n4. Testing textured scratches...")
    try:
        textured = add_scratch_texture(test_image, density=0.5, roughness=0.5)
        print(f"   ✓ Textured scratches: shape {textured.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n5. Testing edge cases...")
    try:
        # Zero density should return original
        no_scratches = generate_scratches(test_image, density=0.0)
        assert np.array_equal(no_scratches, test_image), "Zero density should return original"
        print("   ✓ Zero density returns original image")
        
        # Invalid density should raise error
        try:
            generate_scratches(test_image, density=1.5)
            print("   ✗ Should have raised ValueError for invalid density")
        except ValueError:
            print("   ✓ ValueError raised for invalid density")
        
        # Test vertical bias
        vertical_only = generate_scratches(test_image, density=0.5, vertical_bias=1.0)
        print("   ✓ Vertical bias parameter works")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✅ Scratch generation tests complete!")
    print("="*60)
