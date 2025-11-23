"""
Dust Particle Synthesis Module
Specification Reference: Section 3.2.3

Implements random elliptical particles simulating dust and dirt on film.

Author: VintageGAN Project
Date: 2024
"""

from typing import Tuple
import numpy as np
import cv2


def generate_dust(
    image: np.ndarray,
    count: float,
    size_range: Tuple[int, int] = (2, 8)
) -> np.ndarray:
    """
    Generate dust particles on image.
    
    Specification Reference: Section 3.2.3 - Dust Particle Synthesis
    
    Algorithm:
    1. Determine number of particles based on count parameter
    2. Generate random elliptical particles
    3. Apply as dark spots with soft edges (Gaussian blur)
    
    Args:
        image: Input image (H, W, 3) uint8 in range [0, 255]
        count: Dust density in range [0.0, 1.0]
            0.0 = no dust
            1.0 = maximum dust (50 particles)
        size_range: Min and max particle size in pixels
    
    Returns:
        Image with dust applied (H, W, 3) uint8
    
    Raises:
        ValueError: If count not in [0, 1]
    
    Example:
        >>> img = np.ones((512, 512, 3), dtype=np.uint8) * 128
        >>> dusty = generate_dust(img, count=0.5)
        >>> dusty.shape
        (512, 512, 3)
    """
    # Validate inputs
    if not 0.0 <= count <= 1.0:
        raise ValueError(f"Count must be in [0, 1], got {count}")
    
    # Early exit for zero count
    if count < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    
    # Calculate number of particles (spec: 0-50 particles)
    num_particles = int(count * 50)
    if num_particles == 0:
        return image.copy()
    
    # Create dust mask
    dust_mask = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(num_particles):
        # Random position
        center = (np.random.randint(0, w), np.random.randint(0, h))
        
        # Random ellipse size (spec: 2-8 pixels)
        axes = (
            np.random.randint(size_range[0], size_range[1]),
            np.random.randint(size_range[0], size_range[1])
        )
        
        # Random orientation
        angle = np.random.randint(0, 180)
        
        # Draw filled ellipse
        cv2.ellipse(dust_mask, center, axes, angle, 0, 360, 255, -1)
    
    # Apply Gaussian blur for soft edges (spec)
    dust_mask_blur = cv2.GaussianBlur(dust_mask, (5, 5), 2)
    
    # Apply as dark spots
    # Spec: dust darkens image with alpha blending
    defected = image.copy().astype(np.float32)
    alpha = (dust_mask_blur / 255.0)[:, :, np.newaxis] * 0.5  # Max 50% darkening
    
    # Darken image where dust is present
    defected = image * (1 - alpha) + (image * 0.3) * alpha
    
    return defected.astype(np.uint8)


def generate_clustered_dust(
    image: np.ndarray,
    count: float,
    cluster_probability: float = 0.3
) -> np.ndarray:
    """
    Generate dust with clustering (dust tends to accumulate).
    
    Args:
        image: Input image (H, W, 3) uint8
        count: Overall dust density [0, 1]
        cluster_probability: Probability of creating clusters [0, 1]
    
    Returns:
        Image with clustered dust (H, W, 3) uint8
    """
    if count < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    num_particles = int(count * 50)
    
    dust_mask = np.zeros((h, w), dtype=np.uint8)
    
    i = 0
    while i < num_particles:
        if np.random.rand() < cluster_probability and i < num_particles - 3:
            # Create cluster
            cluster_center = (np.random.randint(0, w), np.random.randint(0, h))
            cluster_size = np.random.randint(3, 8)
            
            for _ in range(cluster_size):
                if i >= num_particles:
                    break
                
                # Position near cluster center
                offset_x = int(np.random.randn() * 20)
                offset_y = int(np.random.randn() * 20)
                center = (
                    np.clip(cluster_center[0] + offset_x, 0, w - 1),
                    np.clip(cluster_center[1] + offset_y, 0, h - 1)
                )
                
                axes = (np.random.randint(2, 6), np.random.randint(2, 6))
                angle = np.random.randint(0, 180)
                cv2.ellipse(dust_mask, center, axes, angle, 0, 360, 255, -1)
                i += 1
        else:
            # Single particle
            center = (np.random.randint(0, w), np.random.randint(0, h))
            axes = (np.random.randint(2, 8), np.random.randint(2, 8))
            angle = np.random.randint(0, 180)
            cv2.ellipse(dust_mask, center, axes, angle, 0, 360, 255, -1)
            i += 1
    
    dust_mask_blur = cv2.GaussianBlur(dust_mask, (5, 5), 2)
    defected = image.copy().astype(np.float32)
    alpha = (dust_mask_blur / 255.0)[:, :, np.newaxis] * 0.5
    defected = image * (1 - alpha) + (image * 0.3) * alpha
    
    return defected.astype(np.uint8)


def add_hair_particles(
    image: np.ndarray,
    count: float
) -> np.ndarray:
    """
    Add hair/fiber particles (longer than dust spots).
    
    Args:
        image: Input image (H, W, 3) uint8
        count: Hair density [0, 1]
    
    Returns:
        Image with hair particles (H, W, 3) uint8
    """
    if count < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    num_hairs = int(count * 10)  # Fewer hairs than dust
    
    hair_mask = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(num_hairs):
        # Random start position
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        
        # Random length and angle
        length = np.random.randint(20, 100)
        angle = np.random.uniform(0, 2 * np.pi)
        
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)
        
        # Draw thin line
        cv2.line(hair_mask, (x1, y1), (x2, y2), 255, 1)
    
    hair_mask_blur = cv2.GaussianBlur(hair_mask, (3, 3), 1)
    defected = image.copy().astype(np.float32)
    alpha = (hair_mask_blur / 255.0)[:, :, np.newaxis] * 0.6
    defected = image * (1 - alpha) + (image * 0.2) * alpha
    
    return defected.astype(np.uint8)


if __name__ == "__main__":
    """Test script for dust generation."""
    print("Dust Particle Synthesis - Test Script")
    print("="*60)
    
    test_image = np.ones((512, 512, 3), dtype=np.uint8) * 128
    for i in range(512):
        test_image[i, :, :] = int(i / 2)
    
    print("\n1. Testing basic dust generation...")
    try:
        dusty_low = generate_dust(test_image, count=0.3)
        print(f"   ✓ Low count (0.3): shape {dusty_low.shape}")
        
        dusty_high = generate_dust(test_image, count=1.0)
        print(f"   ✓ High count (1.0): shape {dusty_high.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n2. Testing clustered dust...")
    try:
        clustered = generate_clustered_dust(test_image, count=0.5)
        print(f"   ✓ Clustered dust: shape {clustered.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n3. Testing hair particles...")
    try:
        hair = add_hair_particles(test_image, count=0.5)
        print(f"   ✓ Hair particles: shape {hair.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✅ Dust synthesis tests complete!")
    print("="*60)
