"""
Vignetting Effect Module
Specification Reference: Section 3.2.4

Implements radial gradient darkening to simulate lens vignetting.

Author: VintageGAN Project
Date: 2024
"""

from typing import Tuple
import numpy as np
import cv2


def generate_vignette(
    image: np.ndarray,
    strength: float,
    falloff_power: float = 1.5
) -> np.ndarray:
    """
    Generate vignetting effect on image.
    
    Specification Reference: Section 3.2.4 - Vignetting
    
    Algorithm:
    1. Create radial gradient from center
    2. Apply power function for natural falloff
    3. Multiply with image to darken edges
    
    Args:
        image: Input image (H, W, 3) uint8 in range [0, 255]
        strength: Vignette intensity in range [0.0, 1.0]
            0.0 = no vignetting
            1.0 = maximum vignetting
        falloff_power: Power for falloff curve (default: 1.5)
            Higher = sharper transition
    
    Returns:
        Image with vignetting applied (H, W, 3) uint8
    
    Raises:
        ValueError: If strength not in [0, 1]
    
    Example:
        >>> img = np.ones((512, 512, 3), dtype=np.uint8) * 200
        >>> vignetted = generate_vignette(img, strength=0.5)
        >>> vignetted.shape
        (512, 512, 3)
    """
    # Validate inputs
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Strength must be in [0, 1], got {strength}")
    
    # Early exit for zero strength
    if strength < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    
    # Create radial gradient from center (spec)
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Normalize to [0, 1]
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette_mask = dist_from_center / max_dist
    
    # Apply power function for natural falloff (spec: 1.5 + strength)
    vignette_mask = vignette_mask ** (falloff_power + strength)
    
    # Scale by strength (spec: max 70% darkening)
    vignette_mask = 1 - (vignette_mask * strength * 0.7)
    
    # Ensure mask doesn't go below reasonable darkness
    vignette_mask = np.clip(vignette_mask, 0.3, 1.0)
    
    # Apply to image
    vignette_mask = vignette_mask[:, :, np.newaxis]
    defected = image.astype(np.float32) * vignette_mask
    
    # Clip and convert back
    defected = np.clip(defected, 0, 255).astype(np.uint8)
    
    return defected


def generate_elliptical_vignette(
    image: np.ndarray,
    strength: float,
    aspect_ratio: float = 1.2
) -> np.ndarray:
    """
    Generate elliptical vignetting (common in wide-angle lenses).
    
    Args:
        image: Input image (H, W, 3) uint8
        strength: Vignette intensity [0, 1]
        aspect_ratio: Width/height ratio of ellipse
    
    Returns:
        Image with elliptical vignetting (H, W, 3) uint8
    """
    if strength < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    
    # Elliptical distance
    dist_x = (X - center_x) / (center_x / aspect_ratio)
    dist_y = (Y - center_y) / center_y
    dist_from_center = np.sqrt(dist_x**2 + dist_y**2)
    
    max_dist = np.sqrt((center_x / aspect_ratio)**2 + center_y**2)
    vignette_mask = dist_from_center / max_dist
    vignette_mask = vignette_mask ** (1.5 + strength)
    vignette_mask = 1 - (vignette_mask * strength * 0.7)
    vignette_mask = np.clip(vignette_mask, 0.3, 1.0)
    
    vignette_mask = vignette_mask[:, :, np.newaxis]
    defected = (image.astype(np.float32) * vignette_mask).astype(np.uint8)
    
    return defected


def generate_corner_vignette(
    image: np.ndarray,
    strength: float
) -> np.ndarray:
    """
    Generate corner-focused vignetting (emphasizes corners).
    
    Args:
        image: Input image (H, W, 3) uint8
        strength: Vignette intensity [0, 1]
    
    Returns:
        Image with corner vignetting (H, W, 3) uint8
    """
    if strength < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    
    # Calculate distance from each corner
    corners = [(0, 0), (0, w), (h, 0), (h, w)]
    vignette_mask = np.ones((h, w), dtype=np.float32)
    
    Y, X = np.ogrid[:h, :w]
    
    for cy, cx in corners:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(h**2 + w**2)
        corner_mask = 1 - (dist / max_dist)
        corner_mask = corner_mask ** (2.0 + strength)
        vignette_mask *= (1 - corner_mask * strength * 0.3)
    
    vignette_mask = np.clip(vignette_mask, 0.4, 1.0)
    vignette_mask = vignette_mask[:, :, np.newaxis]
    defected = (image.astype(np.float32) * vignette_mask).astype(np.uint8)
    
    return defected


if __name__ == "__main__":
    """Test script for vignetting."""
    print("Vignetting - Test Script")
    print("="*60)
    
    test_image = np.ones((512, 512, 3), dtype=np.uint8) * 200
    
    print("\n1. Testing basic vignetting...")
    try:
        vig_low = generate_vignette(test_image, strength=0.3)
        print(f"   ✓ Low strength (0.3): range [{vig_low.min()}, {vig_low.max()}]")
        
        vig_high = generate_vignette(test_image, strength=1.0)
        print(f"   ✓ High strength (1.0): range [{vig_high.min()}, {vig_high.max()}]")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n2. Testing elliptical vignetting...")
    try:
        ellip = generate_elliptical_vignette(test_image, strength=0.6)
        print(f"   ✓ Elliptical: shape {ellip.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n3. Testing corner vignetting...")
    try:
        corner = generate_corner_vignette(test_image, strength=0.5)
        print(f"   ✓ Corner: shape {corner.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✅ Vignetting tests complete!")
    print("="*60)
