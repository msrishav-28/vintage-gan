"""
Color Shift Module
Specification Reference: Section 3.2.5

Implements vintage film stock color degradation and fading.

Author: VintageGAN Project
Date: 2024
"""

from typing import Tuple
import numpy as np
import cv2


def generate_color_shift(
    image: np.ndarray,
    shift_amount: float,
    shift_type: str = "sepia"
) -> np.ndarray:
    """
    Generate color shift/degradation effect.
    
    Specification Reference: Section 3.2.5 - Color Shift
    
    Algorithm:
    1. Convert to LAB color space
    2. Shift toward sepia/faded tones
    3. Reduce saturation
    4. Convert back to RGB
    
    Args:
        image: Input image (H, W, 3) uint8 in range [0, 255]
        shift_amount: Color degradation intensity in range [0.0, 1.0]
            0.0 = no shift
            1.0 = maximum shift
        shift_type: Type of color shift ('sepia', 'cyan', 'magenta', 'yellow')
    
    Returns:
        Image with color shift applied (H, W, 3) uint8
    
    Raises:
        ValueError: If shift_amount not in [0, 1]
    
    Example:
        >>> img = np.ones((512, 512, 3), dtype=np.uint8) * 128
        >>> shifted = generate_color_shift(img, shift_amount=0.5)
        >>> shifted.shape
        (512, 512, 3)
    """
    # Validate inputs
    if not 0.0 <= shift_amount <= 1.0:
        raise ValueError(f"Shift amount must be in [0, 1], got {shift_amount}")
    
    # Early exit for zero shift
    if shift_amount < 0.01:
        return image.copy()
    
    # Convert to LAB color space (spec)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    # Apply color shift based on type
    if shift_type == "sepia":
        # Shift toward warm/sepia tones (spec: a+20, b+30)
        a = a * (1 - shift_amount * 0.4) + shift_amount * 20
        b = b * (1 - shift_amount * 0.4) + shift_amount * 30
    elif shift_type == "cyan":
        # Cyan shift (faded blues)
        a = a - shift_amount * 15
        b = b - shift_amount * 10
    elif shift_type == "magenta":
        # Magenta shift
        a = a + shift_amount * 25
        b = b - shift_amount * 10
    elif shift_type == "yellow":
        # Yellow shift
        a = a - shift_amount * 10
        b = b + shift_amount * 25
    
    # Reduce saturation (spec: reduce by 30%)
    a = a * (1 - shift_amount * 0.3)
    b = b * (1 - shift_amount * 0.3)
    
    # Clip to valid LAB range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    
    # Merge and convert back to RGB
    lab = cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
    defected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return defected


def apply_fade_effect(
    image: np.ndarray,
    fade_amount: float
) -> np.ndarray:
    """
    Apply overall fading (brightness + desaturation).
    
    Simulates aged/faded film appearance.
    
    Args:
        image: Input image (H, W, 3) uint8
        fade_amount: Fade intensity [0, 1]
    
    Returns:
        Faded image (H, W, 3) uint8
    """
    if fade_amount < 0.01:
        return image.copy()
    
    # Convert to HSV for saturation control
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # Reduce saturation
    s = s * (1 - fade_amount * 0.6)
    
    # Slightly increase brightness (faded look)
    v = v * (1 + fade_amount * 0.2)
    v = np.clip(v, 0, 255)
    
    hsv = cv2.merge([h, s, v])
    defected = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return defected


def apply_color_cast(
    image: np.ndarray,
    cast_color: Tuple[int, int, int],
    intensity: float
) -> np.ndarray:
    """
    Apply color cast overlay.
    
    Args:
        image: Input image (H, W, 3) uint8
        cast_color: RGB color tuple (R, G, B)
        intensity: Cast intensity [0, 1]
    
    Returns:
        Image with color cast (H, W, 3) uint8
    """
    if intensity < 0.01:
        return image.copy()
    
    # Create color overlay
    cast = np.full_like(image, cast_color, dtype=np.float32)
    image_float = image.astype(np.float32)
    
    # Blend with original
    defected = image_float * (1 - intensity * 0.3) + cast * (intensity * 0.3)
    defected = np.clip(defected, 0, 255).astype(np.uint8)
    
    return defected


def simulate_film_stock_colors(
    image: np.ndarray,
    stock_type: str,
    intensity: float = 0.5
) -> np.ndarray:
    """
    Simulate specific vintage film stock color profiles.
    
    Args:
        image: Input image (H, W, 3) uint8
        stock_type: Film stock ('kodachrome', 'ektachrome', 'fuji', 'agfa')
        intensity: Effect intensity [0, 1]
    
    Returns:
        Image with film stock colors (H, W, 3) uint8
    """
    if intensity < 0.01:
        return image.copy()
    
    if stock_type == "kodachrome":
        # Warm, saturated colors
        shifted = generate_color_shift(image, intensity * 0.5, "sepia")
        # Boost saturation slightly
        hsv = cv2.cvtColor(shifted, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * (1 + intensity * 0.2), 0, 255)
        hsv = cv2.merge([h, s, v])
        defected = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    elif stock_type == "ektachrome":
        # Cooler tones, slightly cyan
        defected = generate_color_shift(image, intensity * 0.6, "cyan")
    
    elif stock_type == "fuji":
        # Greenish cast in shadows
        defected = generate_color_shift(image, intensity * 0.5, "yellow")
    
    elif stock_type == "agfa":
        # Magenta shadows
        defected = generate_color_shift(image, intensity * 0.6, "magenta")
    
    else:
        defected = image.copy()
    
    return defected


if __name__ == "__main__":
    """Test script for color shift."""
    print("Color Shift - Test Script")
    print("="*60)
    
    test_image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    
    print("\n1. Testing basic color shift...")
    try:
        sepia = generate_color_shift(test_image, shift_amount=0.5, shift_type="sepia")
        print(f"   ✓ Sepia shift: shape {sepia.shape}")
        
        cyan = generate_color_shift(test_image, shift_amount=0.5, shift_type="cyan")
        print(f"   ✓ Cyan shift: shape {cyan.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n2. Testing fade effect...")
    try:
        faded = apply_fade_effect(test_image, fade_amount=0.6)
        print(f"   ✓ Faded: shape {faded.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n3. Testing color cast...")
    try:
        cast = apply_color_cast(test_image, (200, 150, 100), intensity=0.5)
        print(f"   ✓ Color cast: shape {cast.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n4. Testing film stock simulation...")
    try:
        kodak = simulate_film_stock_colors(test_image, "kodachrome", intensity=0.6)
        print(f"   ✓ Kodachrome: shape {kodak.shape}")
        
        fuji = simulate_film_stock_colors(test_image, "fuji", intensity=0.6)
        print(f"   ✓ Fuji: shape {fuji.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✅ Color shift tests complete!")
    print("="*60)
