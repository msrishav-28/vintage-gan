"""
Film Grain Synthesis Module
Specification Reference: Section 3.2.1

Implements frequency-based grain pattern generation using FFT filtering.
The intensity parameter (0.0-1.0) controls grain visibility by adjusting
the frequency bandpass filter cutoff.

Author: VintageGAN Project
Date: 2024
"""

from typing import Tuple
import numpy as np
import cv2


def create_bandpass_filter(
    height: int,
    width: int,
    low_freq: float = 0.1,
    high_freq: float = 0.5
) -> np.ndarray:
    """
    Create bandpass frequency filter for grain texture.
    
    Film grain has characteristic frequency spectrum - not pure noise.
    This filter allows only certain frequency ranges to pass through.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        low_freq: Low frequency cutoff (0.0-1.0)
        high_freq: High frequency cutoff (0.0-1.0)
    
    Returns:
        Bandpass filter mask (height, width) with values in [0, 1]
    
    Example:
        >>> filter_mask = create_bandpass_filter(512, 512, 0.1, 0.5)
        >>> filter_mask.shape
        (512, 512)
    """
    # Create frequency grid
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance from center (DC component)
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Normalize to [0, 1]
    max_dist = np.sqrt(center_x**2 + center_y**2)
    normalized_dist = dist_from_center / max_dist
    
    # Create bandpass: allow frequencies between low and high cutoff
    bandpass = np.zeros((height, width), dtype=np.float32)
    mask = (normalized_dist >= low_freq) & (normalized_dist <= high_freq)
    bandpass[mask] = 1.0
    
    # Smooth the transition
    bandpass = cv2.GaussianBlur(bandpass, (5, 5), 1.0)
    
    return bandpass


def generate_film_grain(
    image: np.ndarray,
    intensity: float,
    grain_size: str = "medium"
) -> np.ndarray:
    """
    Generate film grain and apply to image.
    
    Specification Reference: Section 3.2.1 - Film Grain Synthesis
    
    Algorithm:
    1. Generate Gaussian noise base
    2. Apply frequency filtering (film grain has characteristic spectrum)
    3. Add to image with luma blending
    4. Clip values to valid range
    
    Parameters (from spec):
    - Low intensity (0.0-0.3): Fine grain, ISO 100-400 equivalent
    - Medium (0.4-0.6): Noticeable grain, ISO 800-1600
    - High (0.7-1.0): Heavy grain, ISO 3200+ or pushed film
    
    Args:
        image: Input image (H, W, 3) uint8 in range [0, 255]
        intensity: Grain intensity in range [0.0, 1.0]
            0.0 = no grain, 1.0 = maximum grain
        grain_size: Grain coarseness ('fine', 'medium', 'coarse')
    
    Returns:
        Image with grain applied (H, W, 3) uint8 in range [0, 255]
    
    Raises:
        ValueError: If intensity not in [0, 1] or invalid grain_size
    
    Example:
        >>> import numpy as np
        >>> img = np.ones((512, 512, 3), dtype=np.uint8) * 128
        >>> grainy = generate_film_grain(img, intensity=0.5)
        >>> grainy.shape
        (512, 512, 3)
    """
    # Validate inputs
    if not 0.0 <= intensity <= 1.0:
        raise ValueError(f"Intensity must be in [0, 1], got {intensity}")
    
    if grain_size not in ['fine', 'medium', 'coarse']:
        raise ValueError(f"grain_size must be 'fine', 'medium', or 'coarse', got {grain_size}")
    
    # Early exit for zero intensity
    if intensity < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    
    # Generate Gaussian noise base
    # Scale by intensity - higher intensity = more visible grain
    noise_std = intensity * 25.0  # Max std of 25 for intensity=1.0
    noise = np.random.normal(0, noise_std, (h, w)).astype(np.float32)
    
    # Set frequency limits based on grain size
    if grain_size == 'fine':
        low_freq, high_freq = 0.2, 0.7  # Higher frequencies = finer grain
    elif grain_size == 'medium':
        low_freq, high_freq = 0.1, 0.5  # Default from spec
    else:  # coarse
        low_freq, high_freq = 0.05, 0.3  # Lower frequencies = coarser grain
    
    # Adjust high frequency based on intensity (spec: higher intensity = coarser)
    high_freq = high_freq - (intensity * 0.2)
    
    # Apply frequency filtering using FFT
    noise_fft = np.fft.fft2(noise)
    noise_fft_shifted = np.fft.fftshift(noise_fft)
    
    # Create and apply bandpass filter
    freq_mask = create_bandpass_filter(h, w, low_freq, high_freq)
    filtered_fft = noise_fft_shifted * freq_mask
    
    # Convert back to spatial domain
    filtered_fft_unshifted = np.fft.ifftshift(filtered_fft)
    filtered_noise = np.real(np.fft.ifft2(filtered_fft_unshifted))
    
    # Normalize filtered noise to reasonable range
    filtered_noise = filtered_noise.astype(np.float32)
    
    # Apply grain to image
    # Convert image to float for processing
    image_float = image.astype(np.float32)
    
    # Add grain to each channel
    # Grain is added as luma (affects all channels similarly)
    grain_layer = filtered_noise[:, :, np.newaxis]
    defected = image_float + grain_layer
    
    # Clip to valid range and convert back to uint8
    defected = np.clip(defected, 0, 255).astype(np.uint8)
    
    return defected


def generate_colored_grain(
    image: np.ndarray,
    intensity: float,
    color_variation: float = 0.3
) -> np.ndarray:
    """
    Generate colored film grain (chroma noise).
    
    Some film stocks exhibit colored grain, especially in shadows.
    This adds separate noise to each color channel with slight variations.
    
    Args:
        image: Input image (H, W, 3) uint8
        intensity: Grain intensity [0, 1]
        color_variation: Amount of color variation [0, 1]
            0.0 = monochrome grain, 1.0 = full color variation
    
    Returns:
        Image with colored grain (H, W, 3) uint8
    
    Example:
        >>> img = np.ones((512, 512, 3), dtype=np.uint8) * 128
        >>> colored_grain = generate_colored_grain(img, 0.5, 0.3)
    """
    if intensity < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    image_float = image.astype(np.float32)
    
    # Generate separate grain for each channel
    grain_channels = []
    base_std = intensity * 20.0
    
    for c in range(3):
        # Vary the grain strength slightly per channel
        channel_std = base_std * (1.0 + (np.random.rand() - 0.5) * color_variation)
        noise = np.random.normal(0, channel_std, (h, w)).astype(np.float32)
        
        # Apply frequency filtering
        noise_fft = np.fft.fft2(noise)
        noise_fft_shifted = np.fft.fftshift(noise_fft)
        freq_mask = create_bandpass_filter(h, w, 0.1, 0.5 - intensity * 0.2)
        filtered_fft = noise_fft_shifted * freq_mask
        filtered_noise = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))
        
        grain_channels.append(filtered_noise)
    
    # Stack and add to image
    grain = np.stack(grain_channels, axis=2)
    defected = image_float + grain
    defected = np.clip(defected, 0, 255).astype(np.uint8)
    
    return defected


def add_grain_variation(
    image: np.ndarray,
    intensity: float,
    variation_scale: float = 0.2
) -> np.ndarray:
    """
    Add spatially-varying grain intensity (like film exposure variations).
    
    Real film grain intensity varies across the frame due to:
    - Uneven exposure
    - Development variations
    - Film density differences
    
    Args:
        image: Input image (H, W, 3) uint8
        intensity: Base grain intensity [0, 1]
        variation_scale: Amount of spatial variation [0, 1]
    
    Returns:
        Image with varied grain (H, W, 3) uint8
    """
    if intensity < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    
    # Create smooth variation map
    variation = np.random.rand(h // 8, w // 8).astype(np.float32)
    variation = cv2.resize(variation, (w, h), interpolation=cv2.INTER_CUBIC)
    variation = cv2.GaussianBlur(variation, (51, 51), 20)
    
    # Normalize to [1-variation_scale, 1+variation_scale]
    variation = 1.0 + (variation - 0.5) * variation_scale * 2
    
    # Apply grain with varying intensity
    image_float = image.astype(np.float32)
    
    for y in range(0, h, 64):
        for x in range(0, w, 64):
            y_end = min(y + 64, h)
            x_end = min(x + 64, w)
            patch = image[y:y_end, x:x_end]
            
            # Get local variation
            local_intensity = intensity * np.mean(variation[y:y_end, x:x_end])
            local_intensity = np.clip(local_intensity, 0, 1)
            
            # Apply grain to patch
            grainy_patch = generate_film_grain(patch, local_intensity)
            image_float[y:y_end, x:x_end] = grainy_patch
    
    return image_float.astype(np.uint8)


if __name__ == "__main__":
    """
    Test script for film grain synthesis.
    
    Usage:
        python defects/grain.py
    """
    print("Film Grain Synthesis - Test Script")
    print("="*60)
    
    # Create test image
    test_image = np.ones((512, 512, 3), dtype=np.uint8) * 128
    
    # Add gradient for better visualization
    for i in range(512):
        test_image[i, :, :] = int(i / 2)
    
    print("\n1. Testing basic grain generation...")
    try:
        grain_low = generate_film_grain(test_image, intensity=0.3)
        print(f"   ✓ Low intensity (0.3): shape {grain_low.shape}, range [{grain_low.min()}, {grain_low.max()}]")
        
        grain_med = generate_film_grain(test_image, intensity=0.6)
        print(f"   ✓ Medium intensity (0.6): shape {grain_med.shape}, range [{grain_med.min()}, {grain_med.max()}]")
        
        grain_high = generate_film_grain(test_image, intensity=1.0)
        print(f"   ✓ High intensity (1.0): shape {grain_high.shape}, range [{grain_high.min()}, {grain_high.max()}]")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n2. Testing grain sizes...")
    try:
        grain_fine = generate_film_grain(test_image, 0.5, grain_size='fine')
        print(f"   ✓ Fine grain: shape {grain_fine.shape}")
        
        grain_coarse = generate_film_grain(test_image, 0.5, grain_size='coarse')
        print(f"   ✓ Coarse grain: shape {grain_coarse.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n3. Testing colored grain...")
    try:
        colored = generate_colored_grain(test_image, 0.5, color_variation=0.3)
        print(f"   ✓ Colored grain: shape {colored.shape}, range [{colored.min()}, {colored.max()}]")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n4. Testing edge cases...")
    try:
        # Zero intensity should return original
        no_grain = generate_film_grain(test_image, intensity=0.0)
        assert np.array_equal(no_grain, test_image), "Zero intensity should return original"
        print("   ✓ Zero intensity returns original image")
        
        # Invalid intensity should raise error
        try:
            generate_film_grain(test_image, intensity=1.5)
            print("   ✗ Should have raised ValueError for invalid intensity")
        except ValueError:
            print("   ✓ ValueError raised for invalid intensity")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✅ Film grain synthesis tests complete!")
    print("="*60)
