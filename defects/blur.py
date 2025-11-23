"""
Blur Effect Module
Specification Reference: Section 3.2.6

Implements combined motion and lens blur to simulate camera shake and defocus.

Author: VintageGAN Project
Date: 2024
"""

from typing import Tuple
import numpy as np
import cv2


def create_motion_blur_kernel(
    kernel_size: int,
    angle: float
) -> np.ndarray:
    """
    Create motion blur kernel.
    
    Args:
        kernel_size: Size of kernel (must be odd)
        angle: Blur direction in degrees
    
    Returns:
        Motion blur kernel (kernel_size, kernel_size)
    """
    # Create line kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # Draw line through center at specified angle
    angle_rad = np.radians(angle)
    
    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * np.cos(angle_rad))
        y = int(center + offset * np.sin(angle_rad))
        
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    
    # Normalize
    kernel = kernel / np.sum(kernel)
    
    return kernel.astype(np.float32)


def generate_blur(
    image: np.ndarray,
    amount: float,
    blur_type: str = "mixed"
) -> np.ndarray:
    """
    Generate blur effect on image.
    
    Specification Reference: Section 3.2.6 - Blur
    
    Algorithm:
    1. Determine blur type (motion or Gaussian)
    2. Calculate kernel size based on amount (3-13 pixels)
    3. Apply blur filter
    
    Args:
        image: Input image (H, W, 3) uint8 in range [0, 255]
        amount: Blur intensity in range [0.0, 1.0]
            0.0 = no blur
            1.0 = maximum blur
        blur_type: Type of blur ('motion', 'gaussian', 'mixed')
    
    Returns:
        Image with blur applied (H, W, 3) uint8
    
    Raises:
        ValueError: If amount not in [0, 1]
    
    Example:
        >>> img = np.ones((512, 512, 3), dtype=np.uint8) * 128
        >>> blurred = generate_blur(img, amount=0.5)
        >>> blurred.shape
        (512, 512, 3)
    """
    # Validate inputs
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"Amount must be in [0, 1], got {amount}")
    
    # Early exit for minimal blur (spec: < 0.1 returns original)
    if amount < 0.1:
        return image.copy()
    
    # Calculate kernel size (spec: 3-13 pixels based on amount)
    kernel_size = int(3 + amount * 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Determine blur type
    if blur_type == "mixed":
        # Random choice between motion and Gaussian (spec: 50/50)
        actual_blur_type = "motion" if np.random.random() < 0.5 else "gaussian"
    else:
        actual_blur_type = blur_type
    
    # Apply blur
    if actual_blur_type == "motion":
        # Motion blur at random angle
        angle = np.random.randint(0, 180)
        kernel = create_motion_blur_kernel(kernel_size, angle)
        defected = cv2.filter2D(image, -1, kernel)
    else:
        # Gaussian (lens defocus) - spec
        sigma = amount * 3.0
        defected = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    return defected


def generate_directional_blur(
    image: np.ndarray,
    amount: float,
    direction: str = "horizontal"
) -> np.ndarray:
    """
    Generate directional blur (horizontal or vertical).
    
    Args:
        image: Input image (H, W, 3) uint8
        amount: Blur intensity [0, 1]
        direction: Blur direction ('horizontal', 'vertical', 'diagonal')
    
    Returns:
        Image with directional blur (H, W, 3) uint8
    """
    if amount < 0.1:
        return image.copy()
    
    kernel_size = int(3 + amount * 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create directional kernel
    if direction == "horizontal":
        angle = 0
    elif direction == "vertical":
        angle = 90
    elif direction == "diagonal":
        angle = 45
    else:
        angle = np.random.randint(0, 180)
    
    kernel = create_motion_blur_kernel(kernel_size, angle)
    defected = cv2.filter2D(image, -1, kernel)
    
    return defected


def generate_radial_blur(
    image: np.ndarray,
    amount: float,
    center: Tuple[int, int] = None
) -> np.ndarray:
    """
    Generate radial blur (zoom blur effect).
    
    Args:
        image: Input image (H, W, 3) uint8
        amount: Blur intensity [0, 1]
        center: Center point for radial blur, or None for image center
    
    Returns:
        Image with radial blur (H, W, 3) uint8
    """
    if amount < 0.1:
        return image.copy()
    
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    
    # Number of iterations for blur effect
    num_iterations = int(amount * 10) + 1
    
    # Accumulate blurred versions
    result = np.zeros_like(image, dtype=np.float32)
    
    for i in range(num_iterations):
        # Calculate scale for this iteration
        scale = 1.0 + (i / num_iterations) * amount * 0.1
        
        # Resize and crop/pad to original size
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(image, (new_w, new_h))
        
        # Crop or pad to match original size
        if new_h >= h and new_w >= w:
            # Crop from center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            cropped = scaled[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad (shouldn't happen with scale > 1, but handle anyway)
            cropped = image.copy()
        
        result += cropped.astype(np.float32)
    
    # Average
    result /= num_iterations
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def generate_defocus_blur(
    image: np.ndarray,
    amount: float,
    bokeh_shape: str = "circle"
) -> np.ndarray:
    """
    Generate defocus blur with bokeh effect.
    
    Args:
        image: Input image (H, W, 3) uint8
        amount: Blur intensity [0, 1]
        bokeh_shape: Shape of bokeh ('circle', 'hexagon')
    
    Returns:
        Image with defocus blur (H, W, 3) uint8
    """
    if amount < 0.1:
        return image.copy()
    
    # Use larger Gaussian blur for defocus effect
    kernel_size = int(5 + amount * 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    sigma = amount * 5.0
    defected = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    return defected


def apply_partial_blur(
    image: np.ndarray,
    amount: float,
    blur_region: str = "edges"
) -> np.ndarray:
    """
    Apply blur to only part of image (depth of field simulation).
    
    Args:
        image: Input image (H, W, 3) uint8
        amount: Blur intensity [0, 1]
        blur_region: Which region to blur ('edges', 'center', 'top', 'bottom')
    
    Returns:
        Image with partial blur (H, W, 3) uint8
    """
    if amount < 0.1:
        return image.copy()
    
    h, w = image.shape[:2]
    
    # Generate blur mask
    mask = np.zeros((h, w), dtype=np.float32)
    
    if blur_region == "edges":
        # Blur edges, keep center sharp
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        mask = dist / max_dist  # 0 at center, 1 at edges
    
    elif blur_region == "center":
        # Blur center, keep edges sharp
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        mask = 1 - (dist / max_dist)  # 1 at center, 0 at edges
    
    elif blur_region == "top":
        mask = np.linspace(1, 0, h)[:, np.newaxis].repeat(w, axis=1)
    
    elif blur_region == "bottom":
        mask = np.linspace(0, 1, h)[:, np.newaxis].repeat(w, axis=1)
    
    # Smooth mask
    mask = cv2.GaussianBlur(mask, (51, 51), 20)
    mask = mask[:, :, np.newaxis]
    
    # Apply blur
    blurred = generate_blur(image, amount, blur_type="gaussian")
    
    # Blend based on mask
    defected = (image * (1 - mask) + blurred * mask).astype(np.uint8)
    
    return defected


if __name__ == "__main__":
    """Test script for blur effects."""
    print("Blur Effects - Test Script")
    print("="*60)
    
    test_image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    
    print("\n1. Testing basic blur...")
    try:
        blurred_low = generate_blur(test_image, amount=0.3)
        print(f"   ✓ Low amount (0.3): shape {blurred_low.shape}")
        
        blurred_high = generate_blur(test_image, amount=1.0)
        print(f"   ✓ High amount (1.0): shape {blurred_high.shape}")
        
        motion = generate_blur(test_image, amount=0.5, blur_type="motion")
        print(f"   ✓ Motion blur: shape {motion.shape}")
        
        gaussian = generate_blur(test_image, amount=0.5, blur_type="gaussian")
        print(f"   ✓ Gaussian blur: shape {gaussian.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n2. Testing directional blur...")
    try:
        horiz = generate_directional_blur(test_image, amount=0.5, direction="horizontal")
        print(f"   ✓ Horizontal blur: shape {horiz.shape}")
        
        vert = generate_directional_blur(test_image, amount=0.5, direction="vertical")
        print(f"   ✓ Vertical blur: shape {vert.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n3. Testing special blur effects...")
    try:
        radial = generate_radial_blur(test_image, amount=0.5)
        print(f"   ✓ Radial blur: shape {radial.shape}")
        
        defocus = generate_defocus_blur(test_image, amount=0.6)
        print(f"   ✓ Defocus blur: shape {defocus.shape}")
        
        partial = apply_partial_blur(test_image, amount=0.5, blur_region="edges")
        print(f"   ✓ Partial blur: shape {partial.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n4. Testing edge cases...")
    try:
        no_blur = generate_blur(test_image, amount=0.0)
        assert np.array_equal(no_blur, test_image), "Zero amount should return original"
        print("   ✓ Zero amount returns original image")
        
        try:
            generate_blur(test_image, amount=1.5)
            print("   ✗ Should have raised ValueError")
        except ValueError:
            print("   ✓ ValueError raised for invalid amount")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n✅ Blur effects tests complete!")
    print("="*60)
