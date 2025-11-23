"""
Single Image Inference for VintageGAN
Specification Reference: Day 14

Apply vintage film effects to individual images using trained generator.

Author: VintageGAN Project
Date: 2024
"""

import argparse
from pathlib import Path
from typing import Union, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Generator
from defects import create_preset_conditions


class VintageFilter:
    """
    VintageGAN filter for applying vintage effects to images.
    
    Example:
        >>> filter = VintageFilter('checkpoints/generator_final.pth')
        >>> output = filter.apply('input.jpg', conditions={'grain': 0.7, 'vignette': 0.5})
        >>> output.save('output.jpg')
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize vintage filter.
        
        Args:
            checkpoint_path: Path to generator checkpoint
            device: Device to use
        """
        self.device = torch.device(device)
        
        # Load generator
        self.generator = Generator().to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'generator_state_dict' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            self.generator.load_state_dict(checkpoint)
        
        self.generator.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        ])
    
    def apply(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        conditions: Union[Dict[str, float], np.ndarray, str] = None,
        return_tensor: bool = False
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Apply vintage filter to image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            conditions: Defect conditions (dict, array, or preset name)
            return_tensor: Return tensor instead of PIL Image
        
        Returns:
            Processed image
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        # Transform to tensor
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Parse conditions
        cond_tensor = self._parse_conditions(conditions)
        
        # Generate
        with torch.no_grad():
            output = self.generator(img_tensor, cond_tensor)
        
        if return_tensor:
            return output.squeeze(0)
        
        # Convert to PIL Image
        output = self.denormalize(output.squeeze(0).cpu())
        output = output.clamp(0, 1)
        output = transforms.ToPILImage()(output)
        
        return output
    
    def _parse_conditions(self, conditions) -> torch.Tensor:
        """Parse conditions to tensor format."""
        if conditions is None:
            # Default medium preset
            cond = create_preset_conditions('medium')
        elif isinstance(conditions, str):
            # Preset name
            cond = create_preset_conditions(conditions)
        elif isinstance(conditions, dict):
            # Dictionary format
            defect_names = ['grain', 'scratch', 'dust', 'vignette', 'color_shift', 'blur']
            cond = np.array([conditions.get(name, 0.0) for name in defect_names], dtype=np.float32)
        elif isinstance(conditions, (list, tuple)):
            cond = np.array(conditions, dtype=np.float32)
        elif isinstance(conditions, np.ndarray):
            cond = conditions
        else:
            raise ValueError(f"Invalid conditions type: {type(conditions)}")
        
        # Convert to tensor
        cond_tensor = torch.from_numpy(cond).unsqueeze(0).to(self.device)
        
        return cond_tensor
    
    def apply_batch(
        self,
        images: list,
        conditions: Union[Dict[str, float], np.ndarray, str] = None
    ) -> list:
        """
        Apply filter to batch of images.
        
        Args:
            images: List of images
            conditions: Shared conditions for all images
        
        Returns:
            List of processed images
        """
        results = []
        for img in images:
            result = self.apply(img, conditions)
            results.append(result)
        return results


def main():
    """Command-line interface for vintage filter."""
    parser = argparse.ArgumentParser(description='Apply VintageGAN filter to images')
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('output', type=str, help='Output image path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to generator checkpoint')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['light', 'medium', 'heavy', 'grain_only', 'faded', 'scratched'],
                        help='Preset condition name')
    parser.add_argument('--grain', type=float, default=None, help='Grain intensity [0-1]')
    parser.add_argument('--scratch', type=float, default=None, help='Scratch density [0-1]')
    parser.add_argument('--dust', type=float, default=None, help='Dust count [0-1]')
    parser.add_argument('--vignette', type=float, default=None, help='Vignette strength [0-1]')
    parser.add_argument('--color-shift', type=float, default=None, help='Color shift [0-1]')
    parser.add_argument('--blur', type=float, default=None, help='Blur amount [0-1]')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VINTAGEGAN IMAGE FILTER")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    
    # Initialize filter
    print("\nLoading model...")
    filter = VintageFilter(args.checkpoint, device=args.device)
    
    # Parse conditions
    if args.preset:
        conditions = args.preset
        print(f"Using preset: {args.preset}")
    else:
        conditions = {}
        if args.grain is not None:
            conditions['grain'] = args.grain
        if args.scratch is not None:
            conditions['scratch'] = args.scratch
        if args.dust is not None:
            conditions['dust'] = args.dust
        if args.vignette is not None:
            conditions['vignette'] = args.vignette
        if args.color_shift is not None:
            conditions['color_shift'] = args.color_shift
        if args.blur is not None:
            conditions['blur'] = args.blur
        
        if not conditions:
            conditions = 'medium'
            print("No conditions specified, using 'medium' preset")
        else:
            print(f"Custom conditions: {conditions}")
    
    # Apply filter
    print("\nProcessing image...")
    output = filter.apply(args.input, conditions)
    
    # Save result
    output.save(args.output)
    print(f"\n✓ Saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
