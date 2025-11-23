"""
Batch Processing for VintageGAN
Specification Reference: Day 14

Process multiple images in a folder with vintage effects.

Author: VintageGAN Project
Date: 2024
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.apply_filter import VintageFilter


def process_folder(
    input_dir: str,
    output_dir: str,
    checkpoint: str,
    conditions: str = 'medium',
    extensions: tuple = ('.jpg', '.jpeg', '.png'),
    device: str = 'cuda'
):
    """
    Process all images in a folder.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        checkpoint: Generator checkpoint path
        conditions: Condition preset or custom values
        extensions: Image file extensions to process
        device: Device to use
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Initialize filter
    print("Loading model...")
    filter = VintageFilter(checkpoint, device=device)
    
    # Process images
    print(f"\nProcessing with preset: {conditions}")
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # Apply filter
            output = filter.apply(img_path, conditions)
            
            # Save with same filename
            output_file = output_path / img_path.name
            output.save(output_file, quality=95)
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue
    
    print(f"\n✓ Processed {len(image_files)} images")
    print(f"✓ Saved to: {output_dir}")


def main():
    """Command-line interface for batch processing."""
    parser = argparse.ArgumentParser(description='Batch process images with VintageGAN')
    parser.add_argument('input_dir', type=str, help='Input directory')
    parser.add_argument('output_dir', type=str, help='Output directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to generator checkpoint')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['light', 'medium', 'heavy', 'grain_only', 'faded', 'scratched'],
                        help='Preset condition name')
    parser.add_argument('--device', type=str, 
                        default='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VINTAGEGAN BATCH PROCESSING")
    print("="*60)
    
    process_folder(
        args.input_dir,
        args.output_dir,
        args.checkpoint,
        args.preset,
        device=args.device
    )
    
    print("="*60)


if __name__ == "__main__":
    main()
