#!/usr/bin/env python
"""
Automated Notebook Compiler for VintageGAN

Executes all Jupyter notebooks and generates results automatically.
Useful for generating all figures, tables, and analysis for the report.

Usage:
    python compile_notebooks.py [--output-dir DIR] [--format html pdf]

Author: VintageGAN Project
Date: 2024
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def execute_notebook(notebook_path: Path, output_format: str = 'notebook') -> bool:
    """
    Execute a Jupyter notebook using nbconvert.
    
    Args:
        notebook_path: Path to notebook file
        output_format: Output format ('notebook', 'html', 'pdf')
    
    Returns:
        True if successful
    """
    print(f"\n{'='*60}")
    print(f"Executing: {notebook_path.name}")
    print(f"{'='*60}")
    
    try:
        # Execute and convert
        cmd = [
            'jupyter', 'nbconvert',
            '--to', output_format,
            '--execute',
            '--ExecutePreprocessor.timeout=600',
            '--output-dir', str(notebook_path.parent),
            str(notebook_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900  # 15 minutes max
        )
        
        if result.returncode == 0:
            print(f"✓ Successfully executed: {notebook_path.name}")
            return True
        else:
            print(f"✗ Error executing: {notebook_path.name}")
            print(result.stderr)
            return False
    
    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout executing: {notebook_path.name}")
        return False
    except FileNotFoundError:
        print("✗ jupyter nbconvert not found. Install with: pip install nbconvert")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Compile all VintageGAN notebooks')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--format', type=str, nargs='+', default=['notebook'],
                        choices=['notebook', 'html', 'pdf'],
                        help='Output formats')
    parser.add_argument('--notebooks', type=str, nargs='+', default=None,
                        help='Specific notebooks to compile (default: all)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VINTAGEGAN NOTEBOOK COMPILER")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output formats: {', '.join(args.format)}")
    
    # Find notebooks
    notebooks_dir = Path(__file__).parent / 'notebooks'
    
    if args.notebooks:
        notebooks = [notebooks_dir / nb if not nb.endswith('.ipynb') else notebooks_dir / f"{nb}.ipynb"
                    for nb in args.notebooks]
    else:
        # All notebooks except demo.ipynb (interactive only)
        notebooks = [
            notebooks_dir / 'results_analysis.ipynb',
        ]
    
    # Filter existing notebooks
    notebooks = [nb for nb in notebooks if nb.exists()]
    
    if not notebooks:
        print("\n✗ No notebooks found to compile")
        return 1
    
    print(f"\nFound {len(notebooks)} notebook(s) to compile:")
    for nb in notebooks:
        print(f"  • {nb.name}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute notebooks
    results = []
    
    for notebook in notebooks:
        for fmt in args.format:
            success = execute_notebook(notebook, output_format=fmt)
            results.append((notebook.name, fmt, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPILATION SUMMARY")
    print(f"{'='*60}")
    
    total = len(results)
    passed = sum(1 for _, _, success in results if success)
    failed = total - passed
    
    print(f"\n{'Notebook':<30} {'Format':<10} {'Status':<10}")
    print("-" * 60)
    for nb_name, fmt, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"{nb_name:<30} {fmt:<10} {status:<10}")
    
    print(f"\nTotal: {passed}/{total} compilations successful")
    
    if passed == total:
        print(f"\n🎉 All notebooks compiled successfully!")
        print(f"\nResults saved to: {output_dir.absolute()}/")
        print("\nGenerated files can be used directly in your report.")
        return 0
    else:
        print(f"\n⚠️  {failed} compilation(s) failed")
        print("\nTip: Make sure you have trained the model first")
        print("     or use dummy data for testing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
