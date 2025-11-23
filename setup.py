"""
VintageGAN Package Setup

Installation script for VintageGAN project.

Author: VintageGAN Project
Date: 2024
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='vintagegan',
    version='1.0.0',
    author='VintageGAN Project',
    author_email='your.email@university.edu',
    description='Parametric Controllable Vintage Film Defect Synthesis Using Conditional GANs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/VintageGAN',
    packages=find_packages(exclude=['tests', 'notebooks', 'data', 'checkpoints', 'logs', 'outputs']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'notebook': [
            'jupyter>=1.0.0',
            'ipywidgets>=7.6.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'vintagegan-filter=inference.apply_filter:main',
            'vintagegan-batch=inference.batch_process:main',
            'vintagegan-pretrain=training.pretrain:main',
            'vintagegan-train=training.gan_train:main',
            'vintagegan-ablation=evaluation.ablation:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
