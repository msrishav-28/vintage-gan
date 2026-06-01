"""
VintageGAN Defect Synthesis Module

Synthetic defect generation for vintage film effects.

Modules:
    grain: Film grain synthesis
    scratches: Scratch generation
    dust: Dust particles
    vignette: Vignetting effects
    color_shift: Color degradation
    blur: Motion/lens blur
    combined: Combined defect application
"""

from .grain import generate_film_grain, generate_colored_grain
from .scratches import generate_scratches
from .dust import generate_dust
from .vignette import generate_vignette
from .color_shift import generate_color_shift
from .blur import generate_blur
from .combined import (
    apply_vintage_defects,
    generate_random_conditions,
    batch_apply_defects,
    visualize_defect_progression,
)
from .presets import (
    CONDITION_NAMES,
    create_preset_conditions,
    condition_dict_to_vector,
    condition_vector_to_dict,
    load_presets,
    validate_condition_vector,
)

__all__ = [
    # Individual defects
    "generate_film_grain",
    "generate_colored_grain",
    "generate_scratches",
    "generate_dust",
    "generate_vignette",
    "generate_color_shift",
    "generate_blur",
    # Combined application
    "apply_vintage_defects",
    "generate_random_conditions",
    "create_preset_conditions",
    "batch_apply_defects",
    "visualize_defect_progression",
    "CONDITION_NAMES",
    "condition_dict_to_vector",
    "condition_vector_to_dict",
    "load_presets",
    "validate_condition_vector",
]

__version__ = "0.1.0"
__author__ = "VintageGAN Project"
