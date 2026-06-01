"""Named VintageGAN condition presets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import yaml

CONDITION_NAMES = ("grain", "scratch", "dust", "vignette", "color_shift", "blur")

DEFAULT_PRESETS: Dict[str, np.ndarray] = {
    "soft_fade": np.array([0.25, 0.05, 0.08, 0.20, 0.45, 0.05], dtype=np.float32),
    "warm_film": np.array([0.40, 0.12, 0.10, 0.25, 0.65, 0.08], dtype=np.float32),
    "dusty_archive": np.array([0.45, 0.20, 0.65, 0.45, 0.55, 0.15], dtype=np.float32),
    "scratched_negative": np.array(
        [0.35, 0.85, 0.35, 0.35, 0.50, 0.18], dtype=np.float32
    ),
    "heavy_vintage": np.array([0.75, 0.55, 0.45, 0.70, 0.80, 0.35], dtype=np.float32),
    # Backward-compatible aliases used by the original CLI/docs.
    "light": np.array([0.30, 0.10, 0.10, 0.20, 0.20, 0.00], dtype=np.float32),
    "medium": np.array([0.50, 0.30, 0.20, 0.40, 0.50, 0.20], dtype=np.float32),
    "heavy": np.array([0.80, 0.60, 0.40, 0.70, 0.80, 0.40], dtype=np.float32),
    "grain_only": np.array([0.70, 0.00, 0.00, 0.00, 0.00, 0.00], dtype=np.float32),
    "faded": np.array([0.40, 0.20, 0.30, 0.50, 0.80, 0.10], dtype=np.float32),
    "scratched": np.array([0.30, 0.90, 0.40, 0.30, 0.40, 0.20], dtype=np.float32),
}


def validate_condition_vector(values: Iterable[float]) -> np.ndarray:
    """Validate and normalize a six-value condition vector."""
    vector = np.asarray(list(values), dtype=np.float32)
    if vector.shape != (6,):
        raise ValueError(
            f"Condition vector must contain exactly 6 values, got {vector.shape}"
        )
    if not np.all((vector >= 0.0) & (vector <= 1.0)):
        raise ValueError("Condition values must be in the [0, 1] range")
    return vector


def load_presets(config_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load presets from YAML, falling back to built-in defaults."""
    if config_path is None:
        config_path = str(
            Path(__file__).resolve().parent.parent / "configs" / "presets.yaml"
        )

    path = Path(config_path)
    if not path.exists():
        return dict(DEFAULT_PRESETS)

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    preset_values = raw.get("presets", raw)
    presets = {}
    for name, values in preset_values.items():
        if isinstance(values, Mapping):
            values = [values[key] for key in CONDITION_NAMES]
        presets[name] = validate_condition_vector(values)
    return presets


def create_preset_conditions(
    preset_name: str, presets_path: Optional[str] = None
) -> np.ndarray:
    """Return the six-dimensional condition vector for a named preset."""
    presets = load_presets(presets_path)
    if preset_name not in presets:
        choices = ", ".join(sorted(presets))
        raise ValueError(f"Unknown preset '{preset_name}'. Choose from: {choices}")
    return presets[preset_name].astype(np.float32)


def condition_dict_to_vector(values: Mapping[str, float]) -> np.ndarray:
    """Convert a named condition mapping to the canonical vector order."""
    return validate_condition_vector(values.get(name, 0.0) for name in CONDITION_NAMES)


def condition_vector_to_dict(values: Iterable[float]) -> Dict[str, float]:
    """Convert a condition vector to a JSON-friendly dictionary."""
    vector = validate_condition_vector(values)
    return {name: float(vector[idx]) for idx, name in enumerate(CONDITION_NAMES)}
