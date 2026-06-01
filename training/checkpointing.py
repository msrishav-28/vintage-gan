"""Checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import torch


def load_checkpoint_file(path: Union[str, Path], map_location: Any = "cpu") -> Any:
    """Load a checkpoint with PyTorch's safer weights-only path when available."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)
