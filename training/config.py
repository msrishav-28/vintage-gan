"""Configuration and reproducibility helpers for VintageGAN."""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base``."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str, profile: Optional[str] = None) -> Dict[str, Any]:
    """Load a YAML config and apply an optional profile override."""
    text = Path(config_path).read_text(encoding="utf-8")
    try:
        raw_config = yaml.safe_load(text)
    except yaml.YAMLError:
        raw_config = yaml.safe_load(text.replace("\\", "/"))

    profiles = raw_config.pop("profiles", {}) or {}
    active_profile = profile or raw_config.get("active_profile")
    config = copy.deepcopy(raw_config)

    if active_profile:
        if active_profile not in profiles:
            available = ", ".join(sorted(profiles))
            raise ValueError(
                f"Unknown profile '{active_profile}'. Available profiles: {available}"
            )
        _deep_update(config, copy.deepcopy(profiles[active_profile]))
        config["active_profile"] = active_profile

    return config


def seed_everything(
    seed: Optional[int], deterministic: bool = True, benchmark: bool = False
) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def get_gradient_accumulation_steps(config: Dict[str, Any]) -> int:
    """Return a safe gradient accumulation factor."""
    return max(1, int(config.get("hardware", {}).get("gradient_accumulation_steps", 1)))


def amp_enabled(config: Dict[str, Any], device: torch.device) -> bool:
    """Return whether CUDA automatic mixed precision should be enabled."""
    return bool(
        config.get("hardware", {}).get("mixed_precision", False)
        and device.type == "cuda"
    )


def get_git_commit() -> str:
    """Return the current git commit if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def write_experiment_metadata(
    config: Dict[str, Any],
    config_path: str,
    run_name: str,
    output_dir: str,
) -> Path:
    """Write a lightweight reproducibility manifest for a training run."""
    metadata_dir = Path(output_dir) / "experiments"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_name": run_name,
        "created_at_unix": int(time.time()),
        "git_commit": get_git_commit(),
        "config_path": config_path,
        "active_profile": config.get("active_profile"),
        "dataset": config.get("dataset", {}),
        "hardware": {
            "device_count": torch.cuda.device_count(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "configured": config.get("hardware", {}),
        },
        "reproducibility": config.get("reproducibility", {}),
    }

    path = metadata_dir / f"{run_name}_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path
