"""Single-image VintageGAN inference with presets, sliders, and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from defects import (  # noqa: E402
    CONDITION_NAMES,
    condition_dict_to_vector,
    condition_vector_to_dict,
    create_preset_conditions,
    load_presets,
    validate_condition_vector,
)
from models import Generator  # noqa: E402
from training.checkpointing import load_checkpoint_file  # noqa: E402


class VintageFilter:
    """Apply a trained conditioned generator as a vintage filter."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        *,
        checkpoint: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        image_size: int = 256,
        presets_config: str = "configs/presets.yaml",
    ):
        self.checkpoint_path = checkpoint_path or checkpoint
        if self.checkpoint_path is None:
            raise ValueError("Provide checkpoint_path or checkpoint")

        self.device = torch.device(device)
        self.image_size = image_size
        self.presets_config = presets_config
        self.presets = load_presets(presets_config)

        self.generator = Generator().to(self.device)
        checkpoint_data = load_checkpoint_file(
            self.checkpoint_path, map_location=self.device
        )
        state_dict = checkpoint_data.get("generator_state_dict", checkpoint_data)
        self.generator.load_state_dict(state_dict)
        self.generator.eval()

    def apply(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        conditions: Union[Dict[str, float], np.ndarray, list, tuple, str, None] = None,
        return_tensor: bool = False,
    ) -> Union[Image.Image, torch.Tensor]:
        """Apply a preset name or six manual slider values to one image."""
        img = self._load_image(image)
        img_tensor = self._image_to_tensor(img).unsqueeze(0).to(self.device)
        cond_tensor = self._parse_conditions(conditions)

        with torch.no_grad():
            output = self.generator(img_tensor, cond_tensor)

        if return_tensor:
            return output.squeeze(0)

        return self._tensor_to_image(output.squeeze(0).cpu())

    def metadata_for(
        self,
        conditions: Union[Dict[str, float], np.ndarray, list, tuple, str, None],
        preset_name: Optional[str] = None,
    ) -> Dict[str, object]:
        """Create JSON-friendly metadata for an inference output."""
        vector = self._parse_conditions(conditions).squeeze(0).detach().cpu().numpy()
        return {
            "checkpoint": str(self.checkpoint_path),
            "image_size": self.image_size,
            "preset": preset_name,
            "condition_names": list(CONDITION_NAMES),
            "condition_vector": [float(v) for v in vector],
            "conditions": condition_vector_to_dict(vector),
        }

    def _load_image(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        return image.convert("RGB")

    def _resize_center_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.image_size / min(width, height)
        resized = image.resize(
            (round(width * scale), round(height * scale)), Image.BICUBIC
        )
        left = (resized.width - self.image_size) // 2
        top = (resized.height - self.image_size) // 2
        return resized.crop((left, top, left + self.image_size, top + self.image_size))

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = self._resize_center_crop(image)
        array = np.asarray(image).astype(np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        try:
            tensor = torch.from_numpy(array).float()
        except RuntimeError:
            tensor = torch.tensor(array.tolist(), dtype=torch.float32)
        return (tensor - 0.5) / 0.5

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
        try:
            array = tensor.numpy()
        except RuntimeError:
            array = np.array(tensor.tolist(), dtype=np.float32)
        array = (array.transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image.fromarray(array)

    def _parse_conditions(self, conditions) -> torch.Tensor:
        if conditions is None:
            cond = create_preset_conditions("soft_fade", self.presets_config)
        elif isinstance(conditions, str):
            cond = create_preset_conditions(conditions, self.presets_config)
        elif isinstance(conditions, dict):
            cond = condition_dict_to_vector(conditions)
        elif isinstance(conditions, (list, tuple, np.ndarray)):
            cond = validate_condition_vector(conditions)
        else:
            raise ValueError(f"Invalid conditions type: {type(conditions)}")

        cond = cond.astype(np.float32)
        try:
            tensor = torch.from_numpy(cond).float()
        except RuntimeError:
            tensor = torch.tensor(cond.tolist(), dtype=torch.float32)
        return tensor.unsqueeze(0).to(self.device)

    def apply_batch(
        self,
        images: list,
        conditions: Union[Dict[str, float], np.ndarray, list, tuple, str, None] = None,
    ) -> list:
        return [self.apply(img, conditions) for img in images]


def _manual_condition_dict(args: argparse.Namespace) -> Dict[str, float]:
    values = {}
    for name in CONDITION_NAMES:
        cli_name = name.replace("_", "-")
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a trained VintageGAN filter to one image"
    )
    parser.add_argument("input", type=str, help="Input image path")
    parser.add_argument("output", type=str, help="Output image path")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to generator checkpoint"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Named preset from configs/presets.yaml",
    )
    parser.add_argument(
        "--grain", type=float, default=None, help="Grain intensity [0-1]"
    )
    parser.add_argument(
        "--scratch", type=float, default=None, help="Scratch density [0-1]"
    )
    parser.add_argument("--dust", type=float, default=None, help="Dust count [0-1]")
    parser.add_argument(
        "--vignette", type=float, default=None, help="Vignette strength [0-1]"
    )
    parser.add_argument(
        "--color-shift",
        dest="color_shift",
        type=float,
        default=None,
        help="Color shift [0-1]",
    )
    parser.add_argument("--blur", type=float, default=None, help="Blur amount [0-1]")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Inference crop size"
    )
    parser.add_argument("--presets-config", type=str, default="configs/presets.yaml")
    parser.add_argument(
        "--metadata-output", type=str, default=None, help="Optional JSON metadata path"
    )
    args = parser.parse_args()

    manual_values = _manual_condition_dict(args)
    if manual_values:
        conditions = manual_values
        preset_name = None
    else:
        conditions = args.preset or "soft_fade"
        preset_name = conditions

    vintage_filter = VintageFilter(
        checkpoint=args.checkpoint,
        device=args.device,
        image_size=args.image_size,
        presets_config=args.presets_config,
    )

    output = vintage_filter.apply(args.input, conditions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)

    if args.metadata_output:
        metadata_path = Path(args.metadata_output)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = vintage_filter.metadata_for(conditions, preset_name=preset_name)
        metadata["input"] = args.input
        metadata["output"] = str(output_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved filtered image to {output_path}")


if __name__ == "__main__":
    main()
