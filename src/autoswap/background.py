from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def resize_background(background: Image.Image, size: tuple[int, int]) -> Image.Image:
    return background.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def composite_foreground(
    foreground: Image.Image,
    subject_mask: Image.Image,
    background: Image.Image,
) -> Image.Image:
    fg = foreground.convert("RGB")
    bg = resize_background(background, fg.size)
    alpha = subject_mask.convert("L").resize(fg.size, Image.Resampling.BILINEAR)
    return Image.composite(fg, bg, alpha)


def save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")