"""
imageprep.py - turn any input image into the model's standard format.

What this does:
    to_model_size(image, 224) makes a 224x224, 3-channel RGB image by
      1. converting to RGB (grayscale is copied into all 3 channels, because the
         pretrained model expects color input),
      2. padding the shorter side so the image is square WITHOUT stretching the
         shape (a stretched triangle would teach the model the wrong thing),
         using the color of the image's own border so the padding blends in,
      3. resizing to 224x224.
    load_photo(path) opens a photo the right way up: phones save rotation as a
    note in the file (EXIF data) instead of rotating the pixels.

Why:
    CLAUDE.md §5d. prepare_data.py uses this for every training image and
    predict.py will use it for live photos, so training and use see identical images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_photo(path: str | Path) -> Image.Image:
    """Open an image file, apply the phone's rotation note (EXIF orientation), return RGB."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)   # rotate the pixels the way the phone intended
        return flatten_to_rgb(image)


def flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB copy. Transparent areas (if any) become white, like paper."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image)
    return image.convert("RGB")


def border_color(image: Image.Image) -> tuple[int, int, int]:
    """Typical color of the image's outer edge (median of the border pixels)."""
    pixels = np.asarray(image)
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    return tuple(int(v) for v in np.median(border, axis=0))


def pad_to_square(image: Image.Image) -> Image.Image:
    """Center the image on a square canvas filled with its own border color."""
    width, height = image.size
    side = max(width, height)
    if width == height:
        return image
    canvas = Image.new("RGB", (side, side), border_color(image))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def to_model_size(image: Image.Image, size: int = 224) -> Image.Image:
    """RGB -> pad to square -> resize to size x size."""
    image = pad_to_square(flatten_to_rgb(image))
    # LANCZOS is a high-quality resizing method; it keeps thin pen lines visible.
    return image.resize((size, size), Image.LANCZOS)
