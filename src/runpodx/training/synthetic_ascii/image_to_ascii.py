from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True)
class AsciiRenderConfig:
    width: int = 64
    charset: str = "@%#*+=-:. "
    invert: bool = False


def render_image_to_ascii(image_path: str | Path | Image.Image, config: AsciiRenderConfig) -> str:
    if isinstance(image_path, Image.Image):
        image = image_path.convert("L")
    else:
        image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)
    if config.invert:
        image = ImageOps.invert(image)
    aspect = image.height / image.width if image.width else 1.0
    target_height = max(1, int(round(config.width * aspect * 0.55)))
    image = image.resize((config.width, target_height))
    pixel_access = image.load()
    chars = config.charset
    out_lines = []
    for row_idx in range(target_height):
        row = []
        for col_idx in range(config.width):
            value = pixel_access[col_idx, row_idx]
            char_idx = round((value / 255) * (len(chars) - 1))
            row.append(chars[char_idx])
        out_lines.append("".join(row).rstrip())

    # Guarantee at least one visible line for downstream filtering/tests.
    if not any(line for line in out_lines):
        out_lines = [chars[0] * config.width]
    return "\n".join(out_lines).rstrip() + "\n"
