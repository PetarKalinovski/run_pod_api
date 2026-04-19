from __future__ import annotations

from collections import deque
from io import BytesIO
from typing import Iterable

from PIL import Image


def _color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _border_points(width: int, height: int) -> Iterable[tuple[int, int]]:
    for x in range(width):
        yield x, 0
        yield x, height - 1
    for y in range(1, height - 1):
        yield 0, y
        yield width - 1, y


def remove_light_background(image: Image.Image, *, tolerance: int = 36, min_lightness: int = 200) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()

    seeds: list[tuple[int, int]] = []
    for x, y in _border_points(width, height):
        color = pixels[x, y]
        if min(color) >= min_lightness:
            seeds.append((x, y))

    if not seeds:
        return rgb

    ref = tuple(
        sum(pixels[x, y][channel] for x, y in seeds) // len(seeds)
        for channel in range(3)
    )

    visited = [[False for _ in range(height)] for _ in range(width)]
    queue: deque[tuple[int, int]] = deque(seeds)

    while queue:
        x, y = queue.popleft()
        if visited[x][y]:
            continue
        visited[x][y] = True
        color = pixels[x, y]
        if min(color) < min_lightness:
            continue
        if _color_distance(color, ref) > tolerance:
            continue
        pixels[x, y] = (255, 255, 255)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and not visited[nx][ny]:
                queue.append((nx, ny))

    return rgb


def apply_mask_to_white_background(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = mask.convert("L")
    rgba.putalpha(alpha)
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white, rgba)
    return composited.convert("RGB")


def remove_background_with_rembg_bytes(image: Image.Image, *, rembg_module=None) -> Image.Image:
    if rembg_module is None:
        import rembg as rembg_module  # type: ignore

    buf = BytesIO()
    image.convert("RGBA").save(buf, format="PNG")
    removed = rembg_module.remove(buf.getvalue())
    rgba = Image.open(BytesIO(removed)).convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white, rgba)
    return composited.convert("RGB")
