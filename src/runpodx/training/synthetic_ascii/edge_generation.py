from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass(frozen=True)
class EdgeGenerationConfig:
    method: str = "find_edges"
    invert: bool = False
    blur_sigma: float = 0.0
    threshold: float | None = None
    dilate_radius: int = 0
    erode_radius: int = 0
    open_radius: int = 0
    close_radius: int = 0
    post_blur_sigma: float = 0.0
    canny_low_threshold: float | None = None
    canny_high_threshold: float | None = None


def _require_skimage() -> tuple[object, object, object, object, object]:
    try:
        from skimage import exposure, feature, filters, morphology, util
    except ImportError as exc:
        raise ImportError(
            "Advanced edge-generation methods require scikit-image in the active Python environment."
        ) from exc
    return exposure, feature, filters, morphology, util


def _to_array(image: Image.Image, *, blur_sigma: float = 0.0) -> np.ndarray:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    if blur_sigma > 0:
        exposure, _, filters, _, _ = _require_skimage()
        arr = filters.gaussian(arr, sigma=blur_sigma, preserve_range=True)
        arr = exposure.rescale_intensity(arr, in_range="image", out_range=(0.0, 1.0))
    return arr


def _base_edge_array(image: Image.Image, config: EdgeGenerationConfig) -> np.ndarray:
    if config.method in {"find_edges", "contour", "edge_enhance"}:
        gray = image.convert("L")
        if config.method == "find_edges":
            edge = gray.filter(ImageFilter.FIND_EDGES)
        elif config.method == "contour":
            edge = gray.filter(ImageFilter.CONTOUR)
        else:
            edge = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return np.asarray(edge, dtype=np.float32) / 255.0

    exposure, feature, filters, _, util = _require_skimage()
    arr = _to_array(image, blur_sigma=config.blur_sigma)

    if config.method == "sobel":
        edge = filters.sobel(arr)
    elif config.method == "scharr":
        edge = filters.scharr(arr)
    elif config.method == "prewitt":
        edge = filters.prewitt(arr)
    elif config.method == "roberts":
        edge = filters.roberts(arr)
    elif config.method == "laplace":
        edge = np.abs(filters.laplace(arr))
    elif config.method == "canny":
        edge = feature.canny(
            util.img_as_float(arr),
            sigma=max(config.blur_sigma, 1.0),
            low_threshold=config.canny_low_threshold,
            high_threshold=config.canny_high_threshold,
        ).astype(np.float32)
    elif config.method == "threshold":
        threshold = config.threshold if config.threshold is not None else filters.threshold_otsu(arr)
        edge = (arr <= threshold).astype(np.float32)
    elif config.method == "hybrid_threshold_sobel":
        threshold_cfg = EdgeGenerationConfig(
            method="threshold",
            threshold=config.threshold if config.threshold is not None else 0.7,
            blur_sigma=config.blur_sigma if config.blur_sigma > 0 else 1.0,
            close_radius=max(1, config.close_radius),
        )
        sobel_cfg = EdgeGenerationConfig(
            method="sobel",
            blur_sigma=config.blur_sigma if config.blur_sigma > 0 else 1.0,
            threshold=0.12,
            close_radius=max(1, config.close_radius),
            dilate_radius=max(1, config.dilate_radius),
        )
        threshold_edge = _apply_post_processing(_base_edge_array(image, threshold_cfg), threshold_cfg)
        sobel_edge = _apply_post_processing(_base_edge_array(image, sobel_cfg), sobel_cfg)
        edge = np.maximum(threshold_edge, sobel_edge)
    else:
        raise ValueError(f"Unsupported edge method: {config.method}")

    return exposure.rescale_intensity(edge, in_range="image", out_range=(0.0, 1.0))


def _apply_post_processing(edge: np.ndarray, config: EdgeGenerationConfig) -> np.ndarray:
    exposure, _, filters, morphology, _ = _require_skimage()
    edge = exposure.rescale_intensity(edge, in_range="image", out_range=(0.0, 1.0))

    if config.threshold is not None and config.method != "threshold":
        edge = (edge >= config.threshold).astype(np.float32)

    binary = edge > 0.5
    if config.open_radius > 0:
        binary = morphology.opening(binary, morphology.disk(config.open_radius))
    if config.close_radius > 0:
        binary = morphology.closing(binary, morphology.disk(config.close_radius))
    if config.dilate_radius > 0:
        binary = morphology.dilation(binary, morphology.disk(config.dilate_radius))
    if config.erode_radius > 0:
        binary = morphology.erosion(binary, morphology.disk(config.erode_radius))

    if any(
        radius > 0
        for radius in (config.open_radius, config.close_radius, config.dilate_radius, config.erode_radius)
    ):
        edge = binary.astype(np.float32)

    if config.post_blur_sigma > 0:
        edge = filters.gaussian(edge, sigma=config.post_blur_sigma, preserve_range=True)

    return exposure.rescale_intensity(edge, in_range="image", out_range=(0.0, 1.0))


def generate_edge_image(image: Image.Image, config: EdgeGenerationConfig) -> Image.Image:
    edge = _base_edge_array(image, config)

    if config.method != "hybrid_threshold_sobel" and any(
        value not in (0, 0.0, None)
        for value in (
            config.threshold,
            config.dilate_radius,
            config.erode_radius,
            config.open_radius,
            config.close_radius,
            config.post_blur_sigma,
        )
    ):
        edge = _apply_post_processing(edge, config)

    edge_img = Image.fromarray(np.clip(edge * 255.0, 0, 255).astype(np.uint8), mode="L")
    edge_img = ImageOps.autocontrast(edge_img)
    if config.invert:
        edge_img = ImageOps.invert(edge_img)
    return edge_img.convert("RGB")
