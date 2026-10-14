"""Blur detection via OpenCV Laplacian variance.

The Laplacian variance of a grayscale patch is a classic, robust focus metric:
sharp images have high variance (strong second derivatives at edges), defocused
images have near-zero variance.

To avoid *megapixel bias* (a 48MP frame always has more edge pixels than a
4MP one), the metric is computed on a center-cropped, fixed-size patch so the
value is comparable across resolutions.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import QualityThresholds


def _center_crop_patch(bgr: np.ndarray, side: int = 1024) -> np.ndarray:
    """Center-crop and resize to a square ``side`` patch for stable comparison."""
    h, w = bgr.shape[:2]
    s = min(h, w)
    if s <= 0:
        return bgr
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    patch = bgr[y0:y0 + s, x0:x0 + s]
    if s != side:
        patch = cv2.resize(patch, (side, side), interpolation=cv2.INTER_AREA)
    return patch


def compute_laplacian_variance(bgr: np.ndarray, patch_side: int = 1024) -> float:
    """Laplacian variance of a center-cropped, resolution-normalized patch."""
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    patch = _center_crop_patch(gray, patch_side)
    lap = cv2.Laplacian(patch, cv2.CV_64F)
    return float(lap.var())


def compute_edge_frequency(bgr: np.ndarray, patch_side: int = 512) -> float:
    """Secondary metric: fraction of strong-gradient pixels (0..1).

    Guards against the rare case where a high-contrast but defocused pattern
    (e.g. a blurred halftone) still scores a non-trivial Laplacian variance.
    """
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    patch = _center_crop_patch(gray, patch_side)
    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    # Pixels with a gradient magnitude in the top decile are "edges".
    return float(np.mean(mag > np.percentile(mag, 90)))


class BlurDetector:
    """Score image sharpness against :class:`QualityThresholds`."""

    def __init__(self, thresholds: QualityThresholds | None = None):
        self.t = thresholds or QualityThresholds()

    def evaluate(self, bgr: np.ndarray) -> dict:
        """Return raw + normalized sharpness metrics for one BGR image."""
        lap_var = compute_laplacian_variance(bgr)
        edge_freq = compute_edge_frequency(bgr)
        # Normalize the Laplacian variance to 0..1 with a soft curve anchored
        # at the blur_low / blur_high thresholds.
        lo, hi = self.t.blur_low, self.t.blur_high
        if hi <= lo:
            norm = 1.0 if lap_var >= hi else 0.0
        else:
            norm = float(np.clip((lap_var - lo) / (hi - lo), 0.0, 1.0))
        return {
            "laplacian_variance": lap_var,
            "edge_frequency": edge_freq,
            "sharpness_norm": round(norm, 4),
            "is_blurry": lap_var < lo,
        }


def compute_blur(bgr: np.ndarray, thresholds: QualityThresholds | None = None) -> dict:
    return BlurDetector(thresholds).evaluate(bgr)
