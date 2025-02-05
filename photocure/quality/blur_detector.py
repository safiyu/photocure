"""Blur detection via OpenCV Laplacian variance.

The Laplacian variance of a grayscale patch is a classic focus metric: sharp images
have high variance, defocused images have near-zero variance. It also reads low
on smooth scenes, so a re-blur test confirms blur before an image is flagged.

To avoid *megapixel bias* (a 48MP frame always has more edge pixels than a
4MP one), the metric is computed on a center-cropped, fixed-size patch so the
value is comparable across resolutions.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import QualityThresholds


def _center_crop_patch(bgr: np.ndarray, side: int = 1024) -> np.ndarray:
    """Center-crop a square and shrink it to at most ``side`` pixels.

    Never upscales: interpolating a small crop up to ``side`` smooths it and
    lowers the Laplacian variance, which made sharp wide frames look blurry.
    """
    h, w = bgr.shape[:2]
    s = min(h, w)
    if s <= 0:
        return bgr
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    patch = bgr[y0:y0 + s, x0:x0 + s]
    if s > side:
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


def compute_blur_retained(bgr: np.ndarray, patch_side: int = 1024, sigma: float = 1.5) -> float:
    """Share of edge energy that survives an extra blur (re-blur test).

    A sharp image loses much of its edge energy when blurred again (about 0.2 to 0.45),
    an already blurry one barely changes (0.65 and up). Unlike Laplacian variance this
    does not depend on how much texture the scene has, so smooth skies, snow and
    flat graphics are not mistaken for blur.
    """
    if bgr is None or bgr.size == 0:
        return 1.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    p = _center_crop_patch(gray, patch_side).astype(np.float32)

    def energy(a: np.ndarray) -> float:
        gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
        return float((gx * gx + gy * gy).sum())

    base = energy(p)
    if base < 1e-6:
        return 1.0
    return energy(cv2.GaussianBlur(p, (0, 0), sigma)) / base


class BlurDetector:
    """Score image sharpness against :class:`QualityThresholds`."""

    def __init__(self, thresholds: QualityThresholds | None = None):
        self.t = thresholds or QualityThresholds()

    def evaluate(self, bgr: np.ndarray) -> dict:
        """Return raw + normalized sharpness metrics for one BGR image."""
        lap_var = compute_laplacian_variance(bgr)
        # Normalize the Laplacian variance to 0..1 with a soft curve anchored
        # at the blur_low / blur_high thresholds.
        lo, hi = self.t.blur_low, self.t.blur_high
        if hi <= lo:
            norm = 1.0 if lap_var >= hi else 0.0
        else:
            norm = float(np.clip((lap_var - lo) / (hi - lo), 0.0, 1.0))
        # Low Laplacian variance alone also fires on smooth but sharp scenes, so
        # only call it blurry when the re-blur test agrees (skipped when clearly sharp).
        blurry = lap_var < lo
        retained = None
        if blurry:
            retained = compute_blur_retained(bgr)
            blurry = retained >= float(getattr(self.t, "blur_retained_min", 0.58))
        return {
            "laplacian_variance": lap_var,
            "sharpness_norm": round(norm, 4),
            "is_blurry": bool(blurry),
            "blur_retained": None if retained is None else round(retained, 4),
        }


def compute_blur(bgr: np.ndarray, thresholds: QualityThresholds | None = None) -> dict:
    return BlurDetector(thresholds).evaluate(bgr)
