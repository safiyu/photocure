"""Composite quality scoring producing a single :class:`QualityMetric` row."""
from __future__ import annotations

import numpy as np

from ..config import QualityThresholds
from .blur_detector import BlurDetector
from .brightness_detector import BrightnessDetector


class QualityEngine:
    """Combines blur + brightness metrics into one composite quality score."""

    def __init__(self, thresholds: QualityThresholds | None = None):
        self.t = thresholds or QualityThresholds()
        self.blur = BlurDetector(self.t)
        self.bright = BrightnessDetector(self.t)

    def score(self, bgr: np.ndarray) -> dict:
        """Return a dict matching the :class:`QualityMetric` fields."""
        blur = self.blur.evaluate(bgr)
        bright = self.bright.evaluate(bgr)

        is_blurry = blur["is_blurry"]
        is_dark = bright["is_too_dark"]
        is_over = bright["is_overexposed"]
        low_contrast = bright["is_low_contrast"]

        # Composite 0..1 quality score: start from sharpness, penalize the
        # pathological exposure/contrast cases.
        score = blur["sharpness_norm"]
        if is_dark:
            score *= 0.25
        if is_over:
            score *= 0.40
        if low_contrast:
            score *= 0.60
        score = float(np.clip(score, 0.0, 1.0))

        # A single "needs review" verdict used by the decision matrix.
        needs_review = is_blurry or is_dark or is_over or low_contrast

        return {
            "laplacian_variance": blur["laplacian_variance"],
            "sharpness_norm": blur["sharpness_norm"],
            "mean_luminance": bright["mean_luminance"],
            "rms_contrast": bright["rms_contrast"],
            "black_clip_pct": bright["black_clip_pct"],
            "white_clip_pct": bright["white_clip_pct"],
            "is_blurry": bool(is_blurry),
            "is_too_dark": bool(is_dark),
            "is_overexposed": bool(is_over),
            "is_low_contrast": bool(low_contrast),
            "quality_score": round(score, 4),
            "needs_review": bool(needs_review),
        }
