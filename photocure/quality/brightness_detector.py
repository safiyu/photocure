"""Brightness / exposure analysis.

Detects the three pathological exposure cases that dominate "junk" libraries:
- **pitch-black** frames (lens cap, closed lid): low mean luminance + high
  fraction of near-black pixels.
- **blown-out** frames: high mean luminance + high fraction of near-white
  pixels.
- **low-contrast** frames: very small RMS contrast (flat, featureless).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import QualityThresholds


def compute_luminance_metrics(bgr: np.ndarray,
                              thresholds: QualityThresholds | None = None,
                              black_thr: int = 12,
                              white_thr: int = 243) -> dict:
    """Compute luminance, RMS contrast and clipping statistics for a BGR image."""
    t = thresholds or QualityThresholds()
    if bgr is None or bgr.size == 0:
        return {
            "mean_luminance": 0.0, "rms_contrast": 0.0,
            "black_clip_pct": 1.0, "white_clip_pct": 0.0,
            "is_too_dark": True, "is_overexposed": False,
            "is_low_contrast": True,
        }

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    g = gray.astype(np.float32)

    mean = float(g.mean())
    # RMS contrast: std dev of the grayscale signal.
    rms = float(g.std())
    # Subsample the histogram for very large frames (fast percentile clip).
    if g.size > 2_000_000:
        step = max(1, int(np.sqrt(g.size / 2_000_000)))
        sample = g[::step]
    else:
        sample = g

    black_pct = float(np.mean(sample <= black_thr))
    white_pct = float(np.mean(sample >= white_thr))

    is_dark = (mean < t.brightness_min) or (black_pct >= t.black_clip_pct)
    is_over = (mean > t.brightness_max) or (white_pct >= t.white_clip_pct)
    # Flat / low-contrast: treat very small RMS as featureless.
    is_low_contrast = rms < float(getattr(t, "low_contrast_rms", 6.0))

    return {
        "mean_luminance": round(mean, 2),
        "rms_contrast": round(rms, 2),
        "black_clip_pct": round(black_pct, 4),
        "white_clip_pct": round(white_pct, 4),
        "is_too_dark": bool(is_dark),
        "is_overexposed": bool(is_over),
        "is_low_contrast": bool(is_low_contrast),
    }


class BrightnessDetector:
    """Wrap :func:`compute_luminance_metrics` as a stateful evaluator."""

    def __init__(self, thresholds: QualityThresholds | None = None,
                 black_thr: int = 12, white_thr: int = 243):
        self.t = thresholds or QualityThresholds()
        self.black_thr = black_thr
        self.white_thr = white_thr

    def evaluate(self, bgr: np.ndarray) -> dict:
        return compute_luminance_metrics(bgr, self.t, self.black_thr, self.white_thr)


def compute_brightness(bgr: np.ndarray, thresholds: QualityThresholds | None = None) -> dict:
    return BrightnessDetector(thresholds).evaluate(bgr)
