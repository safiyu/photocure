"""Tests for image quality metrics (blur + brightness) on synthetic frames."""
from __future__ import annotations

import cv2
import numpy as np

from photo_curator.config import QualityThresholds
from photo_curator.quality.blur_detector import (
    BlurDetector,
    compute_blur,
    compute_edge_frequency,
    compute_laplacian_variance,
)
from photo_curator.quality.brightness_detector import (
    BrightnessDetector,
    compute_brightness,
    compute_luminance_metrics,
)
from photo_curator.quality.engine import QualityEngine


def _sharp_image(size=(400, 400), seed=0) -> np.ndarray:
    """High-contrast structured frame (edges everywhere)."""
    rng = np.random.default_rng(seed)
    h = w = size[0]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            v = int(255 * ((x * 7 + y * 13) % 97) / 97)
            img[y:y + 8, x:x + 8] = v
    return img


def _blur_image(size=(400, 400), seed=0) -> np.ndarray:
    return cv2.GaussianBlur(_sharp_image(size, seed), (31, 31), 17)


def test_laplacian_variance_sharp_beats_blur():
    sharp = _sharp_image()
    blurred = _blur_image()
    v_sharp = compute_laplacian_variance(sharp)
    v_blur = compute_laplacian_variance(blurred)
    assert v_sharp > 5 * v_blur, (v_sharp, v_blur)
    assert v_blur > 0


def test_blur_detector_flags_blurry_frame():
    t = QualityThresholds()
    d = BlurDetector(t)
    sharp_res = d.evaluate(_sharp_image(seed=1))
    blur_res = d.evaluate(_blur_image(seed=1))
    assert sharp_res["sharpness_norm"] > blur_res["sharpness_norm"]
    assert blur_res["is_blurry"] is True or blur_res["sharpness_norm"] < 0.3
    assert 0.0 <= sharp_res["sharpness_norm"] <= 1.0


def test_blur_is_resolution_independent():
    """The 400px and 1600px versions of the same scene must score similarly,
    because the metric is computed on a fixed center-crop patch."""
    small = _sharp_image(size=(400, 400), seed=4)
    big = cv2.resize(small, (1600, 1600), interpolation=cv2.INTER_CUBIC)
    v_small = compute_laplacian_variance(small)
    v_big = compute_laplacian_variance(big)
    # Same scene: variance should be within an order of magnitude.
    assert 0.1 * v_small < v_big < 10 * v_small


def test_edge_frequency_in_range():
    f = compute_edge_frequency(_sharp_image(seed=2))
    assert 0.0 <= f <= 1.0


def test_brightness_detects_dark_frame():
    black = np.zeros((300, 300, 3), dtype=np.uint8)
    m = compute_luminance_metrics(black)
    assert m["is_too_dark"] is True
    assert m["black_clip_pct"] > 0.95
    assert m["mean_luminance"] < 10


def test_brightness_detects_overexposed_frame():
    white = np.full((300, 300, 3), 252, dtype=np.uint8)
    m = compute_luminance_metrics(white)
    assert m["is_overexposed"] is True
    assert m["white_clip_pct"] > 0.9
    assert m["is_too_dark"] is False


def test_brightness_normal_frame_has_metrics():
    normal = _sharp_image(size=(300, 300), seed=9)
    m = compute_luminance_metrics(normal)
    assert 0.0 <= m["mean_luminance"] <= 255.0
    assert m["rms_contrast"] > 10.0  # structured frame is not flat
    assert 0.0 <= m["black_clip_pct"] <= 1.0
    assert 0.0 <= m["white_clip_pct"] <= 1.0
    assert m["is_too_dark"] is False
    assert m["is_overexposed"] is False


def test_empty_image_gives_safe_defaults():
    m = compute_luminance_metrics(np.zeros((0, 0, 3), dtype=np.uint8))
    assert m["is_too_dark"] is True
    assert m["rms_contrast"] == 0.0


def test_composite_engine_penalizes_bad_frames():
    eng = QualityEngine(QualityThresholds())
    good = eng.score(_sharp_image(seed=10))
    dark = eng.score(np.zeros((300, 300, 3), dtype=np.uint8))
    assert good["quality_score"] > dark["quality_score"]
    assert good["needs_review"] is False or good["quality_score"] > 0.3
    assert dark["is_too_dark"] is True
    assert dark["needs_review"] is True
    assert 0.0 <= good["quality_score"] <= 1.0
    assert 0.0 <= dark["quality_score"] <= 1.0
