"""Tests for image quality metrics (blur + brightness) on synthetic frames."""
from __future__ import annotations

import cv2
import numpy as np

from photocure.config import QualityThresholds
from photocure.quality.blur_detector import (
    BlurDetector,
    compute_laplacian_variance,
)
from photocure.quality.brightness_detector import (
    compute_luminance_metrics,
)
from photocure.quality.engine import QualityEngine


def _sharp_image(size=(400, 400), seed=0) -> np.ndarray:
    """High-contrast structured frame (edges everywhere)."""
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


def test_blur_patch_never_upscales_wide_frames():
    """A sharp panorama scores like its untouched center square (no upscale)."""
    from photocure.quality.blur_detector import _center_crop_patch
    sharp = _sharp_image(size=(400, 400))
    pano = np.concatenate([sharp, sharp, sharp], axis=1)   # 400 x 1200
    patch = _center_crop_patch(pano, 1024)
    assert patch.shape[:2] == (400, 400)
    center = pano[:, 400:800]
    assert compute_laplacian_variance(pano) == compute_laplacian_variance(center)


def test_blur_patch_downscales_big_frames_to_cap():
    from photocure.quality.blur_detector import _center_crop_patch
    big = np.zeros((2000, 3000, 3), dtype=np.uint8)
    assert _center_crop_patch(big, 1024).shape[:2] == (1024, 1024)


def test_sharp_wide_frame_is_not_flagged_blurry():
    t = QualityThresholds()
    pano = np.concatenate([_sharp_image(size=(400, 400))] * 3, axis=1)
    assert BlurDetector(t).evaluate(pano)["is_blurry"] is False


def test_blur_detector_result_has_no_constant_edge_metric():
    assert "edge_frequency" not in BlurDetector().evaluate(_sharp_image())


def test_quality_engine_reports_low_contrast_flag():
    flat = np.full((200, 200, 3), 128, dtype=np.uint8)
    q = QualityEngine().score(flat)
    assert q["is_low_contrast"] is True
    assert QualityEngine().score(_sharp_image())["is_low_contrast"] is False


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


def test_smooth_but_sharp_scene_is_not_blurry():
    """A smooth gradient with one crisp edge has low Laplacian variance but is sharp."""
    import cv2

    img = np.zeros((900, 1200, 3), np.uint8)
    for y in range(900):
        img[y, :] = (200 - y // 8, 160 - y // 12, 110)
    cv2.fillPoly(img, [np.array([[0, 600], [300, 330], [650, 560], [900, 300], [1200, 600]])], (95, 105, 80))
    res = BlurDetector().evaluate(img)
    assert res["laplacian_variance"] < 50
    assert res["blur_retained"] < 0.58
    assert res["is_blurry"] is False


def test_blurred_scene_is_still_blurry():
    import cv2

    img = np.zeros((900, 1200, 3), np.uint8)
    cv2.fillPoly(img, [np.array([[0, 600], [300, 330], [650, 560], [900, 300], [1200, 600]])], (250, 250, 250))
    res = BlurDetector().evaluate(cv2.GaussianBlur(img, (0, 0), 6))
    assert res["is_blurry"] is True and res["blur_retained"] >= 0.58
