import numpy as np

from photocure.quality.brightness_detector import compute_luminance_metrics


def _bgr(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray] * 3, axis=-1).astype(np.uint8)


def test_white_backdrop_with_dark_subject_is_not_overexposed():
    g = np.full((200, 300), 255, np.uint8)
    rng = np.random.default_rng(0)
    g[60:200, 80:220] = rng.integers(10, 140, (140, 140), dtype=np.uint8)
    m = compute_luminance_metrics(_bgr(g))
    assert m["white_clip_pct"] >= 0.5
    assert m["is_overexposed"] is False


def test_washed_out_frame_is_still_overexposed():
    g = np.full((200, 300), 255, np.uint8)
    rng = np.random.default_rng(1)
    g[60:200, 80:220] = rng.integers(190, 240, (140, 140), dtype=np.uint8)
    m = compute_luminance_metrics(_bgr(g))
    assert m["is_overexposed"] is True


def test_high_mean_is_still_overexposed():
    g = np.full((100, 100), 250, np.uint8)
    assert compute_luminance_metrics(_bgr(g))["is_overexposed"] is True
