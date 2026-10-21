"""Open-eye composite engine: alignment, blending limits, and the two output files."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from photocure.composite import (
    FaceRef,
    compose_open_eyes,
    fixed_jpeg_path,
    layered_psd_path,
    write_fixed_jpeg,
    write_layered_psd,
)

W, H = 900, 700
FACE = (450, 330)  # face centre in the base frame
R = 60


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = rng.random((35, 45, 3)).astype(np.float32)
    img = cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC) * 200
    img += rng.random((H, W, 3)).astype(np.float32) * 40
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(80):
        p1 = tuple(int(v) for v in rng.integers(0, [W, H]))
        p2 = tuple(int(v) for v in rng.integers(0, [W, H]))
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, p1, p2, col, 2)
        else:
            cv2.circle(img, p1, int(rng.integers(5, 40)), col, 2)
    return img


def _face(img: np.ndarray, centre: tuple[int, int], open_eyes: bool) -> None:
    cx, cy = centre
    cv2.circle(img, (cx, cy), R, (120, 160, 210), -1)
    for dx in (-22, 22):
        if open_eyes:
            cv2.circle(img, (cx + dx, cy - 12), 9, (255, 255, 255), -1)
            cv2.circle(img, (cx + dx, cy - 12), 4, (20, 20, 20), -1)
        else:
            cv2.line(img, (cx + dx - 9, cy - 12), (cx + dx + 9, cy - 12), (20, 20, 20), 3)
    cv2.line(img, (cx - 20, cy + 25), (cx + 20, cy + 25), (40, 40, 160), 3)


def _frac(centre, half=R + 10):
    cx, cy = centre
    return ((cx - half) / W, (cy - half) / H, (cx + half) / W, (cy + half) / H)


@pytest.fixture(scope="module")
def pair():
    tex = _texture(1)
    base = tex.copy()
    _face(base, FACE, open_eyes=False)
    open_pre = tex.copy()
    _face(open_pre, FACE, open_eyes=True)
    m = cv2.getRotationMatrix2D((W / 2, H / 2), 2.0, 1.0)
    m[:, 2] += (12, -8)
    donor = cv2.warpAffine(open_pre, m, (W, H), borderMode=cv2.BORDER_REFLECT)
    dc = m @ np.array([FACE[0], FACE[1], 1.0])
    return base, donor, (int(dc[0]), int(dc[1])), m


def test_compose_changes_only_the_face_area(pair):
    base, donor, dc, _ = pair
    res = compose_open_eyes(base, donor, [FaceRef(_frac(FACE), "closed")],
                            [FaceRef(_frac(dc), "open")])
    assert res.ok and res.applied == 1 and res.reason == "" and res.inliers >= 25
    assert res.bgr.shape == base.shape and res.aligned_donor.shape == base.shape
    far = np.ones((H, W), bool)
    far[FACE[1] - 140:FACE[1] + 140, FACE[0] - 140:FACE[0] + 140] = False
    assert np.abs(res.bgr.astype(int) - base.astype(int))[far].max() == 0
    eye = (slice(FACE[1] - 20, FACE[1] - 4), slice(FACE[0] - 30, FACE[0] - 14))
    assert np.abs(res.bgr[eye].astype(int) - base[eye].astype(int)).mean() > 15
    assert res.mask.dtype == np.uint8 and res.mask[FACE[1], FACE[0]] >= 250
    assert res.mask[20, 20] == 0 and res.mask[H - 20, W - 20] == 0


def test_alignment_matches_true_motion(pair):
    base, donor, dc, _ = pair
    res = compose_open_eyes(base, donor, [FaceRef(_frac(FACE), "closed")],
                            [FaceRef(_frac(dc), "open")])
    # Away from the face, the aligned donor should sit on top of the base.
    patch = (slice(80, 200), slice(80, 200))
    aligned = np.abs(res.aligned_donor[patch].astype(int) - base[patch].astype(int)).mean()
    raw = np.abs(donor[patch].astype(int) - base[patch].astype(int)).mean()
    assert aligned < 0.4 * raw


def test_unrelated_frames_do_not_align(pair):
    base, _, _, _ = pair
    other = _texture(99)
    res = compose_open_eyes(base, other, [FaceRef(_frac(FACE), "closed")],
                            [FaceRef(_frac(FACE), "open")])
    assert not res.ok and "aligned" in res.reason and res.bgr is None and res.applied == 0


def test_no_open_donor_face_is_skipped(pair):
    base, donor, dc, _ = pair
    res = compose_open_eyes(base, donor, [FaceRef(_frac(FACE), "closed")],
                            [FaceRef(_frac(dc), "closed")])
    assert not res.ok and res.applied == 0
    assert res.skipped and "open eyes" in res.skipped[0]


def test_open_base_faces_are_left_alone(pair):
    base, donor, dc, _ = pair
    res = compose_open_eyes(base, donor, [FaceRef(_frac(FACE), "open")],
                            [FaceRef(_frac(dc), "open")])
    assert not res.ok and res.applied == 0 and res.skipped == []


def test_head_moved_too_much_is_skipped():
    tex = _texture(5)
    base = tex.copy()
    _face(base, FACE, open_eyes=False)
    moved = tex.copy()
    shifted = (FACE[0] + 75, FACE[1] + 20)
    _face(moved, shifted, open_eyes=True)
    res = compose_open_eyes(base, moved, [FaceRef(_frac(FACE), "closed")],
                            [FaceRef(_frac(shifted), "open")])
    assert not res.ok and res.applied == 0
    assert any("head moved" in s for s in res.skipped)


def test_fixed_jpeg_path_never_collides(tmp_path):
    src = tmp_path / "IMG_1042.jpg"
    src.write_bytes(b"x")
    assert fixed_jpeg_path(src).name == "IMG_1042_fixed.jpg"
    (tmp_path / "IMG_1042_fixed.jpg").write_bytes(b"x")
    assert fixed_jpeg_path(src).name == "IMG_1042_fixed2.jpg"
    assert layered_psd_path(src).name == "IMG_1042_layers.psd"


def test_write_fixed_jpeg_copies_exif_and_never_overwrites(tmp_path):
    from PIL import Image
    src = tmp_path / "IMG_1.jpg"
    im = Image.new("RGB", (64, 48), (10, 20, 30))
    exif = Image.Exif()
    exif[0x010F] = "TestCam"
    exif[0x0112] = 6
    im.save(src, exif=exif.tobytes())
    out = fixed_jpeg_path(src)
    bgr = np.full((48, 64, 3), 200, np.uint8)
    assert write_fixed_jpeg(bgr, src, out) == out
    with Image.open(out) as got:
        assert got.size == (64, 48)
        assert got.getexif()[0x0112] == 1 and got.getexif()[0x010F] == "TestCam"
    with pytest.raises(FileExistsError):
        write_fixed_jpeg(bgr, src, out)
    assert not list(tmp_path.glob("*.tmp"))


def test_write_fixed_jpeg_without_source_exif(tmp_path):
    from PIL import Image
    src = tmp_path / "raw.cr2"
    src.write_bytes(b"not an image")
    out = tmp_path / "raw_fixed.jpg"
    write_fixed_jpeg(np.zeros((20, 30, 3), np.uint8), src, out)
    with Image.open(out) as got:
        assert got.size == (30, 20)


def test_layered_psd_is_a_real_layered_file(tmp_path):
    psd_tools = pytest.importorskip("psd_tools")
    from PIL import Image
    rng = np.random.default_rng(3)
    h, w = 60, 80
    base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    donor = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    merged = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), np.uint8)
    mask[10:30, 20:50] = 255
    mask[20:30, 20:30] = 128
    path = tmp_path / "a_layers.psd"
    assert write_layered_psd(path, base, donor, mask, merged) == path
    psd = psd_tools.PSDImage.open(path)
    assert psd.size == (w, h)
    layers = list(psd)
    assert [x.name for x in layers] == ["Best frame", "Open eyes (aligned)"]
    assert np.array_equal(np.array(layers[0].topil().convert("RGB")), base[:, :, ::-1])
    assert np.array_equal(np.array(layers[1].topil().convert("RGB")), donor[:, :, ::-1])
    assert layers[0].mask is None and layers[1].mask is not None
    got = np.array(layers[1].mask.topil())
    assert got.shape == (h, w) and np.array_equal(got, mask)
    assert np.array_equal(np.array(psd.topil().convert("RGB")), merged[:, :, ::-1])
    with Image.open(path) as pil:
        assert pil.size == (w, h)
    with pytest.raises(FileExistsError):
        write_layered_psd(path, base, donor, mask)
    assert not list(tmp_path.glob("*.tmp"))


def test_layered_psd_defaults_merged_to_base_and_validates_sizes(tmp_path):
    psd_tools = pytest.importorskip("psd_tools")
    base = np.full((10, 12, 3), 50, np.uint8)
    path = write_layered_psd(tmp_path / "b.psd", base, base, np.zeros((10, 12), np.uint8))
    merged = np.array(psd_tools.PSDImage.open(path).topil().convert("RGB"))
    assert np.array_equal(merged, base[:, :, ::-1])
    with pytest.raises(ValueError):
        write_layered_psd(tmp_path / "c.psd", base, base[:5], np.zeros((10, 12), np.uint8))
    with pytest.raises(ValueError):
        write_layered_psd(tmp_path / "d.psd", base, base, np.zeros((5, 5), np.uint8))
