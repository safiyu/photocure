"""Perceptual hashing and the near-duplicate index."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from photocure.scanner.phash import (
    NearDupIndex,
    compute_dhash,
    compute_phash,
    from_hex,
    hamming,
    to_hex,
)
from tests.conftest import make_image


def _photo(path, seed=1, size=(600, 800)):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    img = cv2.resize(small, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def test_hash_format_and_determinism(tmp_path):
    p = _photo(tmp_path / "a.jpg")
    h = compute_phash(p)
    assert len(h) == 16 and int(h, 16) >= 0
    assert compute_phash(p) == h
    d = compute_dhash(p)
    assert len(d) == 16 and compute_dhash(p) == d


def test_resized_and_recompressed_copies_are_close(tmp_path):
    p = _photo(tmp_path / "a.jpg")
    img = cv2.imread(str(p))
    cv2.imwrite(str(tmp_path / "small.jpg"), cv2.resize(img, (300, 400)),
                [cv2.IMWRITE_JPEG_QUALITY, 60])
    assert hamming(compute_phash(p), compute_phash(tmp_path / "small.jpg")) <= 6
    assert hamming(compute_dhash(p), compute_dhash(tmp_path / "small.jpg")) <= 10


def test_different_images_are_far(tmp_path):
    a = _photo(tmp_path / "a.jpg", seed=1)
    b = _photo(tmp_path / "b.jpg", seed=2)
    assert hamming(compute_phash(a), compute_phash(b)) > 10


def test_exif_orientation_is_applied(tmp_path):
    p = _photo(tmp_path / "a.jpg", size=(400, 600))
    rotated = tmp_path / "rot.jpg"
    Image.open(p).transpose(Image.Transpose.ROTATE_90).save(rotated)
    # Same pixels stored rotated plus an EXIF tag that undoes the rotation.
    ex = Image.Exif()
    ex[274] = 6  # rotate 270 CW to display upright
    Image.open(p).transpose(Image.Transpose.ROTATE_90).save(rotated, exif=ex)
    assert hamming(compute_phash(p), compute_phash(rotated)) <= 6


def test_undecodable_file_returns_none(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    assert compute_phash(bad) is None and compute_dhash(bad) is None


def test_hamming_helpers():
    assert hamming(0b1011, 0b0001) == 2
    assert hamming(to_hex(255), to_hex(0)) == 8
    assert from_hex(to_hex(2**63 + 5)) == 2**63 + 5


def test_index_links_lower_resolution_to_higher():
    idx = NearDupIndex()
    assert idx.add(1, to_hex(0xF0F0F0F0F0F0F0F0), 1000, 10, "q1") == []
    # Differs by 3 bits, lower resolution.
    ups = idx.add(2, to_hex(0xF0F0F0F0F0F0F0F0 ^ 0b10101), 500, 5, "q2")
    assert ups == [(2, 1)]
    assert idx.near_dup_of(2) == 1 and idx.near_dup_of(1) is None


def test_index_higher_resolution_arrival_demotes_group():
    idx = NearDupIndex()
    base = 0x0123456789ABCDEF
    idx.add(1, to_hex(base), 400, 4, "a")
    idx.add(2, to_hex(base ^ 1), 100, 1, "b")
    ups = idx.add(3, to_hex(base ^ 2), 900, 9, "c")
    assert set(ups) == {(1, 3), (2, 3)}
    assert idx.near_dup_of(1) == 3 and idx.near_dup_of(2) == 3
    assert idx.near_dup_of(3) is None


def test_index_threshold_boundary_and_far_hashes():
    idx = NearDupIndex(threshold=6)
    base = 0
    idx.add(1, to_hex(base), 100, 1, "a")
    assert idx.add(2, to_hex(0b1111111 << 20), 50, 1, "b") == []  # 7 bits apart
    assert idx.add(3, to_hex(0b111111), 50, 1, "c") == [(3, 1)]  # 6 bits apart


def test_index_finds_matches_in_every_band():
    idx = NearDupIndex(threshold=6)
    base = 0xDEADBEEFCAFEF00D
    idx.add(1, to_hex(base), 100, 1, "a")
    for i, bit_positions in enumerate([(0, 9, 18, 27, 36, 45), (63, 54, 45, 36, 27, 18),
                                       (1, 2, 3, 4, 5, 6)], start=2):
        flip = sum(1 << b for b in bit_positions)
        assert idx.add(i, to_hex(base ^ flip), 50, 1, f"q{i}") == [(i, 1)]


def test_index_ignores_identical_files():
    idx = NearDupIndex()
    idx.add(1, to_hex(5), 100, 1, "same-q")
    assert idx.add(2, to_hex(5), 50, 1, "same-q") == []
    other = NearDupIndex()
    other.add(3, to_hex(5), 100, 1, "qa", "sha-a")
    assert other.add(4, to_hex(5), 50, 1, "qb", "sha-a") == []


def test_index_seed_and_resolve_chain():
    idx = NearDupIndex()
    idx.seed([(1, to_hex(0), 900, 9, "a", "", None), (2, to_hex(1), 100, 1, "b", "", 1)])
    ups = idx.add(3, to_hex(2), 50, 1, "c")
    assert ups == [(3, 1)]
    assert len(idx) == 3 and 1 in idx


def test_make_image_fixture_hashes_decode(tmp_path):
    assert compute_phash(make_image(tmp_path / "x.jpg", rng_seed=3)) is not None
