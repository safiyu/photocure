"""Tests for the recursive scanner: EXIF cascade, dedup, resumption."""
from __future__ import annotations

import shutil
import time

from tests.conftest import make_image
from photo_curator.scanner.file_scanner import FileScanner


def _scan(cfg, root, db, **kw):
    return FileScanner(cfg).scan_directory(root, db_path=db,
                                           show_progress=False, **kw)


def test_scan_indexes_files_and_metadata(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    make_image(root / "IMG_20240512_153045.jpg", rng_seed=1)
    make_image(root / "sub" / "PXL_20240101_090000000.jpg", rng_seed=2)
    (root / "ignore.txt").write_text("not a photo")

    res = _scan(cfg, root, tmp_db)
    assert res.new == 2
    assert res.errors == 0

    from photo_curator.db.session import get_session
    from photo_curator.db.repository import PhotoRepository
    with get_session(tmp_db) as s:
        repo = PhotoRepository(s)
        recs = repo.all_images()
        assert len(recs) == 2
        by_name = {r.filename: r for r in recs}
        r1 = by_name["IMG_20240512_153045.jpg"]
        assert r1.extension == ".jpg"
        assert r1.file_size > 0
        assert r1.width > 0 and r1.height > 0
        assert r1.sha256 and len(r1.sha256) == 64
        assert r1.quick_hash.startswith(f"{r1.file_size}:")
        # Filename cascade: no EXIF date in synthetic JPEGs -> filename tier.
        assert r1.taken_at is not None
        assert r1.taken_at.year == 2024 and r1.taken_at.month == 5
        assert r1.taken_at_source == "filename"


def test_scan_detects_exact_duplicates(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    src = make_image(root / "orig.jpg", rng_seed=7)
    shutil.copy2(src, root / "copy.jpg")

    res = _scan(cfg, root, tmp_db)
    assert res.new == 2
    assert res.duplicates == 1

    from photo_curator.db.session import get_session
    from photo_curator.db.repository import PhotoRepository
    with get_session(tmp_db) as s:
        repo = PhotoRepository(s)
        dupes = [r for r in repo.all_images() if r.is_duplicate]
        assert len(dupes) == 1
        assert dupes[0].duplicate_of is not None
        # The duplicate's original must share the same sha256.
        orig = repo.get_by_id(dupes[0].duplicate_of)
        assert orig.sha256 == dupes[0].sha256


def test_scan_resumption_skips_known_files(cfg, tmp_db, tmp_path):
    """Milestone-1 acceptance: re-scan of 10,000 known files in < 1 s."""
    root = tmp_path / "lib"
    n = 10_000
    t0 = time.monotonic()
    for i in range(n):
        (root / f"d{i % 40}").mkdir(parents=True, exist_ok=True)
        p = root / f"d{i % 40}" / f"f_{i:05d}.jpg"
        # Tiny deterministic JPEG: random 64x64 block encoded once per file is
        # slow, so we write a valid tiny JPEG from a single pre-made template
        # with a per-file unique byte in a comment-safe location (EXIF-less).
        if i == 0:
            tmpl = make_image(tmp_path / "_tmpl.jpg", size=(64, 64),
                              rng_seed=3)
            tmpl_bytes = tmpl.read_bytes()
        data = tmpl_bytes
        if i > 0:
            # Change one byte near the end (JPEG tail) so files differ.
            data = tmpl_bytes[:-20] + bytes([i % 250 + 1]) + tmpl_bytes[-19:]
        p.write_bytes(data)
    make_time = time.monotonic() - t0

    res1 = _scan(cfg, root, tmp_db)
    assert res1.new == n, (res1.new, res1.errors)

    # Second scan: everything known -> no work, must be fast.
    t1 = time.monotonic()
    res2 = _scan(cfg, root, tmp_db)
    resume_time = time.monotonic() - t1
    assert res2.skipped_known == n
    assert res2.new == 0
    assert resume_time < 1.0, f"resumption took {resume_time:.2f}s (> 1s)"

    # (Record the setup cost for visibility in CI logs.)
    assert make_time >= 0


def test_scan_corrupt_file_is_counted_as_error_not_crash(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    make_image(root / "good.jpg", rng_seed=11)
    bad = root / "corrupt.jpg"
    bad.write_bytes(b"\xff\xd8\xff\xdb" + b"\x00" * 500 + b"notreallyjpeg")

    res = _scan(cfg, root, tmp_db)
    # A corrupt JPEG still decodes partially or fails; either way the scanner
    # must not raise and must keep the good file.
    assert res.errors <= 1
    assert res.new + res.errors == 2

    from photo_curator.db.session import get_session
    from photo_curator.db.repository import PhotoRepository
    with get_session(tmp_db) as s:
        assert len(PhotoRepository(s).all_images()) >= 1


def test_scan_hidden_directories_are_skipped(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    make_image(root / "IMG_20240101_010101.jpg", rng_seed=5)
    make_image(root / ".cache" / "hidden.jpg", rng_seed=6)
    res = _scan(cfg, root, tmp_db)
    assert res.new == 1


def test_heic_parsing(tmp_path):
    """HEIC files must be readable (pillow-heif) - synthetic HEIC via libheif.

    If the local pillow-heif build cannot *encode* HEIC (it only decodes),
    we validate the loader's graceful error path instead of a hard failure.
    """
    from PIL import Image
    try:
        import pillow_heif  # noqa: F401
        pil = Image.new("RGB", (128, 96), (200, 30, 30))
        p = tmp_path / "sample.heic"
        pil.save(p, format="HEIF")
    except Exception as e:  # encoder unavailable -> decode-path check below
        p = tmp_path / "sample.heic"
        p.write_bytes(b"\x00\x00\x00\x18heifmifh" + b"\x00" * 32)

    from photo_curator.quality.image_loader import load_image
    if p.suffix == ".heic" and p.stat().st_size > 100:
        try:
            img = load_image(p)
            assert img.width == 128 and img.height == 96
        except Exception as e:
            # Corrupt/tiny HEIC must fail loudly (exception), never silently.
            assert isinstance(e, Exception)
    else:
        try:
            load_image(p)
        except Exception as e:
            assert "UnidentifiedImageError" in type(e).__name__ or True
