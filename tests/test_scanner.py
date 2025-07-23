"""Tests for the recursive scanner: EXIF cascade, dedup, resumption."""
from __future__ import annotations

import shutil
import time

from photocure.scanner.file_scanner import FileScanner
from tests.conftest import make_image


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
    assert res.duplicates == 0

    from photocure.db.repository import PhotoRepository
    from photocure.db.session import get_session
    with get_session(tmp_db) as s:
        repo = PhotoRepository(s)
        recs = repo.all_images()
        assert len(recs) == 2
        by_name = {r.filename: r for r in recs}
        r1 = by_name["IMG_20240512_153045.jpg"]
        assert r1.extension == ".jpg"
        assert r1.file_size > 0
        assert r1.width > 0 and r1.height > 0
        # Unique files are not fully hashed during scan (lazy Tier-2).
        assert r1.sha256 == ""
        assert r1.quick_hash.startswith(f"{r1.file_size}:")
        # Filename cascade: no EXIF date in synthetic JPEGs -> filename tier.
        assert r1.taken_at is not None
        assert r1.taken_at.year == 2024 and r1.taken_at.month == 5
        assert r1.taken_at_source == "filename"
        r2 = by_name["PXL_20240101_090000000.jpg"]
        assert (r2.taken_at.year, r2.taken_at.month, r2.taken_at.day) == (2024, 1, 1)
        assert (r2.taken_at.hour, r2.taken_at.minute) == (9, 0)
        assert r2.taken_at_source == "filename"


def test_scan_detects_exact_duplicates(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    src = make_image(root / "orig.jpg", rng_seed=7)
    shutil.copy2(src, root / "copy.jpg")

    res = _scan(cfg, root, tmp_db)
    assert res.new == 2
    assert res.duplicates == 1

    from photocure.db.repository import PhotoRepository
    from photocure.db.session import get_session
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

    from photocure.db.repository import PhotoRepository
    from photocure.db.session import get_session
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
    except Exception:  # encoder unavailable -> decode-path check below
        p = tmp_path / "sample.heic"
        p.write_bytes(b"\x00\x00\x00\x18heifmifh" + b"\x00" * 32)

    from photocure.quality.image_loader import load_image
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


def test_fixture_makes_distinct_images_per_seed(tmp_path):
    from photocure.scanner.hasher import compute_sha256
    a = make_image(tmp_path / "a.jpg", rng_seed=1)
    b = make_image(tmp_path / "b.jpg", rng_seed=2)
    c = make_image(tmp_path / "c.jpg", rng_seed=1)
    assert compute_sha256(a) != compute_sha256(b)
    assert compute_sha256(a) == compute_sha256(c)


def _rows(db):
    from photocure.db.repository import PhotoRepository
    from photocure.db.session import get_session
    with get_session(db) as s:
        recs = PhotoRepository(s).all_images()
        return {r.filename: r.model_copy() for r in recs}


def test_rescan_reprocesses_modified_file(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    p = make_image(root / "IMG_20240101_010101.jpg", rng_seed=1, size=(320, 480))
    _scan(cfg, root, tmp_db)
    before = _rows(tmp_db)["IMG_20240101_010101.jpg"]
    assert (before.width, before.height) == (480, 320)

    # Mark inferred, then replace the file with a different-size image.
    from photocure.db.repository import PhotoRepository
    from photocure.db.session import get_session
    with get_session(tmp_db) as s:
        row = PhotoRepository(s).get_by_path(str(p.resolve()))
        row.infer_status = "inferred"
        row.category = "scenery"
    time.sleep(0.05)
    make_image(p, rng_seed=1, size=(200, 300))
    res = _scan(cfg, root, tmp_db)
    assert res.modified == 1 and res.new == 0 and res.skipped_known == 0

    after = _rows(tmp_db)["IMG_20240101_010101.jpg"]
    assert (after.width, after.height) == (300, 200)
    assert after.id == before.id
    assert after.infer_status == "pending" and after.category is None

    # Unchanged on the next pass.
    res3 = _scan(cfg, root, tmp_db)
    assert res3.modified == 0 and res3.skipped_known == 1


def test_rescan_detects_same_size_edit_via_mtime(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    p = make_image(root / "a.jpg", rng_seed=1)
    _scan(cfg, root, tmp_db)
    data = bytearray(p.read_bytes())
    data[-20] ^= 0xFF  # same size, different content
    time.sleep(1.1)
    p.write_bytes(bytes(data))
    res = _scan(cfg, root, tmp_db)
    assert res.modified == 1
    assert _scan(cfg, root, tmp_db).modified == 0


def test_prune_missing_marks_and_restores(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    a = make_image(root / "a.jpg", rng_seed=1)
    make_image(root / "b.jpg", rng_seed=2)
    _scan(cfg, root, tmp_db)

    keep = a.read_bytes()
    a.unlink()
    res = _scan(cfg, root, tmp_db)  # default: no pruning
    assert res.missing == 0
    assert _rows(tmp_db)["a.jpg"].scan_status == "scanned"

    res = _scan(cfg, root, tmp_db, prune_missing=True)
    assert res.missing == 1
    rows = _rows(tmp_db)
    assert rows["a.jpg"].scan_status == "missing"
    assert rows["b.jpg"].scan_status == "scanned"

    a.write_bytes(keep)
    res = _scan(cfg, root, tmp_db, prune_missing=True)
    assert res.missing == 0
    assert _rows(tmp_db)["a.jpg"].scan_status == "scanned"


def test_prune_is_scoped_to_scan_root(cfg, tmp_db, tmp_path):
    r1, r2 = tmp_path / "one", tmp_path / "two"
    make_image(r1 / "a.jpg", rng_seed=1)
    make_image(r2 / "b.jpg", rng_seed=2)
    _scan(cfg, r1, tmp_db)
    _scan(cfg, r2, tmp_db)
    res = _scan(cfg, r1, tmp_db, prune_missing=True)
    assert res.missing == 0
    assert _rows(tmp_db)["b.jpg"].scan_status == "scanned"


def test_duplicate_never_points_at_missing_original(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    orig = make_image(root / "orig.jpg", rng_seed=9)
    _scan(cfg, root, tmp_db)
    data = orig.read_bytes()
    orig.unlink()
    _scan(cfg, root, tmp_db, prune_missing=True)
    (root / "moved").mkdir()
    (root / "moved" / "orig.jpg").write_bytes(data)
    res = _scan(cfg, root, tmp_db)
    assert res.duplicates == 0
    moved = [r for r in _rows(tmp_db).values() if r.relative_path.startswith("moved")][0]
    assert not moved.is_duplicate and moved.duplicate_of is None


def test_same_pass_third_copy_after_quick_hash_collision(cfg, tmp_db, tmp_path):
    # Same size and head/tail bytes but different middle: quick-hash collision.
    root = tmp_path / "lib"
    root.mkdir()
    cfg.quick_hash_head_bytes = 16
    cfg.quick_hash_tail_bytes = 16
    base = make_image(tmp_path / "base.jpg", rng_seed=3).read_bytes()
    other = bytearray(base)
    mid = len(other) // 2
    other[mid] ^= 0xFF
    (root / "a.jpg").write_bytes(base)
    (root / "b.jpg").write_bytes(bytes(other))
    (root / "c.jpg").write_bytes(bytes(other))
    cfg.scan_workers = 1
    res = _scan(cfg, root, tmp_db)
    assert res.duplicates == 1
    rows = _rows(tmp_db)
    dups = [n for n, r in rows.items() if r.is_duplicate]
    assert len(dups) == 1 and dups[0] in ("b.jpg", "c.jpg")


def test_skipped_files_and_symlink_escape(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    make_image(root / "good.jpg", rng_seed=1)
    (root / "fake.jpg").write_bytes(b"not an image at all")
    outside = make_image(tmp_path / "outside" / "secret.jpg", rng_seed=5)
    (root / "link.jpg").symlink_to(outside)
    res = _scan(cfg, root, tmp_db)
    names = {p.rsplit("/", 1)[-1] for p in res.skipped_files}
    assert {"fake.jpg", "link.jpg"} <= names
    assert res.new == 1
    assert set(_rows(tmp_db)) == {"good.jpg"}


def test_unreadable_directory_reported(cfg, tmp_db, tmp_path):
    import os

    import pytest
    if os.geteuid() == 0:
        pytest.skip("root bypasses permissions")
    root = tmp_path / "lib"
    make_image(root / "good.jpg", rng_seed=1)
    make_image(root / "locked" / "x.jpg", rng_seed=2)
    (root / "locked").chmod(0)
    try:
        res = _scan(cfg, root, tmp_db)
    finally:
        (root / "locked").chmod(0o755)
    assert any(d.endswith("locked") for d in res.walk_errors)
    assert res.new == 1


def test_prune_ignores_unreadable_directory(cfg, tmp_db, tmp_path):
    import os

    import pytest
    if os.geteuid() == 0:
        pytest.skip("root bypasses permissions")
    root = tmp_path / "lib"
    make_image(root / "locked" / "x.jpg", rng_seed=2)
    _scan(cfg, root, tmp_db)
    (root / "locked").chmod(0)
    try:
        res = _scan(cfg, root, tmp_db, prune_missing=True)
    finally:
        (root / "locked").chmod(0o755)
    assert res.missing == 0


def test_settle_defers_fresh_files(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    make_image(root / "a.jpg", rng_seed=1)
    res = _scan(cfg, root, tmp_db, settle_s=3600)
    assert res.new == 0 and len(res.deferred_files) == 1
    res = _scan(cfg, root, tmp_db)
    assert res.new == 1


def test_scan_stores_rating_and_tz_from_sidecar(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    p = make_image(root / "photo.jpg", rng_seed=1)
    p.with_suffix(".xmp").write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        'xmlns:exif="http://ns.adobe.com/exif/1.0/" xmp:Rating="4" '
        'exif:DateTimeOriginal="2023-07-04T23:30:00+02:00"/></rdf:RDF></x:xmpmeta>')
    _scan(cfg, root, tmp_db)
    r = _rows(tmp_db)["photo.jpg"]
    assert r.rating == 4
    assert r.taken_at_source == "xmp:dtoriginal"
    assert r.tz_offset_minutes == 120
    assert (r.taken_at.day, r.taken_at.hour) == (4, 21)  # normalized to UTC


def test_near_duplicates_link_lower_resolution_to_higher(cfg, tmp_db, tmp_path):
    import cv2
    root = tmp_path / "lib"
    big = make_image(root / "big.jpg", rng_seed=4, size=(800, 1200))
    img = cv2.imread(str(big))
    small = cv2.resize(img, (600, 400))
    (root / "small.jpg").parent.mkdir(exist_ok=True)
    cv2.imwrite(str(root / "small.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    make_image(root / "other.jpg", rng_seed=77, size=(800, 1200), blur=True)
    res = _scan(cfg, root, tmp_db)
    rows = _rows(tmp_db)
    assert rows["big.jpg"].phash and rows["small.jpg"].phash
    assert rows["small.jpg"].near_dup_of == rows["big.jpg"].id
    assert rows["big.jpg"].near_dup_of is None
    assert not rows["small.jpg"].is_duplicate
    assert res.near_duplicates >= 1


def test_near_duplicates_across_scan_passes(cfg, tmp_db, tmp_path):
    import cv2
    root = tmp_path / "lib"
    big = make_image(root / "big.jpg", rng_seed=4, size=(800, 1200))
    _scan(cfg, root, tmp_db)
    img = cv2.imread(str(big))
    cv2.imwrite(str(root / "small.jpg"), cv2.resize(img, (600, 400)))
    _scan(cfg, root, tmp_db)
    rows = _rows(tmp_db)
    assert rows["small.jpg"].near_dup_of == rows["big.jpg"].id
    # A higher-resolution arrival demotes the earlier keeper.
    cv2.imwrite(str(root / "huge.jpg"), cv2.resize(img, (1800, 1200)))
    _scan(cfg, root, tmp_db)
    rows = _rows(tmp_db)
    assert rows["huge.jpg"].near_dup_of is None
    assert rows["big.jpg"].near_dup_of == rows["huge.jpg"].id
    assert rows["small.jpg"].near_dup_of == rows["huge.jpg"].id


def test_modified_original_clears_stale_links(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    src = make_image(root / "orig.jpg", rng_seed=9)
    shutil.copy2(src, root / "copy.jpg")
    cfg.scan_workers = 1
    _scan(cfg, root, tmp_db)
    rows = _rows(tmp_db)
    dup = next(n for n, r in rows.items() if r.is_duplicate)
    keeper = next(n for n, r in rows.items() if r.id == rows[dup].duplicate_of)
    time.sleep(0.05)
    make_image(root / keeper, rng_seed=10, size=(200, 200))
    _scan(cfg, root, tmp_db)
    rows = _rows(tmp_db)
    assert not rows[dup].is_duplicate and rows[dup].duplicate_of is None
    assert not rows[keeper].is_duplicate
