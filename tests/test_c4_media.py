"""C4 acceptance: orientation, HEIC/RAW decoding, and the WebP thumbnail pyramid."""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pillow_heif
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from photocure import thumbs
from photocure.config import CuratorConfig
from photocure.db import models as M
from photocure.db.session import get_session, init_db
from photocure.quality.image_loader import apply_exif_orientation, load_image
from photocure.scanner.exif_reader import extract_exif

from .conftest import make_image

pillow_heif.register_heif_opener()
ROOT = Path(__file__).resolve().parent.parent


def asym(w=60, h=30) -> np.ndarray:
    """BGR image whose top-left corner is white and whose halves differ, so any flip shows."""
    a = np.zeros((h, w, 3), np.uint8)
    a[:, : w // 2] = (255, 0, 0)
    a[:, w // 2:] = (0, 0, 255)
    a[: h // 5, : w // 5] = 255
    return a


def white_corner(bgr: np.ndarray) -> str:
    h, w = bgr.shape[:2]
    k = max(1, min(h, w) // 8)
    patches = {"tl": bgr[:k, :k], "tr": bgr[:k, -k:], "bl": bgr[-k:, :k], "br": bgr[-k:, -k:]}
    return max(patches, key=lambda n: patches[n].mean())


# ---------------------------------------------------------------- orientation
@pytest.mark.parametrize("o", range(1, 9))
def test_array_and_pillow_paths_agree_with_exif_transpose(o):
    bgr = asym()
    rgb = Image.fromarray(bgr[:, :, ::-1])
    ref = rgb.copy()
    exif = Image.Exif()
    exif[274] = o
    ref.info["exif"] = exif.tobytes()
    expected = np.asarray(ImageOps.exif_transpose(ref))
    via_pil = np.asarray(apply_exif_orientation(rgb, o))
    via_array = apply_exif_orientation(bgr, o)[:, :, ::-1]
    assert expected.shape == via_pil.shape == via_array.shape
    assert np.array_equal(expected, via_pil) and np.array_equal(expected, via_array)


def test_known_orientation_moves_the_marker_where_the_camera_meant():
    bgr = asym()
    assert white_corner(apply_exif_orientation(bgr, 1)) == "tl"
    assert white_corner(apply_exif_orientation(bgr, 6)) == "tr"   # rotate 90 clockwise
    assert white_corner(apply_exif_orientation(bgr, 3)) == "br"
    assert white_corner(apply_exif_orientation(bgr, 8)) == "bl"
    assert apply_exif_orientation(bgr, 6).shape[:2] == (60, 30)


@pytest.mark.parametrize("bad", [None, 0, 9, -1, "x", 1.5 * 100])
def test_unknown_orientation_values_leave_the_image_alone(bad):
    bgr = asym()
    assert apply_exif_orientation(bgr, bad) is bgr


# --------------------------------------------------------------------- HEIC
def make_heic(path: Path, *, orientation=6, date="2024:05:01 12:30:45", offset="+02:00") -> Path:
    exif = Image.Exif()
    exif[274] = orientation
    exif[271], exif[272] = "TestCam", "Model X"
    ifd = exif.get_ifd(0x8769)
    ifd[36867] = date
    ifd[36881] = offset
    Image.fromarray(asym()[:, :, ::-1]).save(path, "HEIF", exif=exif.tobytes())
    return path


def test_heic_date_orientation_and_size_are_read_correctly(tmp_path):
    info = extract_exif(make_heic(tmp_path / "a.heic"))
    assert info.error == ""
    assert info.taken_at.isoformat() == "2024-05-01T12:30:45"
    assert info.taken_at_source == "exif:dtoriginal" and info.tz_offset_minutes == 120
    assert info.camera_model == "Model X"
    # pillow_heif rotates on open, so the stored frame is already portrait and the tag is spent.
    assert (info.width, info.height, info.orientation) == (30, 60, 1)


def test_heic_pixels_are_oriented_exactly_once(tmp_path):
    p = make_heic(tmp_path / "a.heic")
    img = load_image(p, orientation=extract_exif(p).orientation)
    assert img.bgr.shape[:2] == (60, 30) and white_corner(img.bgr) == "tr"


def test_heic_date_is_read_in_a_fresh_process_that_never_imported_the_loader(tmp_path):
    p = make_heic(tmp_path / "a.heic")
    code = ("from photocure.scanner.exif_reader import extract_exif;"
            f"i = extract_exif({str(p)!r}); print(i.taken_at_source, i.taken_at.isoformat())")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                         cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": ""})
    assert out.stdout.split()[:2] == ["exif:dtoriginal", "2024-05-01T12:30:45"], out.stderr[-300:]


# ----------------------------------------------------------------------- RAW
def _jpeg(bgr, orientation=None) -> bytes:
    buf = io.BytesIO()
    kw = {}
    if orientation:
        exif = Image.Exif()
        exif[274] = orientation
        kw["exif"] = exif.tobytes()
    Image.fromarray(bgr[:, :, ::-1]).save(buf, "JPEG", quality=95, **kw)
    return buf.getvalue()


class FakeRaw:
    def __init__(self, thumb=None, thumb_error=None, develop=None):
        self.thumb, self.thumb_error, self.develop, self.calls = thumb, thumb_error, develop, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_thumb(self):
        if self.thumb_error:
            raise self.thumb_error
        return self.thumb

    def postprocess(self, **kw):
        self.calls.append(kw)
        return self.develop if self.develop is not None else np.full((50, 80, 3), 90, np.uint8)


def fake_rawpy(monkeypatch, raw=None, imread_error=None):
    def imread(path):
        if imread_error:
            raise imread_error
        return raw
    monkeypatch.setattr(thumbs, "rawpy", SimpleNamespace(
        imread=imread, ThumbFormat=SimpleNamespace(JPEG=1, BITMAP=2)))


def test_raw_uses_the_embedded_jpeg_preview_and_the_files_orientation(monkeypatch):
    raw = FakeRaw(SimpleNamespace(format=1, data=_jpeg(asym(600, 300))))  # preview has no tag
    fake_rawpy(monkeypatch, raw)
    out = thumbs.raw_preview("x.cr2", 220, orientation=6)
    assert out.shape[:2] == (600, 300) and white_corner(out) == "tr"
    assert raw.calls == []  # never developed


def test_a_preview_that_carries_its_own_orientation_is_not_rotated_twice(monkeypatch):
    raw = FakeRaw(SimpleNamespace(format=1, data=_jpeg(asym(600, 300), orientation=1)))
    fake_rawpy(monkeypatch, raw)
    out = thumbs.raw_preview("x.cr2", 220, orientation=6)  # file says 6, preview says "as is"
    assert out.shape[:2] == (300, 600) and white_corner(out) == "tl"


def test_raw_bitmap_preview_is_wrapped_via_pil(monkeypatch):
    rgb = asym(400, 300)[:, :, ::-1].copy()
    fake_rawpy(monkeypatch, FakeRaw(SimpleNamespace(format=2, data=rgb)))
    out = thumbs.raw_preview("x.nef", 220, orientation=1)
    assert out.shape[:2] == (300, 400) and white_corner(out) == "tl"


def test_a_preview_smaller_than_the_tier_falls_back_to_a_half_size_develop(monkeypatch):
    raw = FakeRaw(SimpleNamespace(format=1, data=_jpeg(asym(160, 100))))
    fake_rawpy(monkeypatch, raw)
    out = thumbs.raw_preview("x.arw", 800, orientation=6)
    assert raw.calls and raw.calls[0]["half_size"] is True
    assert out.shape[:2] == (50, 80)  # LibRaw already applied the camera flip; not rotated again


def test_no_embedded_preview_falls_back_to_a_develop(monkeypatch):
    raw = FakeRaw(thumb_error=RuntimeError("no thumbnail"))
    fake_rawpy(monkeypatch, raw)
    assert thumbs.raw_preview("x.dng", 220).shape[:2] == (50, 80) and raw.calls


def test_unreadable_raw_gives_none_so_the_generic_loader_can_try(monkeypatch):
    fake_rawpy(monkeypatch, imread_error=OSError("corrupt"))
    assert thumbs.raw_preview("x.cr2", 220) is None


def test_render_tier_for_raw_encodes_webp_within_bounds(monkeypatch, tmp_path):
    fake_rawpy(monkeypatch, FakeRaw(SimpleNamespace(format=1, data=_jpeg(asym(1200, 800)))))
    data = thumbs.render_tier(tmp_path / "x.cr2", ".CR2", 1, "micro")
    with Image.open(io.BytesIO(data)) as im:
        assert im.format == "WEBP" and max(im.size) == 220


# ------------------------------------------------------------- WebP pyramid
@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_CACHE_DIR", str(tmp_path / "cache"))
    from photocure.scanner.file_scanner import FileScanner
    from photocure.ui.app import create_app
    db = tmp_path / "t.db"
    cfg = CuratorConfig()
    cfg.db_path = db
    init_db(db, drop=True)
    lib = tmp_path / "lib"
    make_image(lib / "IMG_20240501_100000.jpg", size=(3000, 2000), rng_seed=1)   # tall (h, w), large
    make_image(lib / "IMG_20240501_100001.jpg", size=(80, 60), rng_seed=2)       # smaller than every tier
    make_heic(lib / "IMG_20240501_100002.heic")
    FileScanner(cfg).scan_directory(lib, db_path=db, show_progress=False)
    app = create_app(str(db))
    client = TestClient(app, headers={"X-PCUR-Token": app.state.token})
    with get_session(db) as s:
        from sqlmodel import select
        ids = {r.filename: r.id for r in s.exec(select(M.ImageRecord)).all()}
    return SimpleNamespace(client=client, app=app, anon=TestClient(app), ids=ids, cache=tmp_path / "cache",
                           big=ids["IMG_20240501_100000.jpg"], small=ids["IMG_20240501_100001.jpg"],
                           heic=ids["IMG_20240501_100002.heic"])


@pytest.mark.parametrize("tier,bound", [("micro", 220), ("medium", 800), ("preview", 1600)])
def test_each_tier_is_webp_within_its_bound(api, tier, bound):
    r = api.client.get(f"/thumbnail/{api.big}", params={"tier": tier})
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    assert r.headers["cache-control"] == "public, max-age=31536000, immutable"
    with Image.open(io.BytesIO(r.content)) as im:
        assert im.format == "WEBP" and max(im.size) == bound
        assert im.size[1] > im.size[0]  # still portrait: 2000x3000 source


def test_small_photos_are_never_upscaled(api):
    for tier in ("micro", "medium", "preview"):
        with Image.open(io.BytesIO(api.client.get(f"/thumbnail/{api.small}?tier={tier}").content)) as im:
            assert im.size == (60, 80)


def test_tier_defaults_to_micro_and_unknown_tiers_are_refused(api):
    with Image.open(io.BytesIO(api.client.get(f"/thumbnail/{api.big}").content)) as im:
        assert max(im.size) == 220
    assert api.client.get(f"/thumbnail/{api.big}?tier=huge").status_code == 422


def test_cache_layout_hits_return_identical_bytes_and_leave_no_tmp(api):
    first = api.client.get(f"/thumbnail/{api.big}?tier=medium").content
    second = api.client.get(f"/thumbnail/{api.big}?tier=medium").content
    assert first == second
    files = list((api.cache / "thumbs" / "medium").glob("*"))
    assert len(files) == 1 and files[0].suffix == ".webp"
    assert files[0].name.split("_")[1] == str(api.big) and len(files[0].stem.split("_")[2]) == 16
    assert not list((api.cache / "thumbs").rglob("*.tmp"))
    assert not (api.cache / "thumbs" / "micro").exists()  # tiers are cached independently


@pytest.mark.parametrize("v", ["../../etc/passwd", "zzzzzzzzzzzz", "abc", "a" * 65, "0123456789ABCDEF"])
def test_malformed_cache_keys_are_rejected_before_any_work(api, v):
    assert api.client.get(f"/thumbnail/{api.big}", params={"v": v}).status_code == 400
    assert not (api.cache / "thumbs").exists()


def test_cache_key_is_a_buster_only_and_never_part_of_the_path(api):
    a = api.client.get(f"/thumbnail/{api.big}?v=0123456789abcdef")
    b = api.client.get(f"/thumbnail/{api.big}")
    assert a.status_code == b.status_code == 200 and a.content == b.content
    assert len(list((api.cache / "thumbs" / "micro").glob("*.webp"))) == 1


def test_thumbnails_need_auth_and_unknown_images_404(api):
    assert api.anon.get(f"/thumbnail/{api.big}").status_code == 401
    assert api.client.get("/thumbnail/99999").status_code == 404


def test_heic_thumbnail_is_oriented_and_webp(api):
    with Image.open(io.BytesIO(api.client.get(f"/thumbnail/{api.heic}?tier=medium").content)) as im:
        assert im.format == "WEBP" and im.size == (30, 60)  # portrait, not rotated a second time


def test_items_carry_the_fingerprint_the_client_puts_in_v(api):
    item = api.client.get("/api/images").json()["items"][0]
    assert len(item["fp"]) == 16
    assert api.client.get(f"/thumbnail/{item['id']}", params={"v": item["fp"]}).status_code == 200
