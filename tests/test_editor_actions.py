"""Unit and integration tests for editor actions and endpoints."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from photocure.db import models as M
from photocure.db.session import get_session
from photocure.editor_actions import (
    extract_text,
    get_oriented_image_stream,
    remaster_image,
    remove_background,
    save_edited_image,
)

from .conftest import make_image
from .test_ui_api import env  # shared fixture


def _create_oriented_test_image(path: Path, orientation: int = 6):
    """Create a 100x60 test image with EXIF orientation tag."""
    # 100 wide x 60 high
    im = Image.new("RGB", (100, 60), color=(180, 50, 50))
    exif = im.getexif()
    exif[274] = orientation  # Orientation tag
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="JPEG", exif=exif)


def test_get_oriented_image_stream(env):
    # Create image with orientation 6 (90 CW rotation required to become upright)
    test_img = env.lib / "rot.jpg"
    _create_oriented_test_image(test_img, orientation=6)
    env.client.post("/api/scan", json={"path": str(env.lib)})

    with get_session(env.db) as s:
        rec = s.query(M.ImageRecord).filter_by(filename="rot.jpg").first()
        rec.orientation = 6
        rec.width = 100
        rec.height = 60
        s.commit()
        iid = rec.id

    data, mime = get_oriented_image_stream(env.db, iid)
    assert mime in ("image/jpeg", "image/png")
    # Decode returned bytes: orientation 6 should swap width and height so width=60, height=100
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None
    assert arr.shape[0] == 100  # height
    assert arr.shape[1] == 60   # width


def test_save_edited_image_copy(env):
    items = env.client.get("/api/images?limit=1").json()["items"]
    iid = items[0]["id"]

    with get_session(env.db) as s:
        orig = s.get(M.ImageRecord, iid)
        orig.burst_id = 42
        orig.is_favorite = True
        orig.category = "nature"
        s.commit()

    # Generate edited buffer: 120x120 green image
    buf = io.BytesIO()
    Image.new("RGB", (120, 120), (0, 255, 0)).save(buf, format="JPEG")
    edited_bytes = buf.getvalue()

    res1 = save_edited_image(env.db, env.app.state.cfg, iid, edited_bytes, mode="copy")
    assert res1["ok"] is True
    assert "_edited" in res1["filename"]
    assert Path(res1["path"]).exists()

    with get_session(env.db) as s:
        rec1 = s.get(M.ImageRecord, res1["image_id"])
        assert rec1 is not None
        assert rec1.id != iid
        assert rec1.burst_id == 42
        assert rec1.is_favorite is True
        assert rec1.category == "nature"
        assert rec1.width == 120
        assert rec1.height == 120
        assert rec1.orientation == 1

    # Second copy should avoid collision: _edited_2
    res2 = save_edited_image(env.db, env.app.state.cfg, iid, edited_bytes, mode="copy")
    assert res2["ok"] is True
    assert "_edited_2" in res2["filename"]
    assert Path(res2["path"]).exists()


def test_save_edited_image_overwrite(env):
    items = env.client.get("/api/images?limit=1").json()["items"]
    iid = items[0]["id"]

    with get_session(env.db) as s:
        orig = s.get(M.ImageRecord, iid)
        orig_path = Path(orig.path)
        old_size = orig.file_size
        old_sha = orig.sha256

    # Write red square as replacement
    buf = io.BytesIO()
    Image.new("RGB", (150, 150), (255, 0, 0)).save(buf, format="JPEG")
    edited_bytes = buf.getvalue()

    res = save_edited_image(env.db, env.app.state.cfg, iid, edited_bytes, mode="overwrite")
    assert res["ok"] is True
    assert res["image_id"] == iid
    assert res["path"] == str(orig_path)
    assert res["width"] == 150
    assert res["height"] == 150

    with get_session(env.db) as s:
        rec = s.get(M.ImageRecord, iid)
        assert rec.width == 150
        assert rec.height == 150
        assert rec.orientation == 1
        assert rec.sha256 != old_sha
        assert rec.file_size == len(edited_bytes)


def test_remove_background_fallback():
    # Create simple image: blue background with white centered rectangle
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (255, 0, 0)  # blue
    img[25:75, 25:75] = (255, 255, 255)  # white

    ok, buf = cv2.imencode(".png", img)
    assert ok
    out_bytes = remove_background(buf.tobytes())
    assert out_bytes is not None

    # Verify alpha channel exists
    out_arr = cv2.imdecode(np.frombuffer(out_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    assert out_arr is not None
    assert out_arr.shape[2] == 4  # RGBA


def test_extract_text():
    # Test blank image and image with text via RapidOCR
    buf = io.BytesIO()
    Image.new("RGB", (60, 30), (255, 255, 255)).save(buf, format="PNG")
    res = extract_text(buf.getvalue())
    assert isinstance(res, str)

    # Render synthetic text and verify RapidOCR extracts it
    canvas = np.full((120, 400, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "HELLO 2026", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    ok, enc = cv2.imencode(".png", canvas)
    assert ok
    txt = extract_text(enc.tobytes())
    assert "2026" in txt


def test_remaster_image_and_endpoint(env):
    # Create a dark, noisy 160x120 test image
    rng = np.random.default_rng(42)
    dark_img = rng.integers(15, 65, size=(120, 160, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".png", dark_img)
    assert ok

    out_bytes, w, h = remaster_image(enc.tobytes(), denoise=True, low_light=True, upscale=True)
    out_arr = cv2.imdecode(np.frombuffer(out_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert out_arr is not None
    # Small image (< 1600px) with upscale=True is 2x upscaled via ESPCN super-resolution
    assert (w, h) == (320, 240)
    assert out_arr.shape[0] == 240
    assert out_arr.shape[1] == 320
    # Low-light curve should brighten the dark image
    assert float(out_arr.mean()) > float(dark_img.mean())

    # Test POST /api/editor/remaster endpoint
    import base64
    b64_str = base64.b64encode(enc.tobytes()).decode("ascii")
    r = env.client.post(
        "/api/editor/remaster",
        json={"image": f"data:image/png;base64,{b64_str}", "denoise": True, "low_light": True, "upscale": False},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["image"].startswith("data:image/jpeg;base64,")
    assert data["width"] == 160
    assert data["height"] == 120


def test_api_full_and_edit_save(env):
    items = env.client.get("/api/images?limit=1").json()["items"]
    iid = items[0]["id"]

    # 1. GET /api/images/{id}/full
    r = env.client.get(f"/api/images/{iid}/full")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")

    # 2. POST /api/images/{id}/edit-save
    buf = io.BytesIO()
    Image.new("RGB", (80, 80), (100, 150, 200)).save(buf, format="JPEG")
    import base64
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

    save_r = env.client.post(
        f"/api/images/{iid}/edit-save",
        json={"image": f"data:image/jpeg;base64,{b64_str}", "mode": "copy", "format": "jpg"}
    )
    assert save_r.status_code == 200
    res_data = save_r.json()
    assert res_data["ok"] is True
    assert "_edited" in res_data["filename"]

    # 3. POST /api/editor/remove-bg
    bg_r = env.client.post(
        "/api/editor/remove-bg",
        json={"image": f"data:image/jpeg;base64,{b64_str}"}
    )
    assert bg_r.status_code == 200
    assert bg_r.json()["ok"] is True
    assert bg_r.json()["image"].startswith("data:image/png;base64,")

    # 4. POST /api/editor/ocr
    ocr_r = env.client.post(
        "/api/editor/ocr",
        json={"image": f"data:image/jpeg;base64,{b64_str}"}
    )
    assert ocr_r.status_code == 200
    assert "text" in ocr_r.json()


def test_png_alpha_preservation(env):
    # Create RGBA PNG with semi-transparent pixels
    png_path = env.lib / "transparent.png"
    im = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
    im.save(png_path, format="PNG")
    env.client.post("/api/scan", json={"path": str(env.lib)})

    with get_session(env.db) as s:
        rec = s.query(M.ImageRecord).filter_by(filename="transparent.png").first()
        iid = rec.id

    data, mime = get_oriented_image_stream(env.db, iid)
    assert mime == "image/png"
    loaded_im = Image.open(io.BytesIO(data))
    assert loaded_im.mode == "RGBA"
    # Ensure alpha is preserved (not flattened to 255)
    assert loaded_im.getpixel((10, 10))[3] == 128


def test_save_format_validation_and_raw_overwrite_guard(env):
    items = env.client.get("/api/images?limit=1").json()["items"]
    iid = items[0]["id"]

    buf = io.BytesIO()
    Image.new("RGB", (50, 50), (255, 255, 255)).save(buf, format="JPEG")
    b = buf.getvalue()

    # Reject unsupported format (e.g. pdf)
    from photocure.editor_actions import ActionError
    with pytest.raises(ActionError, match="Unsupported storage format"):
        save_edited_image(env.db, env.app.state.cfg, iid, b, mode="copy", storage_format="pdf")

    # Reject overwrite on RAW or HEIF
    with get_session(env.db) as s:
        rec = s.get(M.ImageRecord, iid)
        old_ext = rec.extension
        rec.extension = ".cr2"
        s.commit()

    with pytest.raises(ActionError, match="Cannot overwrite RAW or HEIF"):
        save_edited_image(env.db, env.app.state.cfg, iid, b, mode="overwrite")

    # Restore
    with get_session(env.db) as s:
        rec = s.get(M.ImageRecord, iid)
        rec.extension = old_ext
        s.commit()
