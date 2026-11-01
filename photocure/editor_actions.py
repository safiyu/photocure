"""Single-photo editing, background removal, OCR, and burst collage saving."""
from __future__ import annotations

import hashlib
import io
import logging
import os
import uuid
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from sqlmodel import select

from .config import CuratorConfig
from .db import models as M
from .db.session import get_session
from .quality.image_loader import HEIF_EXTS, RAW_EXTS, load_image
from .scanner.hasher import compute_quick_hash, compute_sha256

log = logging.getLogger("photocure.editor")


class ActionError(ValueError):
    """A refusal with a message fit to show the user and the HTTP status the web app should use."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _existing(*paths) -> str | None:
    return next((p for p in paths if p and os.path.exists(p)), None)


def _fingerprint(rec: M.ImageRecord) -> str:
    return hashlib.sha256(f"{rec.quick_hash}|{rec.sha256}|{rec.file_size}|{rec.filename}".encode()).hexdigest()[:16]


def get_oriented_image_stream(db_path: Path | str, image_id: int) -> tuple[bytes, str]:
    """Retrieve upright, orientation-normalized image bytes and MIME type.
    
    Loads any supported photo format (JPEG, PNG, WebP, HEIC/HEIF, RAW) through load_image,
    ensuring pixels are delivered standard upright (orientation=1).
    """
    with get_session(db_path) as s:
        rec = s.get(M.ImageRecord, image_id)
        if rec is None:
            raise ActionError(f"no photo with id {image_id}", 404)
        source_path = _existing(rec.path, rec.target_path)
        if not source_path:
            raise ActionError(f"file is missing on disk: {rec.path}", 404)
        orientation = rec.orientation or 1
        ext = (rec.extension or Path(source_path).suffix).lower()

    loaded = load_image(source_path, max_side=None, orientation=orientation)
    bgr = loaded.bgr

    if ext == ".png":
        mime_type = "image/png"
        ok, buf = cv2.imencode(".png", bgr)
    elif ext == ".webp":
        mime_type = "image/webp"
        ok, buf = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, 95])
    else:
        mime_type = "image/jpeg"
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    if not ok:
        raise RuntimeError("Image encode failed")
    return buf.tobytes(), mime_type


def save_edited_image(db_path: Path | str, cfg: CuratorConfig, image_id: int, image_bytes: bytes,
                      mode: Literal["copy", "overwrite"] = "copy",
                      format: str | None = None) -> dict:
    """Save an edited photo buffer either as a new copy or overwriting the original.
    
    In 'copy' mode:
      - Generates collision-free {stem}_edited{ext} filename in the same directory.
      - Writes atomically to disk.
      - Inserts ImageRecord inheriting metadata and burst_id from original.
      - Pre-renders thumbnail cache entries.
    In 'overwrite' mode:
      - Overwrites existing file atomically.
      - Updates file size, mtime, hashes, dimensions, and resets orientation=1.
      - Purges stale thumbnails so client refreshes immediately.
    """
    if mode not in ("copy", "overwrite"):
        raise ActionError("mode must be 'copy' or 'overwrite'", 400)

    db_fp = hashlib.sha1(str(Path(db_path).resolve()).encode()).hexdigest()[:8]
    head_bytes = getattr(cfg, "quick_hash_head_bytes", 64 * 1024)
    tail_bytes = getattr(cfg, "quick_hash_tail_bytes", 64 * 1024)

    # Determine image dimensions from bytes
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            w, h = pil_img.size
    except Exception:
        arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ActionError("Could not decode image bytes", 400)
        h, w = arr.shape[:2]

    with get_session(db_path) as s:
        rec = s.get(M.ImageRecord, image_id)
        if rec is None:
            raise ActionError(f"no photo with id {image_id}", 404)
        orig_str = _existing(rec.path, rec.target_path)
        if not orig_str:
            raise ActionError(f"original file is missing on disk: {rec.path}", 404)
        orig_path = Path(orig_str)

        # Output extension determination
        if format:
            out_ext = f".{format.lower().lstrip('.')}"
            if out_ext == ".jpeg":
                out_ext = ".jpg"
        else:
            out_ext = orig_path.suffix.lower()
            if out_ext in RAW_EXTS or out_ext in HEIF_EXTS:
                out_ext = ".jpg"

        if mode == "copy":
            stem = orig_path.stem
            target_dir = orig_path.parent
            cand_name = f"{stem}_edited{out_ext}"
            cand_path = target_dir / cand_name
            counter = 2
            while cand_path.exists():
                cand_name = f"{stem}_edited_{counter}{out_ext}"
                cand_path = target_dir / cand_name
                counter += 1

            # Atomic write
            tmp_path = cand_path.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
            tmp_path.write_bytes(image_bytes)
            os.replace(tmp_path, cand_path)

            st = cand_path.stat()
            file_size = st.st_size
            file_mtime = st.st_mtime
            qh = compute_quick_hash(cand_path, head_bytes, tail_bytes)
            sha = compute_sha256(cand_path)
            mime = "image/png" if out_ext == ".png" else "image/webp" if out_ext == ".webp" else "image/jpeg"

            try:
                rel_path = str(cand_path.relative_to(cfg.library_root)) if cfg.library_root else cand_name
            except (ValueError, TypeError):
                rel_path = cand_name

            new_rec = M.ImageRecord(
                path=str(cand_path),
                relative_path=rel_path,
                filename=cand_name,
                extension=out_ext,
                mime_type=mime,
                file_size=file_size,
                file_mtime=file_mtime,
                quick_hash=qh,
                sha256=sha,
                width=w,
                height=h,
                orientation=1,
                taken_at=rec.taken_at,
                taken_at_source=rec.taken_at_source,
                gps_lat=rec.gps_lat,
                gps_lon=rec.gps_lon,
                camera_model=rec.camera_model,
                city=rec.city,
                region=rec.region,
                country_code=rec.country_code,
                is_favorite=rec.is_favorite,
                stage=rec.stage,
                origin_path=rec.origin_path,
                category=rec.category,
                subcategory=rec.subcategory,
                scan_status="scanned",
                infer_status="inferred",
                route_status=rec.route_status,
                manual_bucket=rec.manual_bucket,
                burst_id=rec.burst_id,
            )
            s.add(new_rec)
            s.commit()
            s.refresh(new_rec)
            saved_id = new_rec.id
            saved_filename = new_rec.filename
            saved_path = new_rec.path
            saved_ext = new_rec.extension
            saved_qh = new_rec.quick_hash
            saved_sha = new_rec.sha256
            saved_size = new_rec.file_size
        else:
            # Overwrite mode
            target_path = orig_path
            tmp_path = target_path.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
            tmp_path.write_bytes(image_bytes)
            os.replace(tmp_path, target_path)

            st = target_path.stat()
            file_size = st.st_size
            file_mtime = st.st_mtime
            qh = compute_quick_hash(target_path, head_bytes, tail_bytes)
            sha = compute_sha256(target_path)

            rec.file_size = file_size
            rec.file_mtime = file_mtime
            rec.quick_hash = qh
            rec.sha256 = sha
            rec.width = w
            rec.height = h
            rec.orientation = 1
            s.commit()
            s.refresh(rec)
            saved_id = rec.id
            saved_filename = rec.filename
            saved_path = rec.path
            saved_ext = rec.extension
            saved_qh = rec.quick_hash
            saved_sha = rec.sha256
            saved_size = rec.file_size

    # Cache invalidation & regeneration
    if mode == "overwrite":
        for tier in ("micro", "medium", "preview"):
            tier_dir = cfg.cache_dir / "thumbs" / tier
            if tier_dir.exists():
                for p in tier_dir.glob(f"*_{image_id}_*.webp"):
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass

    # Pre-render fresh thumbnails
    from .thumbs import render_tier
    fp = hashlib.sha256(f"{saved_qh}|{saved_sha}|{saved_size}|{saved_filename}".encode()).hexdigest()[:16]
    for tier in ("micro", "medium", "preview"):
        try:
            tier_dir = cfg.cache_dir / "thumbs" / tier
            tier_dir.mkdir(parents=True, exist_ok=True)
            data = render_tier(saved_path, saved_ext, 1, tier)
            c_file = tier_dir / f"{db_fp}_{saved_id}_{fp}.webp"
            c_tmp = c_file.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
            c_tmp.write_bytes(data)
            os.replace(c_tmp, c_file)
        except Exception as e:
            log.debug("pre-rendering thumb failed for %s: %s", saved_id, e)

    return {
        "ok": True,
        "image_id": saved_id,
        "filename": saved_filename,
        "path": saved_path,
        "mode": mode,
        "width": w,
        "height": h,
    }


def remove_background(image_bytes: bytes) -> bytes:
    """Remove background from image bytes using rembg or OpenCV GrabCut fallback."""
    # 1. Try rembg / AI models if installed
    try:
        import rembg
        return rembg.remove(image_bytes)
    except Exception:
        pass

    # 2. Fallback to OpenCV cv2.grabCut with morphological refinement
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ActionError("Could not decode image for background removal", 400)

    h, w = bgr.shape[:2]
    max_gc = 600
    if max(h, w) > max_gc:
        scale = max_gc / max(h, w)
        small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
        gc_bgr = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        gc_bgr = bgr

    gh, gw = gc_bgr.shape[:2]
    mask = np.zeros((gh, gw), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    margin_x = max(1, int(gw * 0.05))
    margin_y = max(1, int(gh * 0.05))
    rect = (margin_x, margin_y, max(1, gw - 2 * margin_x), max(1, gh - 2 * margin_y))

    cv2.grabCut(gc_bgr, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
    gc_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gc_bin = cv2.morphologyEx(gc_bin, cv2.MORPH_CLOSE, kernel)

    if scale != 1.0:
        full_mask = cv2.resize(gc_bin, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        full_mask = gc_bin

    b, g, r = cv2.split(bgr)
    rgba = cv2.merge([b, g, r, full_mask])
    ok, out_buf = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("Failed to encode background-removed PNG")
    return out_buf.tobytes()


def extract_text(image_bytes: bytes) -> str:
    """Extract text from image using pytesseract if installed."""
    try:
        import pytesseract
        pil_img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(pil_img).strip()
    except Exception as e:
        log.info("OCR not available or failed: %s", e)
        return ""
