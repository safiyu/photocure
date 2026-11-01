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

    if ext == ".png":
        try:
            with Image.open(source_path) as im:
                if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                    rgba = im.convert("RGBA")
                    if orientation and orientation != 1:
                        from .quality.image_loader import apply_exif_orientation
                        rgba = apply_exif_orientation(rgba, orientation)
                    buf = io.BytesIO()
                    rgba.save(buf, format="PNG")
                    return buf.getvalue(), "image/png"
        except Exception:
            pass

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

        # Output extension determination & validation
        if format:
            fmt_clean = f".{format.lower().lstrip('.')}"
            if fmt_clean == ".jpeg":
                fmt_clean = ".jpg"
            if fmt_clean not in (".jpg", ".png", ".webp"):
                raise ActionError(f"Unsupported storage format '{format}'. Supported formats: jpg, png, webp", 400)
            out_ext = fmt_clean
        else:
            out_ext = orig_path.suffix.lower()
            if out_ext == ".jpeg":
                out_ext = ".jpg"
            if out_ext in RAW_EXTS or out_ext in HEIF_EXTS:
                out_ext = ".jpg"

        orig_ext = (rec.extension or orig_path.suffix).lower()
        if mode == "overwrite":
            if orig_ext in RAW_EXTS or orig_ext in HEIF_EXTS or orig_path.suffix.lower() in RAW_EXTS or orig_path.suffix.lower() in HEIF_EXTS:
                raise ActionError("Cannot overwrite RAW or HEIF photos directly. Please use 'Save as copy'.", 400)
            target_ext = orig_ext if orig_ext != ".jpeg" else ".jpg"
            if out_ext != target_ext:
                try:
                    with Image.open(io.BytesIO(image_bytes)) as pil_img:
                        buf = io.BytesIO()
                        if target_ext in (".jpg", ".jpeg"):
                            pil_img.convert("RGB").save(buf, format="JPEG", quality=95)
                        elif target_ext == ".webp":
                            pil_img.save(buf, format="WEBP", quality=95)
                        elif target_ext == ".png":
                            pil_img.save(buf, format="PNG")
                        image_bytes = buf.getvalue()
                        w, h = pil_img.size
                except Exception as e:
                    log.warning("Could not convert image to match original container %s: %s", target_ext, e)

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


_REMBG_SESSIONS: dict[str, object] = {}
_SR_SESSION: object | None = None
_RAPIDOCR_ENGINE: object | None = None


def _get_rembg_session(model_name: str = "birefnet-general"):
    """Load and cache an MIT / Apache-2.0 rembg session (never uses non-commercial bria-rmbg)."""
    import rembg
    allowed = ("birefnet-general", "isnet-general-use", "u2net", "u2netp")
    if model_name not in allowed:
        model_name = "birefnet-general"
    if model_name not in _REMBG_SESSIONS:
        _REMBG_SESSIONS[model_name] = rembg.new_session(model_name)
    return _REMBG_SESSIONS[model_name]


def remove_background(image_bytes: bytes, model_name: str = "birefnet-general") -> bytes:
    """Remove background from image bytes using BiRefNet (MIT) / U2-Net (Apache-2.0) or OpenCV GrabCut fallback."""
    # 1. Try rembg with strictly permissive MIT / Apache-2.0 models (BiRefNet -> U2-Net)
    for candidate in (model_name, "u2net"):
        try:
            import rembg
            session = _get_rembg_session(candidate)
            return rembg.remove(image_bytes, session=session)
        except Exception as exc:
            log.warning("rembg (%s) failed or uninstalled (%s)", candidate, exc)

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

    cv2.grabCut(gc_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
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


def _get_sr_session():
    """Load and cache the bundled ESPCN super-resolution ONNX model (Apache-2.0)."""
    global _SR_SESSION
    if _SR_SESSION is not None:
        return _SR_SESSION
    sr_path = Path(__file__).resolve().parent / "models" / "super_resolution_espcn.onnx"
    if not sr_path.is_file():
        return None
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        _SR_SESSION = ort.InferenceSession(str(sr_path), providers=["CPUExecutionProvider"])
        return _SR_SESSION
    except Exception as exc:
        log.warning("Failed to initialize ONNX super-resolution session: %s", exc)
        return None


def _run_espcn_y(sess, y_float: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Run tiled 224x224 -> 672x672 ESPCN ONNX super-resolution with raised-cosine blending."""
    h, w = y_float.shape
    tile = 224
    stride = 192
    pad_h = max(0, tile - h) if h < tile else (stride - (h - tile) % stride) % stride
    pad_w = max(0, tile - w) if w < tile else (stride - (w - tile) % stride) % stride
    padded = np.pad(y_float, ((0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape

    wy = np.hanning(tile * 3).astype(np.float32)
    wy = np.maximum(wy, 0.05)
    win = wy[:, None] * wy[None, :]

    coords = []
    tiles = []
    for y0 in range(0, ph - tile + 1, stride):
        for x0 in range(0, pw - tile + 1, stride):
            coords.append((y0, x0))
            tiles.append(padded[y0:y0 + tile, x0:x0 + tile])

    batch = np.stack(tiles, axis=0)[:, None, :, :].astype(np.float32)
    outs = []
    for i in range(0, len(batch), 32):
        outs.append(sess.run(None, {"input": batch[i:i + 32]})[0][:, 0, :, :])
    sr_tiles = np.concatenate(outs, axis=0)

    out_y = np.zeros((ph * 3, pw * 3), dtype=np.float32)
    weight = np.zeros((ph * 3, pw * 3), dtype=np.float32)
    for (y0, x0), st in zip(coords, sr_tiles):
        sy, sx = y0 * 3, x0 * 3
        out_y[sy:sy + tile * 3, sx:sx + tile * 3] += st * win
        weight[sy:sy + tile * 3, sx:sx + tile * 3] += win

    out_y /= np.maximum(weight, 1e-6)
    out_y = np.clip(out_y[:h * 3, :w * 3], 0.0, 1.0)
    return cv2.resize(out_y, (target_w, target_h), interpolation=cv2.INTER_AREA)


def remaster_image(
    image_bytes: bytes,
    denoise: bool = True,
    low_light: bool = True,
    upscale: bool = True,
) -> tuple[bytes, int, int]:
    """Run the 3-stage AI & Vision Remaster chain:
    1. Chroma + Luminance edge-preserving denoising (removes sensor & high-ISO grain).
    2. Low-Light & Exposure Correction (Zero-DCE quadratic illumination curve + CLAHE + vibrance).
    3. AI Super-Resolution & Detail Restoration (ESPCN ONNX neural super-resolution).
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ActionError("Could not decode image for remastering", 400)

    h, w = bgr.shape[:2]

    # ---- Stage 1: Chroma & Luminance Denoising ----
    if denoise:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        # Estimate sensor noise sigma via Median Absolute Deviation of Laplacian
        lap = cv2.Laplacian(l_ch, cv2.CV_16S)
        sigma = float(np.median(np.abs(lap))) / 0.6745
        h_luma = float(np.clip(sigma * 0.85, 3.0, 11.0))

        # Edge-preserving bilateral filter on chrominance channels A and B (eliminates color blotches)
        a_dn = cv2.bilateralFilter(a_ch, d=5, sigmaColor=18, sigmaSpace=18)
        b_dn = cv2.bilateralFilter(b_ch, d=5, sigmaColor=18, sigmaSpace=18)

        # Non-Local Means luminance denoising (on proxy if image > 1400px for speed, preserving high-freq edges)
        max_dn = 1400
        if max(h, w) > max_dn:
            sc = max_dn / max(h, w)
            sw, sh = max(1, int(w * sc)), max(1, int(h * sc))
            l_small = cv2.resize(l_ch, (sw, sh), interpolation=cv2.INTER_AREA)
            l_small_dn = cv2.fastNlMeansDenoising(l_small, None, h=h_luma, templateWindowSize=7, searchWindowSize=15)
            diff = cv2.resize(
                (l_small_dn.astype(np.float32) - l_small.astype(np.float32)),
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            l_dn = np.clip(l_ch.astype(np.float32) + diff, 0, 255).astype(np.uint8)
        else:
            l_dn = cv2.fastNlMeansDenoising(l_ch, None, h=h_luma, templateWindowSize=7, searchWindowSize=15)

        bgr = cv2.cvtColor(cv2.merge([l_dn, a_dn, b_dn]), cv2.COLOR_LAB2BGR)

    # ---- Stage 2: Low-Light & Exposure Correction (Zero-DCE curve + CLAHE + Vibrance) ----
    if low_light:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_norm = l_ch.astype(np.float32) / 255.0

        # Guard flat/solid images
        p1, p99 = np.percentile(l_ch, [1.0, 99.0])
        if p99 - p1 >= 4.0:
            # Spatially-smooth illumination estimation (Zero-DCE quadratic enhancement curve)
            illum = cv2.GaussianBlur(l_norm, (0, 0), sigmaX=max(8.0, min(w, h) * 0.03))
            mean_lum = float(np.mean(l_norm))
            # Stronger shadow lift when overall scene or local region is dark
            base_alpha = float(np.clip((0.52 - mean_lum) * 1.4, 0.05, 0.55))
            alpha_map = base_alpha * (1.0 - illum) * (1.0 - 0.5 * l_norm)
            for _ in range(3):
                l_norm = l_norm + alpha_map * l_norm * (1.0 - l_norm)

            l_ifted = np.clip(l_norm * 255.0, 0, 255).astype(np.uint8)

            # Gentle CLAHE micro-contrast blended at 35%
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l_ifted)
            l_blend = cv2.addWeighted(l_ifted, 0.65, l_clahe, 0.35, 0)

            # Soft 1-99% percentile stretch
            lo, hi = np.percentile(l_blend, [1.0, 99.0])
            if hi - lo > 10.0:
                l_stretch = np.clip((l_blend.astype(np.float32) - lo) * (250.0 / (hi - lo)) + 2.0, 0, 255).astype(np.uint8)
                l_ch = cv2.addWeighted(l_blend, 0.4, l_stretch, 0.6, 0)
            else:
                l_ch = l_blend

            # Natural vibrance (+14% boost weighted toward less-saturated pixels so skin tones don't clip)
            af = a_ch.astype(np.float32) - 128.0
            bf = b_ch.astype(np.float32) - 128.0
            chroma = np.hypot(af, bf) / 128.0
            vib_gain = 1.0 + 0.14 * (1.0 - np.clip(chroma, 0.0, 1.0))
            a_ch = np.clip(128.0 + af * vib_gain, 0, 255).astype(np.uint8)
            b_ch = np.clip(128.0 + bf * vib_gain, 0, 255).astype(np.uint8)

            bgr = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # ---- Stage 3: AI Super-Resolution & Detail Restoration (ESPCN ONNX) ----
    if upscale:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y_ch, cr_ch, cb_ch = cv2.split(ycrcb)
        y_float = y_ch.astype(np.float32) / 255.0

        sess = _get_sr_session()
        max_dim = max(h, w)
        if max_dim < 1600:
            # True 2x AI Upscale (capped at 3840px)
            scale_up = min(2.0, 3840.0 / max_dim)
            tw, th = max(1, int(round(w * scale_up))), max(1, int(round(h * scale_up)))
            if sess is not None:
                y_sr = _run_espcn_y(sess, y_float, tw, th)
                y_out = np.clip(y_sr * 255.0, 0, 255).astype(np.uint8)
            else:
                y_out = cv2.resize(y_ch, (tw, th), interpolation=cv2.INTER_CUBIC)
            cr_out = cv2.resize(cr_ch, (tw, th), interpolation=cv2.INTER_CUBIC)
            cb_out = cv2.resize(cb_ch, (tw, th), interpolation=cv2.INTER_CUBIC)
            bgr = cv2.cvtColor(cv2.merge([y_out, cr_out, cb_out]), cv2.COLOR_YCrCb2BGR)
            w, h = tw, th
        else:
            # High-res detail restoration via ESPCN on a working proxy + neural high-frequency injection
            if sess is not None:
                proxy_scale = 960.0 / max_dim
                pw, ph = max(32, int(round(w * proxy_scale))), max(32, int(round(h * proxy_scale)))
                y_small = cv2.resize(y_float, (pw, ph), interpolation=cv2.INTER_AREA)
                y_sr = _run_espcn_y(sess, y_small, w, h)
                y_base = cv2.resize(y_small, (w, h), interpolation=cv2.INTER_LINEAR)
                neural_detail = y_sr - y_base
                y_enhanced = np.clip(y_float + 0.55 * neural_detail, 0.0, 1.0)
                y_out = (y_enhanced * 255.0).astype(np.uint8)
                bgr = cv2.cvtColor(cv2.merge([y_out, cr_ch, cb_ch]), cv2.COLOR_YCrCb2BGR)
            else:
                blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.2)
                bgr = cv2.addWeighted(bgr, 1.25, blur, -0.25, 0)

    ok, out_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Failed to encode remastered image")
    return out_buf.tobytes(), w, h


def _get_rapidocr_engine():
    """Load and cache the RapidOCR (PaddleOCR v4 ONNX, Apache-2.0) engine."""
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is not None:
        return _RAPIDOCR_ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR
        _RAPIDOCR_ENGINE = RapidOCR()
        return _RAPIDOCR_ENGINE
    except Exception as exc:
        log.info("RapidOCR not available (%s)", exc)
        return None


def extract_text(image_bytes: bytes) -> str:
    """Extract text from image using RapidOCR (PaddleOCR v4 ONNX, Apache-2.0) with pytesseract fallback."""
    # 1. Try RapidOCR (Apache-2.0)
    engine = _get_rapidocr_engine()
    if engine is not None:
        try:
            arr = np.frombuffer(image_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                h, w = bgr.shape[:2]
                if max(h, w) > 2000:
                    sc = 2000.0 / max(h, w)
                    bgr = cv2.resize(bgr, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
                result, _ = engine(bgr)
                if not result:
                    return ""
                # Sort detected boxes top-to-bottom, then left-to-right
                sorted_lines = sorted(
                    result,
                    key=lambda item: (
                        round(float(item[0][0][1]) / 12.0),
                        float(item[0][0][0]),
                    ),
                )
                return "\n".join(str(item[1]).strip() for item in sorted_lines if str(item[1]).strip())
        except Exception as exc:
            log.warning("RapidOCR failed (%s); falling back to pytesseract", exc)

    # 2. Fallback to pytesseract (Apache-2.0)
    try:
        import pytesseract
        pil_img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(pil_img).strip()
    except Exception as e:
        log.info("OCR not available or failed: %s", e)
        return ""
