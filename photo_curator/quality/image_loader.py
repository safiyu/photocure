"""Unified image loading for every supported format.

Loads JPEG / PNG / WebP / HEIC / TIFF via Pillow and camera RAW (CR2/NEF/ARW/DNG)
via ``rawpy``, always returning an OpenCV-compatible ``BGR`` uint8 ndarray.

``load_image`` is the single decode path used by the quality engine, face
engine and semantic engine, so HEIC/RAW quirks are handled exactly once.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # HEIC support becomes a no-op
    pillow_heif = None

try:
    import rawpy
except ImportError:  # RAW support becomes a no-op
    rawpy = None


@dataclass
class LoadedImage:
    bgr: np.ndarray
    width: int
    height: int
    decode: str  # "pillow" | "rawpy"
    warning: str = ""


def _bgr_from_pil(im: Image.Image) -> np.ndarray:
    if im.mode not in ("RGB", "RGBA", "L"):
        im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)
    if im.mode == "RGBA":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif im.mode == "RGB":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif im.mode == "L":
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(arr)


def load_image(path: Path | str,
               max_side: Optional[int] = None,
               orientation: Optional[int] = None) -> LoadedImage:
    """Decode any supported photo to a BGR ndarray.

    Args:
        path: source file.
        max_side: if set, downscale so the longest side <= ``max_side``
            (used to keep quality metrics resolution-independent & save RAM).
        orientation: EXIF orientation tag; when provided the image is
            post-rotated so downstream geometry (faces, crops) matches what
            the user sees.
    """
    path = Path(path)
    ext = path.suffix.lower()

    raw_exts = {".cr2", ".nef", ".arw", ".dng", ".rw2", ".orf", ".raf"}
    warning = ""

    if ext in raw_exts and rawpy is not None:
        try:
            with rawpy.imread(str(path)) as rp:
                rgb = rp.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB)
            bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
            img = LoadedImage(bgr=bgr, width=bgr.shape[1], height=bgr.shape[0],
                              decode="rawpy")
        except Exception as e:  # corrupt RAW -> try pillow as last resort
            warning = f"rawpy failed ({e.__class__.__name__}); fell back to pillow"
            bgr = _load_pillow(path, warning)
            img = LoadedImage(bgr=bgr, width=bgr.shape[1], height=bgr.shape[0],
                              decode="pillow", warning=warning)
    else:
        bgr = _load_pillow(path, warning)
        img = LoadedImage(bgr=bgr, width=bgr.shape[1], height=bgr.shape[0],
                          decode="pillow", warning=warning)

    # EXIF orientation correction (Pillow applies EXIF for some formats but
    # not consistently for RAW-derived frames; be explicit when we know it).
    if orientation and orientation != 1:
        bgr = _apply_orientation(img.bgr, orientation)
        img.bgr, img.width, img.height = bgr, bgr.shape[1], bgr.shape[0]

    if max_side and max(img.width, img.height) > max_side:
        scale = max_side / max(img.width, img.height)
        new_w, new_h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img.bgr = cv2.resize(img.bgr, (new_w, new_h), interpolation=interp)
        img.width, img.height = new_w, new_h
    return img


def _load_pillow(path: Path, warning: str = "") -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        return _bgr_from_pil(im)


def _apply_orientation(bgr: np.ndarray, orientation: int) -> np.ndarray:
    """Apply the 8 EXIF orientation values to a BGR array."""
    if orientation == 2:
        return cv2.flip(bgr, 1)
    if orientation == 3:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(bgr, 0)
    if orientation == 5:
        return cv2.flip(cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    if orientation == 6:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), 1)
    if orientation == 8:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def to_thumbnail_jpg(bgr: np.ndarray, max_side: int = 384, quality: int = 82) -> bytes:
    """Encode a BGR image as JPEG bytes (used for cached web thumbnails)."""
    if max(bgr.shape[0], bgr.shape[1]) > max_side:
        scale = max_side / max(bgr.shape[0], bgr.shape[1])
        bgr = cv2.resize(bgr, (max(1, int(bgr.shape[1] * scale)),
                               max(1, int(bgr.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("thumbnail encode failed")
    return buf.tobytes()


def encode_preview(path: Path | str, max_side: int = 384) -> Tuple[bytes, str]:
    """Load + encode a preview; returns (jpeg_bytes, mime)."""
    img = load_image(path, max_side=max_side)
    return to_thumbnail_jpg(img.bgr, max_side=max_side), "image/jpeg"
