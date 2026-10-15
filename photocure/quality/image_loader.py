"""Unified image loading for every supported format.

Loads JPEG / PNG / WebP / HEIC / TIFF via Pillow and camera RAW (CR2/NEF/ARW/DNG)
via ``rawpy``, always returning an OpenCV-compatible ``BGR`` uint8 ndarray.

``load_image`` is the single decode path used by the quality engine, face
engine and semantic engine, so HEIC/RAW quirks are handled exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

# Explicit pixel cap: big enough for 200MP panoramas, small enough that a forged
# header cannot allocate gigabytes. Larger frames are treated as decode failures.
MAX_PIXELS = 500_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
# Decode partially downloaded / truncated JPEGs instead of failing outright.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DecodeError(Exception):
    """The file cannot be decoded into a usable frame."""

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
    # Pixel size of the source before any downscale (scale = max(w, h) / max(orig)).
    orig_width: int = 0
    orig_height: int = 0

    @property
    def scale(self) -> float:
        orig = max(self.orig_width, self.orig_height)
        return max(self.width, self.height) / orig if orig else 1.0


_HIGH_DEPTH_MODES = ("I;16", "I;16L", "I;16B", "I;16N", "I", "F")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Rescale a 16-bit / 32-bit / float plane into 0..255 without clipping."""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    if arr.dtype.kind == "f" and lo >= 0.0 and hi <= 1.0:
        return (a * 255.0).round().astype(np.uint8)
    if lo >= 0.0 and hi <= 255.0:
        return a.round().astype(np.uint8)
    if lo >= 0.0 and hi <= 65535.0:
        return (a / 257.0).round().astype(np.uint8)
    if hi - lo < 1e-12:
        return np.zeros(a.shape, dtype=np.uint8)
    return ((a - lo) * (255.0 / (hi - lo))).round().astype(np.uint8)


def _bgr_from_pil(im: Image.Image) -> np.ndarray:
    if im.mode in _HIGH_DEPTH_MODES:
        gray = _to_uint8(np.asarray(im))
        return np.ascontiguousarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
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
               max_side: int | None = None,
               orientation: int | None = None) -> LoadedImage:
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
    heif_exts = {".heic", ".heif"}
    warning = ""
    # True when the decoder already rotated the pixels per the file's metadata.
    pre_oriented = False

    if ext in raw_exts and rawpy is not None:
        try:
            with rawpy.imread(str(path)) as rp:
                # LibRaw applies the camera flip by default (user_flip unset).
                rgb = rp.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB)
            bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
            img = LoadedImage(bgr=bgr, width=bgr.shape[1], height=bgr.shape[0],
                              decode="rawpy", orig_width=bgr.shape[1],
                              orig_height=bgr.shape[0])
            pre_oriented = True
        except Exception as e:  # corrupt RAW -> try pillow as last resort
            warning = f"rawpy failed ({e.__class__.__name__}); fell back to pillow"
            bgr, ow, oh = _load_pillow(path, max_side)
            img = LoadedImage(bgr=bgr, width=bgr.shape[1], height=bgr.shape[0],
                              decode="pillow", warning=warning,
                              orig_width=ow, orig_height=oh)
    else:
        bgr, ow, oh = _load_pillow(path, max_side)
        img = LoadedImage(bgr=bgr, width=bgr.shape[1], height=bgr.shape[0],
                          decode="pillow", warning=warning,
                          orig_width=ow, orig_height=oh)
        # pillow_heif applies the irot/imir boxes itself; rotating again by the
        # EXIF tag would turn the frame twice.
        pre_oriented = ext in heif_exts and pillow_heif is not None

    if orientation and orientation != 1 and not pre_oriented:
        bgr = _apply_orientation(img.bgr, orientation)
        img.bgr, img.width, img.height = bgr, bgr.shape[1], bgr.shape[0]
        if orientation in (5, 6, 7, 8):
            img.orig_width, img.orig_height = img.orig_height, img.orig_width

    if max_side and max(img.width, img.height) > max_side:
        scale = max_side / max(img.width, img.height)
        new_w, new_h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img.bgr = cv2.resize(img.bgr, (new_w, new_h), interpolation=interp)
        img.width, img.height = new_w, new_h
    return img


def _load_pillow(path: Path, max_side: int | None = None) -> tuple[np.ndarray, int, int]:
    """Decode via Pillow; returns ``(bgr, orig_width, orig_height)``.

    Frames over :data:`MAX_PIXELS` raise :class:`DecodeError`. JPEGs are
    decoded at a reduced DCT scale when ``max_side`` is given.
    """
    try:
        with Image.open(path) as im:
            ow, oh = im.size
            if ow * oh > MAX_PIXELS:
                raise DecodeError(f"{ow}x{oh} exceeds the {MAX_PIXELS} pixel cap")
            if max_side and im.format == "JPEG" and max(ow, oh) > max_side:
                ratio = max_side / max(ow, oh)
                im.draft("RGB", (max(1, int(ow * ratio)), max(1, int(oh * ratio))))
            im.load()
            return _bgr_from_pil(im), ow, oh
    except Image.DecompressionBombError as e:
        raise DecodeError(str(e)) from e


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


def encode_preview(path: Path | str, max_side: int = 384) -> tuple[bytes, str]:
    """Load + encode a preview; returns (jpeg_bytes, mime)."""
    img = load_image(path, max_side=max_side)
    return to_thumbnail_jpg(img.bgr, max_side=max_side), "image/jpeg"
