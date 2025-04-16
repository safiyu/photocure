"""Thumbnail pyramid: three WebP sizes rendered from the cheapest good source.

JPEG and other Pillow formats go through the shared loader (JPEG draft decode, HEIF already
oriented). Camera RAW uses the embedded preview and only falls back to a half-size develop
when that preview is smaller than the tier.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .quality.image_loader import RAW_EXTS, _bgr_from_pil, apply_exif_orientation, load_image

try:
    import rawpy
except ImportError:  # RAW then goes through the generic loader
    rawpy = None

log = logging.getLogger("photocure.thumbs")

# tier -> (longest side in px, WebP quality)
TIERS = {"micro": (440, 85), "medium": (800, 84), "preview": (1600, 86)}


def _fit(bgr: np.ndarray, side: int) -> np.ndarray:
    """Shrink so the longest side is at most ``side``; never upscale."""
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= side:
        return bgr
    k = side / longest
    return cv2.resize(bgr, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)


def raw_preview(path: Path | str, side: int, orientation: int = 1) -> np.ndarray | None:
    """BGR frame from a RAW file, or None when rawpy cannot read it.

    The embedded preview is used when it is at least ``side`` px on its longest edge. A JPEG
    preview that carries its own orientation tag is described by that tag; otherwise the RAW
    file's orientation applies. The developed fallback is already flipped by LibRaw.
    """
    if rawpy is None:
        return None
    try:
        with rawpy.imread(str(path)) as raw:
            pil = None
            own = None
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    pil = Image.open(io.BytesIO(thumb.data))
                    own = pil.getexif().get(274)
                    pil.load()
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    pil = Image.fromarray(thumb.data)
            except Exception as e:  # no preview, or one rawpy cannot give us
                log.debug("no usable embedded preview in %s (%s)", path, e)
                pil = None
            if pil is not None and max(pil.size) >= side:
                return _bgr_from_pil(apply_exif_orientation(pil, own or orientation))
            rgb = raw.postprocess(half_size=True, use_camera_wb=True)
            return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    except Exception as e:
        log.warning("rawpy could not read %s (%s: %s)", path, e.__class__.__name__, e)
        return None


def render_tier(path: Path | str, ext: str, orientation: int, tier: str) -> bytes:
    """Encode one tier of the pyramid as WebP bytes."""
    side, quality = TIERS[tier]
    bgr = raw_preview(path, side, orientation) if ext.lower() in RAW_EXTS else None
    if bgr is None:
        bgr = load_image(path, max_side=side, orientation=orientation).bgr
    ok, buf = cv2.imencode(".webp", _fit(bgr, side), [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:
        raise RuntimeError("WebP encode failed")
    return buf.tobytes()
