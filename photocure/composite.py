"""Open-eye composite for imperfect group shots: align two burst frames and blend faces across.

Pure image code: no database, no web layer. Nothing here touches an original file; outputs are
always new files next to the base photo.
"""
from __future__ import annotations

import contextlib
import os
import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_ALIGN_SIDE = 1600
_MIN_INLIERS = 25
_MIN_INLIER_RATIO = 0.4
_MIN_NCC = 0.5
_BAD_EYES = ("closed", "possible")


@dataclass
class FaceRef:
    box: tuple[float, float, float, float]  # x1,y1,x2,y2 as fractions of the oriented image
    eye_status: str  # open | possible | closed | unknown


@dataclass
class CompositeResult:
    bgr: np.ndarray | None = None
    applied: int = 0
    skipped: list[str] = field(default_factory=list)
    inliers: int = 0
    aligned_donor: np.ndarray | None = None
    mask: np.ndarray | None = None
    ok: bool = False
    reason: str = ""


def _px(box, w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return x1 * w, y1 * h, x2 * w, y2 * h


def _align(base: np.ndarray, donor: np.ndarray, base_faces, donor_faces):
    """Homography from donor to base pixels and the inlier count; (None, n) on failure."""
    bh, bw = base.shape[:2]
    dh, dw = donor.shape[:2]
    sb = min(1.0, _ALIGN_SIDE / max(bh, bw))
    sd = min(1.0, _ALIGN_SIDE / max(dh, dw))

    def prep(img, s, faces):
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        gh, gw = gray.shape
        for f in faces:
            x1, y1, x2, y2 = _px(f.box, gw, gh)
            pw, ph = (x2 - x1) * 0.25, (y2 - y1) * 0.25
            mask[max(0, int(y1 - ph)):int(y2 + ph) + 1, max(0, int(x1 - pw)):int(x2 + pw) + 1] = 0
        return gray, mask

    gb, mb = prep(base, sb, base_faces)
    gd, md = prep(donor, sd, donor_faces)
    orb = cv2.ORB_create(4000)
    kb, db = orb.detectAndCompute(gb, mb)
    kd, dd = orb.detectAndCompute(gd, md)
    if db is None or dd is None or len(kb) < 8 or len(kd) < 8:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dd, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < _MIN_INLIERS:
        return None, 0
    src = np.float32([kd[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    hom, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if hom is None or inl is None:
        return None, 0
    n = int(inl.sum())
    if n < _MIN_INLIERS or n / len(good) < _MIN_INLIER_RATIO:
        return None, n
    scale_b = np.diag([sb, sb, 1.0])
    scale_d = np.diag([sd, sd, 1.0])
    return np.linalg.inv(scale_b) @ hom @ scale_d, n


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) - float(a.mean())
    b = b.astype(np.float32) - float(b.mean())
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return 1.0 if denom < 1e-6 else float((a * b).sum() / denom)


def _blurred_gray(img: np.ndarray) -> np.ndarray:
    k = max(3, (min(img.shape[:2]) // 16) | 1)
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (k, k), 0)


def compose_open_eyes(base_bgr: np.ndarray, donor_bgr: np.ndarray,
                      base_faces: list[FaceRef], donor_faces: list[FaceRef]) -> CompositeResult:
    res = CompositeResult()
    h, w = base_bgr.shape[:2]
    hom, res.inliers = _align(base_bgr, donor_bgr, base_faces, donor_faces)
    if hom is None:
        res.reason = "The two frames could not be aligned."
        return res
    warped = cv2.warpPerspective(donor_bgr, hom, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(donor_bgr.shape[:2], 255, np.uint8), hom, (w, h),
                                flags=cv2.INTER_NEAREST)
    dh, dw = donor_bgr.shape[:2]
    donors = []
    for f in donor_faces:
        if f.eye_status != "open":
            continue
        x1, y1, x2, y2 = _px(f.box, dw, dh)
        c = cv2.perspectiveTransform(np.float32([[[(x1 + x2) / 2, (y1 + y2) / 2]]]), hom)[0, 0]
        donors.append((float(c[0]), float(c[1])))
    out = base_bgr.copy()
    soft = np.zeros((h, w), np.uint8)
    used: set[int] = set()
    for n, f in enumerate(base_faces, 1):
        if f.eye_status not in _BAD_EYES:
            continue
        label = f"Face {n}"
        x1, y1, x2, y2 = _px(f.box, w, h)
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 8:
            res.skipped.append(f"{label}: too small to fix.")
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        limit = 0.6 * float(np.hypot(bw, bh))
        cand = [(np.hypot(cx - dx, cy - dy), i) for i, (dx, dy) in enumerate(donors)
                if i not in used]
        cand = [c for c in cand if c[0] < limit]
        if not cand:
            res.skipped.append(f"{label}: no frame shows this person with open eyes.")
            continue
        _, di = min(cand)
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))
        if valid[iy1:iy2, ix1:ix2].mean() < 250:
            res.skipped.append(f"{label}: the face is outside the other frame.")
            continue
        base_face = _blurred_gray(base_bgr[iy1:iy2, ix1:ix2])
        if _ncc(base_face, _blurred_gray(warped[iy1:iy2, ix1:ix2])) < _MIN_NCC:
            res.skipped.append(f"{label}: head moved too much between frames.")
            continue
        pad = max(8, int(0.3 * max(bw, bh)))
        cx1, cy1 = max(0, ix1 - pad), max(0, iy1 - pad)
        cx2, cy2 = min(w, ix2 + pad), min(h, iy2 + pad)
        ell = np.zeros((cy2 - cy1, cx2 - cx1), np.uint8)
        axes = (max(2, int(bw * 0.45)), max(2, int(bh * 0.45)))
        cv2.ellipse(ell, (int(cx - cx1), int(cy - cy1)), axes, 0, 0, 360, 255, -1)
        bx, by, bwid, bhei = cv2.boundingRect(ell)
        try:
            blended = cv2.seamlessClone(warped[cy1:cy2, cx1:cx2], out[cy1:cy2, cx1:cx2], ell.copy(),
                                        (bx + bwid // 2, by + bhei // 2), cv2.NORMAL_CLONE)
        except cv2.error:
            res.skipped.append(f"{label}: blending failed.")
            continue
        out[cy1:cy2, cx1:cx2] = blended
        sigma = max(2.0, 0.06 * min(bw, bh))
        region = soft[cy1:cy2, cx1:cx2]
        np.maximum(region, cv2.GaussianBlur(ell, (0, 0), sigma), out=region)
        used.add(di)
        res.applied += 1
    if res.applied == 0:
        res.reason = res.skipped[0] if res.skipped else "No closed eyes to fix."
        return res
    res.bgr, res.aligned_donor, res.mask, res.ok = out, warped, soft, True
    return res


def _unique(base_path: Path, suffix: str, ext: str) -> Path:
    base_path = Path(base_path)
    p = base_path.with_name(f"{base_path.stem}{suffix}{ext}")
    n = 2
    while p.exists():
        p = base_path.with_name(f"{base_path.stem}{suffix}{n}{ext}")
        n += 1
    return p


def fixed_jpeg_path(base_path: Path) -> Path:
    return _unique(base_path, "_fixed", ".jpg")


def layered_psd_path(base_path: Path) -> Path:
    return _unique(base_path, "_layers", ".psd")


def _tmp_for(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")


def write_fixed_jpeg(bgr: np.ndarray, base_path: Path, out_path: Path) -> Path:
    from PIL import Image
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(str(out_path))
    exif_bytes = None
    try:
        with Image.open(base_path) as src:
            exif = src.getexif()
            if exif:
                for ifd in (0x8769, 0x8825):
                    with contextlib.suppress(KeyError):
                        exif.get_ifd(ifd)
                exif[0x0112] = 1  # pixels are already oriented
                exif_bytes = exif.tobytes()
    except Exception:
        exif_bytes = None
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    tmp = _tmp_for(out_path)
    try:
        kw = {"exif": exif_bytes} if exif_bytes else {}
        img.save(tmp, format="JPEG", quality=95, **kw)
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return out_path


def _planes(bgr: np.ndarray):
    rgb = bgr[:, :, ::-1]
    return [np.ascontiguousarray(rgb[:, :, c]).tobytes() for c in range(3)]


def _pascal(name: str) -> bytes:
    raw = name.encode("latin-1", "replace")[:255]
    data = bytes([len(raw)]) + raw
    return data + b"\0" * (-len(data) % 4)


def write_layered_psd(path: Path, base_bgr: np.ndarray, donor_bgr: np.ndarray, mask: np.ndarray,
                      merged_bgr: np.ndarray | None = None) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(str(path))
    h, w = base_bgr.shape[:2]
    merged = base_bgr if merged_bgr is None else merged_bgr
    if donor_bgr.shape != base_bgr.shape or merged.shape != base_bgr.shape:
        raise ValueError("layer sizes differ")
    if mask.shape != (h, w) or mask.dtype != np.uint8:
        raise ValueError("bad mask")
    plane = w * h
    opaque = b"\xff" * plane
    # (name, [(channel id, raw bytes)], has mask)
    layers = [
        ("Best frame", [(-1, opaque), *zip((0, 1, 2), _planes(base_bgr), strict=True)], False),
        ("Open eyes (aligned)",
         [(-1, opaque), *zip((0, 1, 2), _planes(donor_bgr), strict=True),
          (-2, mask.tobytes())], True),
    ]
    records = bytearray()
    for name, chans, has_mask in layers:
        records += struct.pack(">iiii", 0, 0, h, w)
        records += struct.pack(">H", len(chans))
        for cid, data in chans:
            records += struct.pack(">hI", cid, 2 + len(data))
        records += b"8BIM" + b"norm" + bytes([255, 0, 0, 0])
        if has_mask:
            mask_data = struct.pack(">iiii", 0, 0, h, w) + bytes([0, 0]) + b"\0\0"
            extra = struct.pack(">I", len(mask_data)) + mask_data
        else:
            extra = struct.pack(">I", 0)
        extra += struct.pack(">I", 0) + _pascal(name)
        records += struct.pack(">I", len(extra)) + extra
    chan_bytes = sum(2 + len(d) for _, chans, _ in layers for _, d in chans)
    info_len = 2 + len(records) + chan_bytes
    pad = info_len % 2
    info_len += pad
    tmp = _tmp_for(path)
    try:
        with open(tmp, "wb") as f:
            header = b"8BPS" + struct.pack(">H", 1) + b"\0" * 6
            f.write(header + struct.pack(">HIIHH", 3, h, w, 8, 3))
            f.write(struct.pack(">I", 0) + struct.pack(">I", 0))
            f.write(struct.pack(">I", 4 + info_len + 4) + struct.pack(">I", info_len))
            f.write(struct.pack(">h", len(layers)) + bytes(records))
            for _, chans, _ in layers:
                for _, data in chans:
                    f.write(struct.pack(">H", 0) + data)
            f.write(b"\0" * pad + struct.pack(">I", 0))
            f.write(struct.pack(">H", 0))
            for p in _planes(merged):
                f.write(p)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
