"""Clean export: find sensitive details in photos, blur them, write a metadata-free share bundle.

Originals are only ever read. The bundle holds fresh JPEGs encoded from pixels, so no EXIF, GPS,
serial numbers, XMP or ICC data can ride along, plus a manifest of hashes and an optional gallery.
"""
from __future__ import annotations

import hashlib
import html as _html
import io
import json
import logging
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

log = logging.getLogger("photocure.clean_export")

PII_LABELS = {
    "license_plate": "License plate",
    "payment_card": "Payment card",
    "screen": "Screen showing text, code or email",
    "house_number": "House number",
    "id_document": "Passport, ID card or driving licence",
}

PII_PROMPTS: dict[str, list[str]] = {
    "license_plate": ["a photo of a car license plate",
                      "a close up of a vehicle registration plate", "a number plate on a car"],
    "payment_card": ["a photo of a credit card", "a bank card with a number on it",
                     "a debit card on a table"],
    "screen": ["a computer screen showing text or code", "a phone screen showing an email",
               "a laptop display with a document open", "a monitor with a spreadsheet"],
    "house_number": ["a house number on a wall", "a street address number plate on a door",
                     "a house with its street number visible"],
    "id_document": ["a photo of a passport", "an identity card", "a driving licence document"],
}

BENIGN_PROMPTS = [
    "a photo of a landscape", "a photo of people smiling", "a photo of a dog",
    "a photo of food on a plate", "a photo of a building", "a photo of a beach",
    "a photo of flowers", "a photo of a street",
]

FLAG_THRESHOLD = 0.5
_MIN_SIDE = 24
_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]")


class PiiUnavailable(RuntimeError):
    """Raised when the AI model needed to look for sensitive details is not available."""


@dataclass
class Region:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str


@dataclass
class Finding:
    image_id: int
    labels: list[str]
    score: float
    regions: list[Region] = field(default_factory=list)
    localized: bool = True


# ------------------------------------------------------------------ text-like regions
def find_text_boxes(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Wide, text-like boxes ``(x1, y1, x2, y2)`` in pixels of ``img``."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return []
    k = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(gray, (max(1, round(w * k)), max(1, round(h * k))),
                       interpolation=cv2.INTER_AREA) if k < 1.0 else gray
    sh, sw = small.shape
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, ellipse)
    if int(grad.max()) < 24:
        return []
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, sw // 30), max(3, sh // 80)))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kern)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, bw_, bh = cv2.boundingRect(c)
        area = bw_ * bh
        too_small = area < 0.002 * sw * sh or bh < max(4, 0.01 * sh)
        if too_small or area > 0.85 * sw * sh or bw_ < 1.5 * bh:
            continue
        out.append((int(x / k), int(y / k), min(w, int((x + bw_) / k)), min(h, int((y + bh) / k))))
    return out


def _overlaps(a: Region, b: Region) -> bool:
    return a.label == b.label and a.x1 < b.x2 and b.x1 < a.x2 and a.y1 < b.y2 and b.y1 < a.y2


def merge_regions(regions: Sequence[Region]) -> list[Region]:
    """Union overlapping regions that carry the same label."""
    out = [Region(r.x1, r.y1, r.x2, r.y2, r.label) for r in regions]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                if _overlaps(out[i], out[j]):
                    a, b = out[i], out.pop(j)
                    out[i] = Region(min(a.x1, b.x1), min(a.y1, b.y1), max(a.x2, b.x2),
                                    max(a.y2, b.y2), a.label)
                    changed = True
                    break
            if changed:
                break
    return out


# ------------------------------------------------------------------ scanner
def _tiles() -> list[tuple[float, float, float, float, bool]]:
    """Fractions ``(x1, y1, x2, y2, is_whole)``: whole photo, 2x2 halves, 3x3 thirds."""
    tiles = [(0.0, 0.0, 1.0, 1.0, True)]
    for n in (2, 3):
        for r in range(n):
            for c in range(n):
                tiles.append((c / n, r / n, (c + 1) / n, (r + 1) / n, False))
    return tiles


class PiiScanner:
    """Zero-shot CLIP check of a photo and its tiles for plates, cards, screens, numbers, IDs."""

    def __init__(self, cfg=None, backend=None, engine=None):
        if engine is None:
            from .inference.semantic_engine import SemanticEngine
            engine = SemanticEngine(cfg, backend=backend)
        self.engine = engine
        try:
            ok = bool(engine.text_search_available())
        except Exception as e:
            self.release()
            raise PiiUnavailable(
                f"The AI model could not be loaded ({e.__class__.__name__}).") from e
        if not ok:
            self.release()
            raise PiiUnavailable("Looking for sensitive details needs the AI model, which is not "
                                 "installed. Run: photocure setup")
        self.labels = list(PII_PROMPTS)
        texts: list[str] = []
        self._idx: list[list[int]] = []
        for lab in self.labels:
            self._idx.append(list(range(len(texts), len(texts) + len(PII_PROMPTS[lab]))))
            texts += PII_PROMPTS[lab]
        texts += BENIGN_PROMPTS
        feats = np.asarray(engine.encode_queries(texts), dtype=np.float32)
        self._text = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-9)

    def release(self) -> None:
        try:
            self.engine.release()
        except Exception:
            log.debug("engine release failed", exc_info=True)

    def scan_image(self, image_id: int, bgr: np.ndarray) -> Finding | None:
        h, w = bgr.shape[:2]
        if h < _MIN_SIDE or w < _MIN_SIDE:
            return None
        tiles = _tiles()
        crops = []
        for x1, y1, x2, y2, _ in tiles:
            px1, py1 = round(x1 * w), round(y1 * h)
            crops.append(bgr[py1:max(py1 + 1, round(y2 * h)), px1:max(px1 + 1, round(x2 * w))])
        results = self.engine.classify_and_embed(crops)
        emb = np.asarray([r.embedding for r in results], dtype=np.float32)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        logits = 100.0 * (emb @ self._text.T)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        mass = np.stack([probs[:, idx].sum(axis=1) for idx in self._idx], axis=1)

        regions: list[Region] = []
        labels: set[str] = set()
        score = 0.0
        tile_labels: set[str] = set()
        whole_hit: list[str] = []
        for ti, (x1, y1, x2, y2, whole) in enumerate(tiles):
            for li, lab in enumerate(self.labels):
                m = float(mass[ti, li])
                if m < FLAG_THRESHOLD:
                    continue
                labels.add(lab)
                score = max(score, m)
                if whole:
                    whole_hit.append(lab)
                else:
                    tile_labels.add(lab)
                    regions += self._refine(bgr, (x1, y1, x2, y2), lab)
        if not labels:
            return None
        whole_only = [lab for lab in whole_hit if lab not in tile_labels]
        for lab in whole_only:
            regions += self._refine(bgr, (0.0, 0.0, 1.0, 1.0), lab)
        localized = not whole_only
        return Finding(image_id, sorted(labels), round(score, 4), merge_regions(regions), localized)

    @staticmethod
    def _refine(bgr: np.ndarray, rect: tuple[float, float, float, float],
                label: str) -> list[Region]:
        h, w = bgr.shape[:2]
        x0, y0 = round(rect[0] * w), round(rect[1] * h)
        x1, y1 = round(rect[2] * w), round(rect[3] * h)
        boxes = find_text_boxes(bgr[y0:y1, x0:x1])
        if not boxes:
            return [Region(rect[0], rect[1], rect[2], rect[3], label)]
        return [Region((x0 + bx1) / w, (y0 + by1) / h, (x0 + bx2) / w, (y0 + by2) / h, label)
                for bx1, by1, bx2, by2 in boxes]


# ------------------------------------------------------------------ blur and encode
def blur_regions(bgr: np.ndarray, regions: Sequence[Region]) -> np.ndarray:
    """Copy of ``bgr`` with each region (padded about 8 percent) heavily blurred and feathered."""
    out = bgr.copy()
    h, w = out.shape[:2]
    for r in regions:
        rx1, ry1, rx2, ry2 = r.x1 * w, r.y1 * h, r.x2 * w, r.y2 * h
        pad = max(2, round(0.08 * max(rx2 - rx1, ry2 - ry1)))
        x1, y1 = max(0, int(rx1) - pad), max(0, int(ry1) - pad)
        x2, y2 = min(w, int(np.ceil(rx2)) + pad), min(h, int(np.ceil(ry2)) + pad)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = out[y1:y2, x1:x2]
        k = max(31, int(0.5 * min(roi.shape[1], roi.shape[0])))
        k = min(k, 301) | 1
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        edge = np.minimum(np.minimum(xx, roi.shape[1] - 1 - xx),
                          np.minimum(yy, roi.shape[0] - 1 - yy))
        alpha = np.clip((edge + 1) / float(pad), 0.0, 1.0)[..., None].astype(np.float32)
        out[y1:y2, x1:x2] = (blurred * alpha + roi * (1.0 - alpha)).round().astype(np.uint8)
    return out


def clean_jpeg_bytes(bgr: np.ndarray, quality: int = 92) -> bytes:
    """JPEG from pixels only, so no EXIF, XMP, ICC, GPS, maker notes or thumbnail survive."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


_MARKERS = (b"Exif", b"http://ns.adobe.com", b"ICC_PROFILE", b"MM\x00*", b"II*\x00")


def _header_bytes(data: bytes) -> bytes:
    """Everything before the image scan of a JPEG (where metadata lives), else all bytes."""
    if data[:2] != b"\xff\xd8":
        return data
    pos = 2
    while pos + 4 <= len(data) and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker == 0xDA:
            return data[:pos]
        pos += 2 + int.from_bytes(data[pos + 2:pos + 4], "big")
    return data[:pos]


def verify_clean(data_or_path: bytes | str | os.PathLike) -> list[str]:
    """Problems that mean a file still carries metadata; an empty list means clean."""
    from PIL import Image
    if isinstance(data_or_path, (bytes, bytearray)):
        data = bytes(data_or_path)
    else:
        data = Path(data_or_path).read_bytes()
    problems = []
    with Image.open(io.BytesIO(data)) as im:
        exif = im.getexif()
        if len(exif):
            problems.append("EXIF data present")
        if exif.get_ifd(0x8825):
            problems.append("GPS data present")
        for key in ("icc_profile", "xmp", "comment", "exif"):
            if im.info.get(key):
                problems.append(f"{key} present")
    head = _header_bytes(data)
    for marker in _MARKERS:
        if marker in head:
            problems.append(f"metadata marker {marker!r} found")
    return problems


# ------------------------------------------------------------------ bundle
@dataclass
class ExportItem:
    image_id: int
    path: str | os.PathLike
    orientation: int = 1
    extension: str = ""
    taken_at: str | None = None


@dataclass
class BundleResult:
    folder: Path
    exported: int = 0
    skipped_flagged: int = 0
    blurred: int = 0
    failures: list[str] = field(default_factory=list)
    flagged: int = 0
    manifest: Path | None = None


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _new_folder(out_root: Path, name: str | None) -> Path:
    safe = _UNSAFE.sub("_", Path(name or "").name).strip(" .") if name else ""
    base = safe or "clean_export_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        folder = out_root / (base if n == 1 else f"{base}-{n}")
        try:
            folder.mkdir(exist_ok=False)
            return folder
        except FileExistsError:
            n += 1


def _default_loader(path, orientation):
    from .quality.image_loader import load_image
    return load_image(path, max_side=None, orientation=orientation).bgr


def _gallery(title: str, files: list[dict], show_names: bool) -> str:
    cells = []
    for i, f in enumerate(files, 1):
        href = "images/" + quote(f["name"])
        alt = _html.escape(f["name"] if show_names else f"Photo {i}", quote=True)
        cap = f"<figcaption>{_html.escape(f['name'])}</figcaption>" if show_names else ""
        img = f'<img src="{href}" alt="{alt}" loading="lazy">'
        cells.append(f'<figure><a href="{href}">{img}</a>{cap}</figure>')
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_html.escape(title)}</title><style>"
        "body{margin:0;padding:24px;font:15px system-ui,sans-serif;background:#111;color:#eee}"
        "h1{font-size:20px;font-weight:600}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%;height:auto;display:block;border-radius:6px}"
        "figcaption{font-size:12px;opacity:.7;margin-top:4px;overflow:hidden;text-overflow:ellipsis}"
        f"</style></head><body><h1>{_html.escape(title)}</h1>"
        f'<div class="grid">{"".join(cells)}</div></body></html>\n'
    )


def export_bundle(items: Sequence[ExportItem], out_root: str | os.PathLike, *,
                  findings: dict[int, Finding] | None = None, policy: str = "blur",
                  html: bool = True, rename: bool = True, bundle_name: str | None = None,
                  on_progress: Callable[[int, int, str], None] | None = None,
                  cancel_check: Callable[[], bool] | None = None,
                  loader: Callable | None = None) -> BundleResult:
    """Write a metadata-free copy of ``items`` into a new folder under ``out_root``."""
    if policy not in ("blur", "skip", "include"):
        raise ValueError(f"policy must be blur, skip or include, got {policy!r}")
    loader = loader or _default_loader
    findings = findings or {}
    ordered = sorted(items, key=lambda it: (it.taken_at is None, it.taken_at or "", it.image_id))
    folder = _new_folder(Path(out_root), bundle_name)
    images = folder / "images"
    images.mkdir()
    res = BundleResult(folder=folder)
    files: list[dict] = []
    used: set[str] = set()
    total = len(ordered)
    for done, item in enumerate(ordered):
        if cancel_check and cancel_check():
            break
        finding = findings.get(item.image_id)
        flagged = bool(finding and finding.labels)
        res.flagged += int(flagged)
        name = ""
        if flagged and policy == "skip":
            res.skipped_flagged += 1
        else:
            try:
                bgr = loader(item.path, item.orientation)
                blurred = False
                if flagged and policy == "blur":
                    regions = finding.regions or [Region(0.0, 0.0, 1.0, 1.0, finding.labels[0])]
                    bgr = blur_regions(bgr, regions)
                    blurred = True
                data = clean_jpeg_bytes(bgr)
                problems = verify_clean(data)
                if problems:
                    raise RuntimeError("; ".join(problems))
                if rename:
                    name = f"photo-{len(files) + 1:04d}.jpg"
                else:
                    stem = _UNSAFE.sub("_", Path(str(item.path)).stem).strip(" .")[:80] or "photo"
                    name, n = f"{stem}.jpg", 2
                    while name.lower() in used:
                        name, n = f"{stem}-{n}.jpg", n + 1
                used.add(name.lower())
                _write_atomic(images / name, data)
                files.append({
                    "name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                    "width": int(bgr.shape[1]), "height": int(bgr.shape[0]),
                    "flagged_labels": list(finding.labels) if flagged else [],
                    "blurred": blurred, "verified_clean": True,
                })
                res.exported += 1
                res.blurred += int(blurred)
            except Exception as e:
                name = ""
                log.warning("clean export failed for image %s: %s", item.image_id, e)
                res.failures.append(f"image {item.image_id}: {e}")
        if on_progress:
            on_progress(done + 1, total, name or "skipped")
    manifest = {
        "version": 1, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": policy,
        "totals": {"exported": res.exported, "flagged": res.flagged, "blurred": res.blurred,
                   "skipped_flagged": res.skipped_flagged, "failures": len(res.failures)},
        "files": files,
    }
    res.manifest = folder / "manifest.json"
    _write_atomic(res.manifest, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    if html:
        title = folder.name if bundle_name else "Photos"
        _write_atomic(folder / "index.html", _gallery(title, files, not rename).encode("utf-8"))
    return res
