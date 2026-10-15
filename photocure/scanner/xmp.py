"""XMP sidecar reader (date, star rating, color label, keywords).

Looks for ``IMG.xmp`` and ``IMG.ext.xmp`` next to a photo. Parsing uses the
stdlib XML parser on a size-capped file with DOCTYPE/entity declarations
rejected, so hostile sidecars cannot trigger entity expansion.
"""
from __future__ import annotations

import datetime as dt
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

_MAX_BYTES = 2 * 1024 * 1024

_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "exif": "http://ns.adobe.com/exif/1.0/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

_DATE_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?)?"
    r"\s*(Z|[+-]\d{2}:?\d{2})?$")


@dataclass
class XmpInfo:
    taken_at: dt.datetime | None = None  # naive wall clock as written in the sidecar
    tz_offset_minutes: int | None = None
    rating: int | None = None
    label: str = ""
    keywords: list[str] = field(default_factory=list)


def sidecar_paths(path: Path | str) -> list[Path]:
    p = Path(path)
    return [p.with_suffix(".xmp"), p.with_name(p.name + ".xmp")]


def find_sidecar(path: Path | str) -> Path | None:
    for c in sidecar_paths(path):
        if c.is_file():
            return c
    return None


def parse_xmp_date(s: str | None) -> tuple[dt.datetime | None, int | None]:
    """Parse an ISO-8601 XMP date into ``(naive wall clock, offset minutes)``."""
    if not s:
        return None, None
    m = _DATE_RE.match(s.strip())
    if not m:
        return None, None
    y, mo, d, hh, mm, ss, tz = m.groups()
    try:
        parsed = dt.datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0),
                             int(ss or 0))
    except ValueError:
        return None, None
    offset = None
    if tz:
        if tz == "Z":
            offset = 0
        else:
            sign = -1 if tz[0] == "-" else 1
            digits = tz[1:].replace(":", "")
            offset = sign * (int(digits[:2]) * 60 + int(digits[2:]))
    return parsed, offset


def _attr_or_child(desc: ET.Element, ns: str, name: str) -> str | None:
    v = desc.attrib.get(f"{{{_NS[ns]}}}{name}")
    if v is not None:
        return v
    el = desc.find(f"{{{_NS[ns]}}}{name}")
    return el.text.strip() if el is not None and el.text else None


def parse_xmp(data: bytes) -> XmpInfo:
    info = XmpInfo()
    if len(data) > _MAX_BYTES or b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        return info
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return info
    for desc in root.iter(f"{{{_NS['rdf']}}}Description"):
        if info.taken_at is None:
            raw = (_attr_or_child(desc, "exif", "DateTimeOriginal")
                   or _attr_or_child(desc, "photoshop", "DateCreated")
                   or _attr_or_child(desc, "xmp", "CreateDate"))
            info.taken_at, info.tz_offset_minutes = parse_xmp_date(raw)
        if info.rating is None:
            r = _attr_or_child(desc, "xmp", "Rating")
            if r is not None:
                try:
                    info.rating = max(-1, min(5, int(float(r))))
                except ValueError:
                    pass
        if not info.label:
            info.label = _attr_or_child(desc, "xmp", "Label") or ""
        subj = desc.find(f"{{{_NS['dc']}}}subject")
        if subj is not None:
            for li in subj.iter(f"{{{_NS['rdf']}}}li"):
                if li.text and li.text.strip() and li.text.strip() not in info.keywords:
                    info.keywords.append(li.text.strip())
    return info


def read_xmp(path: Path | str) -> XmpInfo | None:
    """Read the sidecar for ``path``; ``None`` when no sidecar exists."""
    sc = find_sidecar(path)
    if sc is None:
        return None
    try:
        if sc.stat().st_size > _MAX_BYTES:
            return XmpInfo()
        return parse_xmp(sc.read_bytes())
    except OSError:
        return None
