"""XMP sidecar reader (date, star rating, color label, keywords) and non-destructive writer.

Looks for ``IMG.xmp`` and ``IMG.ext.xmp`` next to a photo. Parsing uses the
stdlib XML parser on a size-capped file with DOCTYPE/entity declarations
rejected, so hostile sidecars cannot trigger entity expansion.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

log = logging.getLogger("photocure.scanner.xmp")

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


# ---------------------------------------------------------------- writer
_RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
_XMP = "{http://ns.adobe.com/xap/1.0/}"
_DC = "{http://purl.org/dc/elements/1.1/}"
OWNED_TAG_PREFIX = "photocure-"


def _writer_sidecar(photo: Path) -> Path | None:
    """Sidecar to update or create; None when both forms exist (ambiguous)."""
    ext_form = photo.with_name(photo.name + ".xmp")
    base_form = photo.with_suffix(".xmp")
    if ext_form.exists() and base_form.exists():
        return None
    return base_form if base_form.exists() else ext_form


def _safe_parse(path: Path):
    """Parse with size cap and DOCTYPE/entity rejection, matching the reader."""
    if path.stat().st_size > _MAX_BYTES:
        raise ValueError("sidecar too large")
    data = path.read_bytes()
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise ValueError("DOCTYPE/entity declarations are not allowed")
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    return etree.fromstring(data, parser)


def _sub(parent, uri_tag: str, prefix: str):
    """SubElement that binds a readable prefix when the document has none for it."""
    uri = uri_tag[1:uri_tag.index("}")]
    nsmap = None if uri in parent.nsmap.values() else {prefix: uri}
    return etree.SubElement(parent, uri_tag, nsmap=nsmap)


def get_on_disk_rating(photo_path: Path | str) -> int | None:
    """xmp:Rating currently on disk for this photo, or None."""
    sc = _writer_sidecar(Path(photo_path))
    if sc is None or not sc.exists():
        return None
    try:
        info = parse_xmp(sc.read_bytes())
    except Exception:
        return None
    return info.rating if info else None


def update_xmp_sidecar(photo_path: Path | str, *, rating: int | None = None,
                       add_tags: tuple[str, ...] = (), remove_tags: tuple[str, ...] = (),
                       label: str | None = None) -> tuple[bool, str]:
    """Modify only xmp:Rating, xmp:Label and photocure-* dc:subject tags.

    Returns (ok, path or reason). Unparsable or ambiguous sidecars are left untouched.
    """
    photo = Path(photo_path)
    sidecar = _writer_sidecar(photo)
    if sidecar is None:
        return False, "sidecar_skipped_dual"
    if sidecar.exists():
        try:
            root = _safe_parse(sidecar)
        except Exception as e:
            log.warning("sidecar_skipped (%s): %s", sidecar, e)
            return False, "sidecar_skipped"
        desc = root.find(f".//{_RDF}Description")
        if desc is None:
            return False, "sidecar_skipped"
    else:
        root = etree.Element(f"{_RDF}RDF", nsmap={"rdf": _RDF[1:-1]})
        desc = etree.SubElement(root, f"{_RDF}Description", nsmap={"xmp": _XMP[1:-1], "dc": _DC[1:-1]})
        desc.set(f"{_RDF}about", "")

    if rating is not None:
        if f"{_XMP}Rating" in desc.attrib:
            desc.set(f"{_XMP}Rating", str(rating))
        else:
            el = desc.find(f"{_XMP}Rating")
            if el is None:
                el = _sub(desc, f"{_XMP}Rating", "xmp")
            el.text = str(rating)

    add = {t for t in add_tags if t.startswith(OWNED_TAG_PREFIX)}
    rem = {t for t in remove_tags if t.startswith(OWNED_TAG_PREFIX)}
    if add or rem:
        subject = desc.find(f"{_DC}subject")
        if subject is None:
            subject = _sub(desc, f"{_DC}subject", "dc")
        bag = subject.find(f"{_RDF}Bag")
        if bag is None:
            bag = etree.SubElement(subject, f"{_RDF}Bag")
        have = set()
        for li in list(bag.findall(f"{_RDF}li")):
            if li.text in rem:
                bag.remove(li)
            else:
                have.add(li.text)
        for t in sorted(add - have):
            etree.SubElement(bag, f"{_RDF}li").text = t

    if label is not None:
        el = desc.find(f"{_XMP}Label")
        if label:
            if el is None:
                el = _sub(desc, f"{_XMP}Label", "xmp")
            el.text = label
        elif el is not None:
            desc.remove(el)
        if f"{_XMP}Label" in desc.attrib and not label:
            del desc.attrib[f"{_XMP}Label"]

    tmp = sidecar.with_name(f"{sidecar.name}.tmp.{os.getpid()}")
    try:
        tmp.write_bytes(etree.tostring(root, encoding="UTF-8", xml_declaration=True))
        os.replace(tmp, sidecar)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        log.warning("sidecar write failed (%s): %s", sidecar, e)
        return False, "write_failed"
    return True, str(sidecar)
