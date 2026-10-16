"""Non-destructive XMP sidecar writer (lxml-based, element-preserving)."""
from __future__ import annotations

import logging
from pathlib import Path

from lxml import etree

log = logging.getLogger("photocure.scanner.xmp")

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def _find_sidecar_path(photo_path: Path) -> Path | None:
    """Resolve which sidecar to use or create (per spec §3.1).
    
    Rules:
    1. If only <filename>.<ext>.xmp exists: use it.
    2. If only <filename>.xmp (legacy) exists: use it.
    3. If neither exists: create <filename>.<ext>.xmp.
    4. If both exist: return None (ambiguous).
    """
    photo = Path(photo_path)
    ext_sidecar = photo.with_suffix(photo.suffix + ".xmp")
    base_sidecar = photo.with_suffix(".xmp")
    ext_exists = ext_sidecar.exists()
    base_exists = base_sidecar.exists()
    if ext_exists and base_exists:
        return None
    if base_exists:
        return base_sidecar
    return ext_sidecar


def update_xmp_sidecar(
    photo_path: Path | str,
    *,
    rating: int | None = None,
    add_tags: tuple[str, ...] = (),
    remove_tags: tuple[str, ...] = (),
    label: str | None = None,
) -> tuple[bool, str]:
    """Update XMP sidecar non-destructively.
    
    Returns (success, message).
    """
    photo = Path(photo_path)
    sidecar = _find_sidecar_path(photo)
    if sidecar is None:
        return (False, f"dual sidecar: both .xmp and .{photo.suffix}.xmp exist")

    creating_new = not sidecar.exists()
    if creating_new:
        root = etree.Element(
            "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF",
            nsmap={None: NS["rdf"], "xmp": NS["xmp"], "dc": NS["dc"]}
        )
        desc = etree.SubElement(root, "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
        desc.set("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about", "")
    else:
        try:
            tree = etree.parse(str(sidecar))
            root = tree.getroot()
            desc = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
            if desc is None:
                return (False, f"corrupt XMP: no rdf:Description in {sidecar}")
        except Exception as e:
            return (False, f"parse error: {e}")

    if rating is not None:
        rating_elem = desc.find("{http://ns.adobe.com/xap/1.0/}Rating")
        if rating_elem is None:
            rating_elem = etree.SubElement(desc, "{http://ns.adobe.com/xap/1.0/}Rating")
        rating_elem.text = str(rating)

    photocure_tags = set(t for t in (add_tags or ()) if t.startswith("photocure-"))
    remove_photocure = set(t for t in (remove_tags or ()) if t.startswith("photocure-"))
    existing_photocure = set()
    subject_bag = desc.find("{http://purl.org/dc/elements/1.1/}subject")
    
    if subject_bag is None and (photocure_tags or remove_photocure):
        subject_bag = etree.SubElement(desc, "{http://purl.org/dc/elements/1.1/}subject")
        subject_bag.set("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}parseType", "Bag")

    if subject_bag is not None:
        for li in subject_bag.findall("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li"):
            tag_text = li.text or ""
            if tag_text.startswith("photocure-"):
                existing_photocure.add(tag_text)
        for li in list(subject_bag.findall("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li")):
            if li.text in remove_photocure:
                subject_bag.remove(li)
                existing_photocure.discard(li.text)
        for tag in photocure_tags:
            if tag not in existing_photocure:
                li = etree.SubElement(
                    subject_bag, "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li"
                )
                li.text = tag

    if label is not None:
        label_elem = desc.find("{http://ns.adobe.com/xap/1.0/}Label")
        if label:
            if label_elem is None:
                label_elem = etree.SubElement(desc, "{http://ns.adobe.com/xap/1.0/}Label")
            label_elem.text = label
        else:
            if label_elem is not None:
                desc.remove(label_elem)

    try:
        tmp_path = sidecar.with_name(sidecar.name + f".tmp.{id(etree)}")
        tmp_path.write_bytes(
            etree.tostring(root, encoding="UTF-8", xml_declaration=True, pretty_print=True)
        )
        tmp_path.replace(sidecar)
        return (True, str(sidecar))
    except OSError as e:
        log.warning(f"Failed to write {sidecar}: {e}")
        return (False, f"write failed: {e}")
