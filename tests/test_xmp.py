"""XMP sidecar parsing and its integration with the date cascade."""
from __future__ import annotations

import datetime as dt

from PIL import Image

from photocure.scanner.exif_reader import extract_exif
from photocure.scanner.xmp import find_sidecar, parse_xmp, parse_xmp_date, read_xmp

_ATTRS = """<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/"
 xmlns:exif="http://ns.adobe.com/exif/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmp:Rating="5" xmp:Label="Red" exif:DateTimeOriginal="2023-07-04T10:11:12.50">
<dc:subject><rdf:Bag><rdf:li>beach</rdf:li><rdf:li>family</rdf:li></rdf:Bag></dc:subject>
</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>"""

_CHILDREN = """<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:exif="http://ns.adobe.com/exif/1.0/">
<xmp:Rating>-1</xmp:Rating><exif:DateTimeOriginal>2020-01-02T03:04:05Z</exif:DateTimeOriginal>
</rdf:Description></rdf:RDF></x:xmpmeta>"""


def test_parse_attributes_and_keywords():
    x = parse_xmp(_ATTRS.encode())
    assert x.rating == 5 and x.label == "Red"
    assert x.taken_at == dt.datetime(2023, 7, 4, 10, 11, 12)
    assert x.tz_offset_minutes is None
    assert x.keywords == ["beach", "family"]


def test_parse_child_elements_and_rejected_rating():
    x = parse_xmp(_CHILDREN.encode())
    assert x.rating == -1
    assert x.taken_at == dt.datetime(2020, 1, 2, 3, 4, 5)
    assert x.tz_offset_minutes == 0


def test_rating_is_clamped():
    data = _ATTRS.replace('xmp:Rating="5"', 'xmp:Rating="9"').encode()
    assert parse_xmp(data).rating == 5


def test_parse_xmp_date_offsets():
    got = parse_xmp_date("2024-05-12T15:30:45+05:30")
    assert got == (dt.datetime(2024, 5, 12, 15, 30, 45), 330)
    assert parse_xmp_date("2024-05-12") == (dt.datetime(2024, 5, 12), None)
    assert parse_xmp_date("garbage") == (None, None)
    assert parse_xmp_date("2024-13-45T00:00:00") == (None, None)


def test_hostile_or_broken_xml_is_ignored():
    bomb = b'<!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>'
    assert parse_xmp(bomb).rating is None
    assert parse_xmp(b"<not-xml").taken_at is None
    assert parse_xmp(b"x" * (3 * 1024 * 1024)).rating is None


def test_sidecar_lookup_variants(tmp_path):
    img = tmp_path / "IMG_1.CR2"
    img.write_bytes(b"raw")
    assert find_sidecar(img) is None and read_xmp(img) is None
    (tmp_path / "IMG_1.CR2.xmp").write_text(_CHILDREN)
    assert find_sidecar(img).name == "IMG_1.CR2.xmp"
    (tmp_path / "IMG_1.xmp").write_text(_ATTRS)
    assert find_sidecar(img).name == "IMG_1.xmp"  # Lightroom style wins
    assert read_xmp(img).rating == 5


def test_xmp_date_used_only_when_exif_has_none(tmp_path):
    p = tmp_path / "IMG_20200101_000000.jpg"
    Image.new("RGB", (8, 8)).save(p)
    p.with_suffix(".xmp").write_text(_ATTRS)
    info = extract_exif(p)
    assert info.taken_at == dt.datetime(2023, 7, 4, 10, 11, 12)
    assert info.taken_at_source == "xmp:dtoriginal"
    assert info.rating == 5 and info.xmp_label == "Red"
    assert info.xmp_keywords == ["beach", "family"]


def test_exif_date_beats_xmp_but_rating_still_read(tmp_path):
    p = tmp_path / "a.jpg"
    ex = Image.Exif()
    ex.get_ifd(0x8769)[36867] = "2019:02:03 04:05:06"
    Image.new("RGB", (8, 8)).save(p, exif=ex)
    p.with_suffix(".xmp").write_text(_ATTRS)
    info = extract_exif(p)
    assert info.taken_at == dt.datetime(2019, 2, 3, 4, 5, 6)
    assert info.taken_at_source == "exif:dtoriginal"
    assert info.rating == 5
