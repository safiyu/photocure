"""C2 acceptance: first-run foreign rating, edited-elsewhere, explicit override."""
from lxml import etree

from photocure.db.models import ImageRecord
from photocure.scanner.xmp import get_on_disk_rating, update_xmp_sidecar
from photocure.sidecar_sync import (EDITED_ELSEWHERE, EXISTING_FOREIGN, NEEDS_CONFIRMATION,
                                    SKIPPED_DUAL, WRITTEN, sync_sidecar)

LR = ('<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
      'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
      'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">'
      '<rdf:Description rdf:about="" xmp:Rating="4" crs:Exposure2012="+0.35"/></rdf:RDF></x:xmpmeta>')


def _img(tmp_path, sidecar=None):
    photo = tmp_path / "IMG_1.JPG"
    photo.write_bytes(b"x")
    if sidecar:
        (tmp_path / "IMG_1.JPG.xmp").write_text(sidecar)
    return ImageRecord(path=str(photo), filename="IMG_1.JPG")


def test_first_run_foreign_rating_is_preserved(tmp_path):
    img = _img(tmp_path, LR)
    r = sync_sidecar(img, rating=2, tags=("photocure-best-shot",))
    assert r.status == EXISTING_FOREIGN
    assert get_on_disk_rating(img.path) == 4  # untouched
    assert img.xmp_written_rating is None
    root = etree.parse(str(tmp_path / "IMG_1.JPG.xmp")).getroot()
    assert "+0.35" in etree.tostring(root).decode()  # develop settings intact


def test_explicit_action_needs_confirmation_then_overrides(tmp_path):
    img = _img(tmp_path, LR)
    assert sync_sidecar(img, rating=5, explicit=True).status == NEEDS_CONFIRMATION
    assert get_on_disk_rating(img.path) == 4
    r = sync_sidecar(img, rating=5, explicit=True, overwrite_foreign=True)
    assert r.status == WRITTEN and get_on_disk_rating(img.path) == 5
    assert img.xmp_written_rating == 5


def test_automated_pass_skips_sidecar_edited_elsewhere(tmp_path):
    img = _img(tmp_path)
    assert sync_sidecar(img, rating=3, explicit=True).status == WRITTEN
    update_xmp_sidecar(img.path, rating=1)  # another tool changes it
    r = sync_sidecar(img, rating=5)
    assert r.status == EDITED_ELSEWHERE and get_on_disk_rating(img.path) == 1


def test_own_writes_are_not_mistaken_for_foreign_edits(tmp_path):
    img = _img(tmp_path)
    assert sync_sidecar(img, rating=3, tags=("photocure-blink",)).status == WRITTEN
    assert sync_sidecar(img, rating=4, tags=("photocure-best-shot",)).status == WRITTEN
    assert get_on_disk_rating(img.path) == 4 and img.xmp_written_tags == "photocure-best-shot"


def test_dual_sidecar_is_reported_and_untouched(tmp_path):
    img = _img(tmp_path, LR)
    (tmp_path / "IMG_1.xmp").write_text(LR)
    assert sync_sidecar(img, rating=2, explicit=True, overwrite_foreign=True).status == SKIPPED_DUAL


RICH = ('<?xml version="1.0" encoding="UTF-8"?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:lr="http://ns.adobe.com/lightroom/1.0/" '
        'xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" xmlns:acme="urn:acme:custom" '
        'xmp:Rating="2" crs:Exposure2012="+0.35" photoshop:City="Rabat" acme:Thing="keep me">'
        '<dc:subject><rdf:Bag><rdf:li>holiday</rdf:li><rdf:li>photocure-blink</rdf:li></rdf:Bag></dc:subject>'
        '<lr:hierarchicalSubject><rdf:Bag><rdf:li>trips|morocco</rdf:li></rdf:Bag></lr:hierarchicalSubject>'
        '<crs:ToneCurvePV2012><rdf:Seq><rdf:li>0, 0</rdf:li><rdf:li>255, 255</rdf:li></rdf:Seq></crs:ToneCurvePV2012>'
        '</rdf:Description></rdf:RDF></x:xmpmeta>')


def _not_ours(path):
    """Canonical form of the sidecar with everything Photocure owns removed."""
    root = etree.fromstring((path).read_bytes())
    x, d, r = ("{http://ns.adobe.com/xap/1.0/}", "{http://purl.org/dc/elements/1.1/}",
               "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}")
    for desc in root.iter(r + "Description"):
        desc.attrib.pop(x + "Rating", None)
        desc.attrib.pop(x + "Label", None)
        for tag in ("Rating", "Label"):
            for el in desc.findall(x + tag):
                desc.remove(el)
        for subj in desc.findall(d + "subject"):
            for li in list(subj.iter(r + "li")):
                if (li.text or "").startswith("photocure-"):
                    li.getparent().remove(li)
    return etree.tostring(root, method="c14n")


def test_everything_we_do_not_own_survives_unchanged(tmp_path):
    img = _img(tmp_path)
    side = tmp_path / "IMG_1.JPG.xmp"
    side.write_text(RICH)
    before = _not_ours(side)
    update_xmp_sidecar(img.path, rating=5, add_tags=("photocure-best-shot",),
                       remove_tags=("photocure-blink",), label="Red")
    assert _not_ours(side) == before
    assert get_on_disk_rating(img.path) == 5
