"""Clean export: PII scanning, blurring, metadata-free bundles."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from photocure import clean_export as ce
from photocure.clean_export import (
    ExportItem,
    Finding,
    PiiScanner,
    PiiUnavailable,
    Region,
    blur_regions,
    export_bundle,
    find_text_boxes,
    verify_clean,
)

LABELS = list(ce.PII_PROMPTS)
BENIGN_AXIS = 10


class FakeEngine:
    def __init__(self, decide=None, available=True):
        self.decide, self.available, self.released = decide or (lambda im: None), available, 0

    def text_search_available(self):
        return self.available

    def encode_queries(self, texts):
        by_text = {t: LABELS.index(lab) for lab, ts in ce.PII_PROMPTS.items() for t in ts}
        out = np.zeros((len(texts), 16), np.float32)
        for i, t in enumerate(texts):
            out[i, by_text.get(t, BENIGN_AXIS)] = 1.0
        return out

    def classify_and_embed(self, images):
        res = []
        for im in images:
            lab = self.decide(im)
            v = np.zeros(16, np.float32)
            v[LABELS.index(lab) if lab else BENIGN_AXIS] = 1.0
            res.append(SimpleNamespace(embedding=v.tolist()))
        return res

    def release(self):
        self.released += 1


def _bright_fraction(im):
    return float((cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) > 200).mean())


def _scene(size=600):
    return np.full((size, size, 3), 100, np.uint8)


# ------------------------------------------------------------------ scanner
def test_unavailable_model_raises_and_releases():
    eng = FakeEngine(available=False)
    with pytest.raises(PiiUnavailable):
        PiiScanner(engine=eng)
    assert eng.released == 1


def test_flagged_tile_becomes_a_region_inside_that_tile():
    img = _scene()
    img[430:580, 430:580] = 255  # square patch inside the bottom-right third
    eng = FakeEngine(lambda im: "license_plate" if _bright_fraction(im) > 0.3 else None)
    f = PiiScanner(engine=eng).scan_image(7, img)
    assert f.image_id == 7 and f.labels == ["license_plate"] and f.localized
    assert len(f.regions) == 1
    r = f.regions[0]
    assert r.x1 >= 0.66 and r.y1 >= 0.66 and r.x2 <= 1.0 and r.y2 <= 1.0
    assert r.x1 <= 430 / 600 and r.x2 >= 580 / 600


def test_text_refinement_tightens_region_to_the_plate():
    img = _scene()
    img[480:540, 410:590] = 255
    cv2.putText(img, "AB 123", (430, 525), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
    eng = FakeEngine(lambda im: "license_plate" if _bright_fraction(im) > 0.2 else None)
    f = PiiScanner(engine=eng).scan_image(1, img)
    assert f.localized and f.regions
    tile_area = (1 / 3) ** 2
    for r in f.regions:
        assert (r.x2 - r.x1) * (r.y2 - r.y1) < 0.6 * tile_area
    assert any(r.x1 <= 430 / 600 and r.x2 >= 570 / 600 and r.y1 <= 490 / 600 and r.y2 >= 530 / 600
               for r in f.regions)


def test_whole_only_flag_is_not_localized():
    img = _scene()
    eng = FakeEngine(lambda im: "id_document" if im.shape[:2] == (600, 600) else None)
    f = PiiScanner(engine=eng).scan_image(2, img)
    assert f.labels == ["id_document"] and f.localized is False
    assert f.regions and f.regions[0].label == "id_document"


def test_label_flagged_only_on_the_whole_photo_still_gets_a_region():
    img = _scene()
    img[430:580, 430:580] = 255

    def decide(im):
        if im.shape[:2] == (600, 600):
            return "id_document"
        return "license_plate" if _bright_fraction(im) > 0.3 else None

    f = PiiScanner(engine=FakeEngine(decide)).scan_image(3, img)
    assert f.labels == ["id_document", "license_plate"] and f.localized is False
    assert {r.label for r in f.regions} == {"id_document", "license_plate"}


def test_thin_strips_are_not_text_boxes():
    img = np.full((600, 800, 3), 90, np.uint8)
    img[300:302, 100:700] = 255
    assert find_text_boxes(img) == []


def test_clean_photo_has_no_finding_and_release_frees_engine():
    eng = FakeEngine()
    sc = PiiScanner(engine=eng)
    assert sc.scan_image(1, _scene()) is None
    assert sc.scan_image(2, np.zeros((10, 10, 3), np.uint8)) is None
    sc.release()
    assert eng.released == 1


def test_find_text_boxes_on_synthetic_plate():
    img = np.full((300, 600, 3), 80, np.uint8)
    cv2.rectangle(img, (100, 100), (500, 200), (255, 255, 255), -1)
    cv2.putText(img, "ABC 123", (130, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5)
    boxes = find_text_boxes(img)
    assert any(x1 <= 130 and x2 >= 470 and y1 <= 110 and y2 >= 190 for x1, y1, x2, y2 in boxes)
    assert find_text_boxes(np.full((300, 600, 3), 80, np.uint8)) == []


def test_merge_regions_only_joins_same_label_overlaps():
    a, b = Region(0, 0, .5, .5, "screen"), Region(.4, .4, .9, .9, "screen")
    c = Region(.45, .45, .6, .6, "license_plate")
    out = ce.merge_regions([a, b, c])
    assert len(out) == 2
    assert any(r.label == "screen" and r.x1 == 0 and r.x2 == .9 for r in out)


# ------------------------------------------------------------------ blur / encode / verify
def _noise(h=400, w=400, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_blur_hides_region_and_leaves_the_rest_untouched():
    img = _noise()
    out = blur_regions(img, [Region(0.25, 0.25, 0.5, 0.5, "screen")])
    lap = lambda a: cv2.Laplacian(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()  # noqa: E731
    assert lap(out[110:190, 110:190]) < 0.02 * lap(img[110:190, 110:190])
    assert np.array_equal(out[:80], img[:80]) and np.array_equal(out[:, 320:], img[:, 320:])
    assert out is not img and out.shape == img.shape


def _exif_jpeg(path):
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "ACME", "Cam 9"
    exif[0xA431] = "SERIAL-1234567"
    exif[0x8825] = {1: "N", 2: (48.0, 51.0, 24.0), 3: "E", 4: (2.0, 21.0, 7.0)}
    Image.fromarray(_noise(120, 160)[:, :, ::-1]).save(path, exif=exif, quality=90)


def test_export_strips_metadata_and_never_touches_the_original(tmp_path):
    src = tmp_path / "IMG_secret.jpg"
    _exif_jpeg(src)
    before = src.read_bytes()
    assert verify_clean(src) and b"SERIAL-1234567" in before
    res = export_bundle([ExportItem(1, src)], tmp_path / "out")
    out = res.folder / "images" / "photo-0001.jpg"
    assert res.exported == 1 and not res.failures
    assert verify_clean(out) == []
    data = out.read_bytes()
    assert b"SERIAL-1234567" not in data and b"ACME" not in data
    assert src.read_bytes() == before


def _items(n=3, **kw):
    return [ExportItem(i, f"/home/alice/secret/IMG_{i}.jpg", taken_at=f"2024-05-0{i}", **kw)
            for i in range(1, n + 1)]


def _solid_loader(path, orientation):
    i = int(str(path).rsplit("_", 1)[1].split(".")[0])
    img = _noise(160, 200, seed=i)
    img[:8] = i * 60
    return img


def test_policies(tmp_path):
    items = _items()
    fnd = {2: Finding(2, ["screen"], 0.9, [Region(0.2, 0.2, 0.8, 0.8, "screen")])}
    skip = export_bundle(items, tmp_path / "a", findings=fnd, policy="skip", loader=_solid_loader)
    assert (skip.exported, skip.skipped_flagged, skip.blurred, skip.flagged) == (2, 1, 0, 1)
    blur = export_bundle(items, tmp_path / "b", findings=fnd, policy="blur", loader=_solid_loader)
    assert (blur.exported, blur.skipped_flagged, blur.blurred) == (3, 0, 1)
    inc = export_bundle(items, tmp_path / "c", findings=fnd, policy="include", loader=_solid_loader)
    assert (inc.exported, inc.blurred, inc.flagged) == (3, 0, 1)
    mid_blur = cv2.imread(str(blur.folder / "images" / "photo-0002.jpg"))
    mid_inc = cv2.imread(str(inc.folder / "images" / "photo-0002.jpg"))
    assert mid_blur[60:100, 60:140].std() < 0.2 * mid_inc[60:100, 60:140].std()
    with pytest.raises(ValueError):
        export_bundle(items, tmp_path / "d", policy="nope", loader=_solid_loader)


def test_manifest_hashes_order_and_no_leaks(tmp_path):
    items = list(reversed(_items()))  # given out of capture order
    res = export_bundle(items, tmp_path / "o", loader=_solid_loader)
    man = json.loads(res.manifest.read_text())
    names = [f["name"] for f in man["files"]]
    assert names == ["photo-0001.jpg", "photo-0002.jpg", "photo-0003.jpg"]
    for f in man["files"]:
        data = (res.folder / "images" / f["name"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == f["sha256"] and f["bytes"] == len(data)
        assert f["verified_clean"] is True
    first = cv2.imread(str(res.folder / "images" / "photo-0001.jpg"))
    assert abs(int(first[2:6, 20:180].mean()) - 60) < 8  # earliest capture is IMG_1
    text = res.manifest.read_text() + (res.folder / "index.html").read_text()
    for leak in ("alice", "secret", "IMG_1", "/home"):
        assert leak not in text


def test_html_escapes_hostile_names_and_makes_no_external_requests(tmp_path):
    hostile = ExportItem(1, "/x/<script>alert(1)</script>.jpg", taken_at="2024-01-01")
    res = export_bundle([hostile], tmp_path / "o", rename=False, loader=lambda p, o: _noise(80, 80))
    page = (res.folder / "index.html").read_text()
    assert "<script" not in page.lower() and "http://" not in page and "https://" not in page
    assert res.exported == 1 and len(list((res.folder / "images").iterdir())) == 1
    safe = export_bundle(_items(1), tmp_path / "p", loader=_solid_loader)
    assert "http://" not in (safe.folder / "index.html").read_text()


def test_no_html_when_disabled(tmp_path):
    res = export_bundle(_items(1), tmp_path / "o", html=False, loader=_solid_loader)
    assert not (res.folder / "index.html").exists() and res.manifest.exists()


def test_new_folder_never_reuses_an_existing_one(tmp_path):
    root = tmp_path / "out"
    (root / "share").mkdir(parents=True)
    keep = root / "share" / "mine.txt"
    keep.write_text("keep")
    a = export_bundle(_items(1), root, bundle_name="share", loader=_solid_loader)
    b = export_bundle(_items(1), root, bundle_name="share", loader=_solid_loader)
    assert len({a.folder, b.folder, root / "share"}) == 3
    assert keep.read_text() == "keep" and list((root / "share").iterdir()) == [keep]
    c = export_bundle(_items(1), root, bundle_name="../../escape", loader=_solid_loader)
    assert c.folder.parent == root


def test_cancel_stops_early_but_keeps_written_files(tmp_path):
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 1

    seen = []
    res = export_bundle(_items(3), tmp_path / "o", loader=_solid_loader, cancel_check=cancel,
                        on_progress=lambda d, t, n: seen.append((d, t, n)))
    assert res.exported == 1 and seen == [(1, 3, "photo-0001.jpg")]
    assert len(json.loads(res.manifest.read_text())["files"]) == 1


def test_loader_failure_is_reported_not_raised(tmp_path):
    def loader(path, orientation):
        if "IMG_2" in str(path):
            raise OSError("boom")
        return _solid_loader(path, orientation)

    res = export_bundle(_items(3), tmp_path / "o", loader=loader)
    assert res.exported == 2 and len(res.failures) == 1 and "boom" in res.failures[0]
