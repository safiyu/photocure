"""The MCP server is read-only and answers from the library database."""
from __future__ import annotations

import types
from datetime import datetime, timezone

import pytest

from photocure.db import models as M
from photocure.db.session import get_session
from photocure.mcp_server import LibraryQueries

from .world import build_world


@pytest.fixture()
def q(tmp_path):
    w = build_world(tmp_path, n=3)
    with get_session(w.db) as s:
        for i, (cat, day) in enumerate([("utility", 5), ("food", 6), ("utility", 20)]):
            r = s.get(M.ImageRecord, w.ids[i])
            r.manual_bucket = cat
            r.taken_at = datetime(2026, 9, day, 12, tzinfo=timezone.utc)
        s.commit()
    return types.SimpleNamespace(w=w, q=LibraryQueries(w.cfg))


def test_overview_counts_categories(q):
    o = q.q.overview()
    assert o["photos"] == 3 and o["by_category"]["utility"] == 2 and o["by_category"]["food"] == 1
    assert o["earliest"].startswith("2026-09-05") and o["latest"].startswith("2026-09-20")


def test_list_filters_by_date_and_category(q):
    r = q.q.list_photos(date_from="2026-09-01", date_to="2026-09-10", category="utility")
    assert r["total"] == 1 and r["items"][0]["id"] == q.w.ids[0]
    assert q.q.list_photos(date_from="2026-09-20", date_to="2026-09-20")["total"] == 1
    assert q.q.list_photos()["items"][0]["id"] == q.w.ids[2]  # newest first


def test_bad_inputs_are_clear_errors(q):
    with pytest.raises(ValueError, match="ISO date"):
        q.q.list_photos(date_from="last month")
    with pytest.raises(ValueError, match="unknown category"):
        q.q.list_photos(category="routers")
    with pytest.raises(ValueError, match="no photo"):
        q.q.info(99999)
    with pytest.raises(ValueError, match="size"):
        q.q.image_bytes(q.w.ids[0], "huge")


def test_info_and_image_come_from_the_row(q):
    info = q.q.info(q.w.ids[0])
    assert info["category"] == "utility" and info["file_exists"]
    data, fmt = q.q.image_bytes(q.w.ids[0], "medium")
    assert fmt == "webp" and data[:4] == b"RIFF"


def test_missing_file_is_reported(q):
    import os
    os.remove(q.q.info(q.w.ids[1])["path"])
    with pytest.raises(FileNotFoundError):
        q.q.image_bytes(q.w.ids[1])


def test_server_registers_read_only_tools(q):
    pytest.importorskip("mcp")
    import asyncio
    from photocure.mcp_server import build_server
    server = build_server(q.q)
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"library_overview", "search_photos", "list_photos", "photo_info", "view_photo",
                     "list_bursts", "create_composite", "clean_export", "list_people", "person_photos"}
    by_name = {t.name: t for t in tools}
    assert by_name["list_bursts"].annotations.read_only_hint is True
    for n in ("create_composite", "clean_export"):
        assert by_name[n].annotations.read_only_hint is False
        assert by_name[n].annotations.destructive_hint is False


def test_search_ranks_by_embedding_and_respects_filters(q):
    import numpy as np
    from photocure.db.repository import quantize_embedding
    dims = 512
    vecs = []
    for i in range(3):
        v = np.zeros(dims, dtype=np.float32)
        v[i] = 1.0
        vecs.append(v)
    with get_session(q.w.db) as s:
        for i, v in enumerate(vecs):
            s.add(M.SemanticScore(
                image_id=q.w.ids[i], model_name="m", backend="clip", top1_category="utility",
                top1_score=0.9, top2_category="food", top2_score=0.1, score_margin=0.8,
                distribution="{}", embedding=quantize_embedding(v.tolist())))
        s.commit()

    class FakeEncoder:
        def __init__(self, v): self.v = v
        def encode(self, text): return self.v
        def close(self): pass

    # A query aimed at photo 0 (5 Sep) wins unfiltered and drops out of later date ranges.
    q.q._encoder = FakeEncoder(vecs[0])
    top = q.q.search("router label")["items"]
    assert top[0]["id"] == q.w.ids[0] and top[0]["score"] > 0.9
    assert q.w.ids[0] not in [i["id"] for i in q.q.search("x", date_from="2026-09-10")["items"]]
    only_food = q.q.search("x", category="food")["items"]
    assert [i["id"] for i in only_food] == [q.w.ids[1]]


# ------------------------------------------------------ bursts and actions
def _group_world(tmp_path):
    """Three real frames: two people, complementary blinks, frame 2 slightly shifted."""
    import cv2
    import numpy as np
    from .test_composite import H, W, _face, _texture
    w = build_world(tmp_path, n=3)
    tex = _texture(7)
    spots = [(300, 330), (600, 330)]

    def frame(states):
        img = tex.copy()
        for c, op in zip(spots, states):
            _face(img, c, op)
        return img

    base, other, both_closed = frame([False, True]), frame([True, False]), frame([False, False])
    m = cv2.getRotationMatrix2D((W / 2, H / 2), 1.5, 1.0)
    m[:, 2] += (9, -6)
    other = cv2.warpAffine(other, m, (W, H), borderMode=cv2.BORDER_REFLECT)
    moved = [np.array([m @ np.array([x, y, 1.0]) for x, y in spots])]
    states = [[("closed"), ("open")], ["open", "closed"], ["closed", "closed"]]
    imgs = [base, other, both_closed]
    with get_session(w.db) as s:
        bg = M.BurstGroup(name="Burst_g", image_count=3, best_shot_id=w.ids[0])
        s.add(bg)
        s.flush()
        for i, iid in enumerate(w.ids):
            rec = s.get(M.ImageRecord, iid)
            cv2.imwrite(rec.path, imgs[i])
            rec.burst_id, rec.is_best_shot, rec.is_alternate = bg.id, i == 0, i != 0
            rec.taken_at = datetime(2026, 9, 12, 12, 0, i, tzinfo=timezone.utc)
            rec.has_blinks = "closed" in states[i]
            centres = spots if i != 1 else [tuple(p) for p in moved[0]]
            for (cx, cy), st in zip(centres, states[i]):
                s.add(M.FaceDetection(
                    image_id=iid, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1, confidence=0.9,
                    x1=(cx - 70) / W, y1=(cy - 70) / H, x2=(cx + 70) / W, y2=(cy + 70) / H,
                    eye_status=st, eye_closed_score=0.9 if st == "closed" else 0.1))
        s.commit()
        bid = bg.id
    return w, bid


def test_list_bursts_finds_fixable_ones(tmp_path):
    w, bid = _group_world(tmp_path)
    qq = LibraryQueries(w.cfg)
    items = qq.list_bursts(only_fixable=True)["items"]
    assert [b["id"] for b in items] == [bid] and items[0]["can_composite"]
    assert items[0]["frames_with_blinks"] == 3 and items[0]["partner_id"] in w.ids
    assert qq.list_bursts(date_from="2027-01-01")["items"] == []
    info = qq.info(w.ids[0])
    assert info["burst_id"] == bid and info["faces"] == ["closed", "open"] and info["has_blinks"]


def test_create_composite_writes_new_files_only(tmp_path):
    from pathlib import Path
    w, bid = _group_world(tmp_path)
    qq = LibraryQueries(w.cfg)
    before = {i: Path(qq.info(i)["path"]).read_bytes() for i in w.ids}
    r = qq.composite(bid, "fixed")
    assert r["applied"] >= 1 and Path(r["path"]).name.endswith("_fixed.jpg") and Path(r["path"]).exists()
    r2 = qq.composite(bid, "layered")
    assert Path(r2["path"]).suffix == ".psd" and Path(r2["path"]).stat().st_size > 1000
    assert {i: Path(qq.info(i)["path"]).read_bytes() for i in w.ids} == before  # originals untouched
    with pytest.raises(ValueError, match="mode"):
        qq.composite(bid, "paint")
    with pytest.raises(ValueError, match="not found"):
        qq.composite(99999)


def test_composite_refuses_a_burst_that_is_not_fixable(q):
    with get_session(q.w.db) as s:
        bg = M.BurstGroup(name="Burst_plain", image_count=2, best_shot_id=q.w.ids[0])
        s.add(bg)
        s.flush()
        for iid in q.w.ids[:2]:
            s.get(M.ImageRecord, iid).burst_id = bg.id
        s.commit()
        bid = bg.id
    with pytest.raises(ValueError, match="combined"):
        q.q.composite(bid)


def test_clean_export_policies_and_refusals(q, tmp_path, monkeypatch):
    from pathlib import Path
    from photocure import clean_export as ce
    ids = q.w.ids[:2]
    out = tmp_path / "share"
    originals = {i: Path(q.q.info(i)["path"]).read_bytes() for i in ids}

    class Boom:
        def __init__(self, *a, **k):
            raise ce.PiiUnavailable("needs the AI model")
    monkeypatch.setattr(ce, "PiiScanner", Boom)
    with pytest.raises(ValueError, match="include"):
        q.q.clean_export(ids, str(out), "blur")
    with pytest.raises(ValueError, match="library"):
        q.q.clean_export(ids, str(Path(q.q.info(ids[0])["path"]).parent / "x"), "include")
    with pytest.raises(ValueError, match="absolute"):
        q.q.clean_export(ids, "relative/out", "include")
    r = q.q.clean_export(ids, str(out), "include")
    assert r["exported"] == 2 and not r["failures"]
    assert all(ce.verify_clean(f) == [] for f in (Path(r["folder"]) / "images").glob("*.jpg"))

    class Fake:
        def __init__(self, *a, **k): pass
        def scan_image(self, iid, bgr):
            if iid != ids[0]:
                return None
            return ce.Finding(iid, ["license_plate"], 0.9, [ce.Region(0.2, 0.2, 0.6, 0.5, "license_plate")], True)
        def release(self): self.released = True
    monkeypatch.setattr(ce, "PiiScanner", Fake)
    r = q.q.clean_export(ids, str(out), "skip")
    assert r["exported"] == 1 and r["skipped_flagged"] == 1
    assert r["flagged_photos"][0]["labels"] == ["license_plate"]
    r = q.q.clean_export(ids, str(out), "blur")
    assert r["exported"] == 2 and r["blurred"] == 1
    assert {i: Path(q.q.info(i)["path"]).read_bytes() for i in ids} == originals


# ------------------------------------------------------------------- people
def test_list_people_tool_reports_named_clusters(q):
    with get_session(q.w.db) as s:
        s.add(M.PersonCluster(name="Safi", face_count=1))
    items = q.q.list_people()["items"]
    assert len(items) == 1
    assert items[0]["name"] == "Safi" and items[0]["face_count"] == 1


def test_person_photos_returns_date_town_and_filename(q):
    iid = q.w.ids[0]
    with get_session(q.w.db) as s:
        rec = s.get(M.ImageRecord, iid)
        rec.taken_at = datetime(2026, 6, 1, 9, tzinfo=timezone.utc)
        rec.city, rec.region, rec.country_code = "Casablanca", "Casablanca-Settat", "MA"
        p = M.PersonCluster(name="Safi", face_count=1)
        s.add(p)
        s.flush()
        s.add(M.FaceDetection(image_id=iid, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1,
                              confidence=0.9, person_id=p.id))
    filename = q.q.info(iid)["filename"]

    r = q.q.person_photos("safi")  # case-insensitive, matches Task 3's find_person_by_name

    assert len(r["items"]) == 1
    item = r["items"][0]
    assert item["id"] == iid and item["city"] == "Casablanca"
    assert item["taken_at"].startswith("2026-06-01")
    assert item["filename"] == filename


def test_person_photos_raises_a_clear_error_for_an_unknown_name(q):
    with pytest.raises(ValueError, match="no person"):
        q.q.person_photos("nobody")
