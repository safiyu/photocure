"""Keyword export: category, highlight and event keywords written to XMP sidecars."""
from __future__ import annotations

import time
import types

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from photocure import keywords
from photocure.config import CuratorConfig
from photocure.db.models import EventGroup, ImageRecord, SemanticScore
from photocure.db.session import get_session, init_db
from photocure.features import save_features
from photocure.highlights import HighlightWeights
from photocure.scanner.xmp import find_sidecar, get_on_disk_owned_tags, parse_xmp, update_xmp_sidecar
from photocure.sidecar_queue import SidecarQueue
from photocure.ui.app import create_app

from .conftest import make_image

W = HighlightWeights(rating=0.1, favorite=0.15, preference=0.0)


def _rec(**kw) -> ImageRecord:
    return ImageRecord(path="/x/a.jpg", filename="a.jpg", **kw)


def _sem(aesthetic=0.9, pref=None) -> SemanticScore:
    return SemanticScore(image_id=1, aesthetic=aesthetic, preference=pref, backend="clip")


# ------------------------------------------------------------------ the rules
def test_a_good_photo_is_a_highlight_and_a_rejected_or_duplicate_one_is_not():
    assert keywords.is_highlight(_rec(), _sem(0.9), W)
    assert not keywords.is_highlight(_rec(), _sem(0.3), W)
    assert not keywords.is_highlight(_rec(rating=-1), _sem(0.99), W)
    assert not keywords.is_highlight(_rec(is_duplicate=True), _sem(0.99), W)
    assert not keywords.is_highlight(_rec(is_alternate=True), _sem(0.99), W)
    assert not keywords.is_highlight(_rec(), None, W)
    assert not keywords.is_highlight(_rec(), _sem(None), W)


def test_favourites_and_stars_can_lift_a_photo_into_the_highlights():
    assert not keywords.is_highlight(_rec(), _sem(0.55), W)
    assert keywords.is_highlight(_rec(is_favorite=True), _sem(0.55), W)
    assert keywords.is_highlight(_rec(rating=5), _sem(0.58), W)


def test_keywords_are_category_highlight_and_a_named_event():
    r = _rec(category="scenery", manual_bucket=None)
    assert keywords.build_keywords(r, highlight=False, event_label=None) == ("photocure-cat-scenery",)
    got = keywords.build_keywords(r, highlight=True, event_label="Summer Trip")
    assert got == ("photocure-cat-scenery", "photocure-event-Summer Trip", "photocure-highlight")


def test_a_manual_category_wins_and_unsorted_photos_get_none():
    r = _rec(category="scenery", manual_bucket="people")
    assert keywords.build_keywords(r, False, None) == ("photocure-cat-people",)
    assert keywords.build_keywords(_rec(), False, None) == ()
    assert keywords.build_keywords(_rec(category="unrouted"), False, None) == ()


def test_commas_and_control_characters_are_removed_from_event_names():
    # Stored tags are joined with commas, so a comma in a name would look like an outside edit.
    r = _rec(category="food")
    got = keywords.build_keywords(r, False, "Paris,  France\x07 trip")
    assert "photocure-event-Paris France trip" in got


@pytest.mark.parametrize("label", [None, "", "  ", "Event12", "event3"])
def test_default_event_names_are_not_keywords(label):
    r = _rec(category="food")
    assert keywords.build_keywords(r, False, label) == ("photocure-cat-food",)


def test_every_keyword_is_ours_and_safe_to_store():
    r = _rec(category="food")
    for t in keywords.build_keywords(r, True, "A" * 200 + " <b>&"):
        assert t.startswith("photocure-")
    long = [t for t in keywords.build_keywords(r, False, "x" * 200) if "event" in t][0]
    assert len(long) <= len("photocure-event-") + 64


# ------------------------------------------------------- queue and bulk write
@pytest.fixture()
def lib(tmp_path):
    db = tmp_path / "k.db"
    init_db(db, drop=True)
    cfg = CuratorConfig()
    cfg.db_path = db
    cfg.write_sidecars = True
    cfg.sidecar_debounce_seconds = 0
    ids = {}
    with get_session(db) as s:
        ev = EventGroup(label="Beach Day", image_count=2)
        s.add(ev)
        s.flush()
        for name, cat, aes in (("a.jpg", "scenery", 0.9), ("b.jpg", "people", 0.2), ("c.jpg", "food", 0.5)):
            p = make_image(tmp_path / "photos" / name, rng_seed=len(ids) + 1)
            img = ImageRecord(path=str(p), filename=name, category=cat, scan_status="scanned",
                              event_id=ev.id if name != "c.jpg" else None)
            s.add(img)
            s.flush()
            s.add(SemanticScore(image_id=img.id, aesthetic=aes, backend="clip"))
            ids[name] = img.id
    return types.SimpleNamespace(db=db, cfg=cfg, ids=ids, root=tmp_path / "photos")


def _tags(lib, name):
    return get_on_disk_owned_tags(lib.root / name)


def test_bulk_write_adds_keywords_and_creates_missing_sidecars(lib):
    out = keywords.write_all(lib.db, lib.cfg)
    assert out["written"] == 3 and out["total"] == 3
    assert _tags(lib, "a.jpg") == {"photocure-cat-scenery", "photocure-event-Beach Day", "photocure-highlight"}
    assert _tags(lib, "b.jpg") == {"photocure-cat-people", "photocure-event-Beach Day"}
    assert _tags(lib, "c.jpg") == {"photocure-cat-food"}


def test_bulk_write_never_touches_a_rating_or_another_programs_keywords(lib):
    update_xmp_sidecar(lib.root / "a.jpg", rating=4)
    from lxml import etree
    sc = find_sidecar(lib.root / "a.jpg")
    root = etree.fromstring(sc.read_bytes())
    ns = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "dc": "http://purl.org/dc/elements/1.1/"}
    desc = root.find(".//rdf:Description", ns)
    subj = etree.SubElement(desc, "{%s}subject" % ns["dc"])
    bag = etree.SubElement(subj, "{%s}Bag" % ns["rdf"])
    etree.SubElement(bag, "{%s}li" % ns["rdf"]).text = "family holiday"
    sc.write_bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8"))
    out = keywords.write_all(lib.db, lib.cfg)
    info = parse_xmp(sc.read_bytes())
    assert info.rating == 4 and "family holiday" in info.keywords
    assert "photocure-highlight" in info.keywords
    assert out["kept_foreign_rating"] == 1


def test_a_sidecar_edited_by_another_program_is_left_alone(lib):
    keywords.write_all(lib.db, lib.cfg)
    update_xmp_sidecar(lib.root / "b.jpg", rating=5)          # changed outside Photocure afterwards
    update_xmp_sidecar(lib.root / "b.jpg", add_tags=("photocure-cat-bogus",))
    with get_session(lib.db) as s:
        s.get(ImageRecord, lib.ids["b.jpg"]).manual_bucket = "food"
    out = keywords.write_all(lib.db, lib.cfg)
    assert out["edited_elsewhere"] == 1
    assert "photocure-cat-bogus" in _tags(lib, "b.jpg")
    assert _tags(lib, "a.jpg") >= {"photocure-cat-scenery"}


def test_removing_keywords_strips_ours_and_creates_nothing(lib):
    keywords.write_all(lib.db, lib.cfg)
    sc_c = find_sidecar(lib.root / "c.jpg")
    sc_c.unlink()                                              # no sidecar for c, none should appear
    out = keywords.write_all(lib.db, lib.cfg, keywords=False)
    assert out["no_sidecar"] == 1 and find_sidecar(lib.root / "c.jpg") is None
    assert _tags(lib, "a.jpg") == set() and _tags(lib, "b.jpg") == set()


def test_removal_keeps_the_flags_the_database_still_asserts(lib):
    with get_session(lib.db) as s:
        s.get(ImageRecord, lib.ids["a.jpg"]).is_best_shot = True
    keywords.write_all(lib.db, lib.cfg)
    keywords.write_all(lib.db, lib.cfg, keywords=False)
    assert _tags(lib, "a.jpg") == {"photocure-best-shot"}


def test_writing_is_refused_when_sidecars_are_switched_off(lib):
    lib.cfg.write_sidecars = False
    with pytest.raises(ValueError, match="turned off"):
        keywords.write_all(lib.db, lib.cfg)
    assert find_sidecar(lib.root / "a.jpg") is None


def test_missing_files_and_missing_photos_are_skipped_not_fatal(lib):
    (lib.root / "b.jpg").unlink()
    with get_session(lib.db) as s:
        s.get(ImageRecord, lib.ids["c.jpg"]).scan_status = "missing"
    out = keywords.write_all(lib.db, lib.cfg)
    assert out["total"] == 2 and out["written"] == 1 and out["errors"] + out["skipped"] == 1


def test_progress_and_cancel_are_honoured(lib):
    seen = []
    out = keywords.write_all(lib.db, lib.cfg, on_progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (3, 3) and out["cancelled"] is False
    for name in lib.ids:
        find_sidecar(lib.root / name).unlink()
    calls = []
    out = keywords.write_all(lib.db, lib.cfg, cancel_check=lambda: calls.append(1) or len(calls) > 1)
    assert out["cancelled"] is True and out["written"] < 3


def test_the_rating_queue_writes_keywords_only_when_the_setting_is_on(lib):
    q = SidecarQueue(lib.db, debounce=60)  # only the explicit flush runs
    q.schedule(lib.ids["a.jpg"], overwrite=True)
    q.flush()
    assert _tags(lib, "a.jpg") == set()
    find_sidecar(lib.root / "a.jpg").unlink()
    save_features(lib.db, {"write_keywords": True})
    q.schedule(lib.ids["a.jpg"], overwrite=True)
    q.flush()
    assert "photocure-cat-scenery" in _tags(lib, "a.jpg")
    save_features(lib.db, {"write_keywords": False})
    q.schedule(lib.ids["a.jpg"], overwrite=True)
    q.flush()
    assert _tags(lib, "a.jpg") == set()


# ----------------------------------------------------------------- API and CLI
@pytest.fixture()
def env(lib):
    app = create_app(str(lib.db))
    app.state.cfg.write_sidecars = True
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    return types.SimpleNamespace(app=app, client=client, lib=lib)


def _wait(client, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_api_status_reports_the_setting_and_the_sidecar_switch(env):
    st = env.client.get("/api/keywords").json()
    assert st == {"enabled": False, "sidecars": True, "photos": 3}
    env.client.post("/api/features", json={"write_keywords": True})
    assert env.client.get("/api/keywords").json()["enabled"] is True


def test_api_write_runs_as_a_job_and_reports_counts(env):
    r = env.client.post("/api/keywords/write", json={})
    assert r.status_code == 202
    job = _wait(env.client, r.json()["job_id"])
    assert job["status"] == "done" and job["result"]["written"] == 3
    assert "photocure-highlight" in _tags(env.lib, "a.jpg")
    r = env.client.post("/api/keywords/write", json={"remove": True})
    job = _wait(env.client, r.json()["job_id"])
    assert job["result"]["written"] == 3 and _tags(env.lib, "a.jpg") == set()


def test_api_write_explains_when_sidecars_are_off_and_needs_a_token(env):
    env.app.state.cfg.write_sidecars = False
    assert env.client.post("/api/keywords/write", json={}).status_code == 400
    assert TestClient(env.app).post("/api/keywords/write", json={}).status_code == 401
    assert TestClient(env.app).get("/api/keywords").status_code == 401


def test_cli_keywords_writes_and_removes(env):
    from photocure import cli
    runner = CliRunner()
    res = runner.invoke(cli.app, ["keywords", "--db", str(env.lib.db)], env={"COLUMNS": "200"})
    assert res.exit_code == 0, res.output
    assert "photocure-cat-people" in _tags(env.lib, "b.jpg")
    res = runner.invoke(cli.app, ["keywords", "--db", str(env.lib.db), "--remove"], env={"COLUMNS": "200"})
    assert res.exit_code == 0 and _tags(env.lib, "b.jpg") == set()


def test_photos_with_nothing_to_say_get_no_sidecar(lib):
    with get_session(lib.db) as s:
        s.get(ImageRecord, lib.ids["c.jpg"]).category = None       # no category, no event, not a highlight
    out = keywords.write_all(lib.db, lib.cfg)
    assert find_sidecar(lib.root / "c.jpg") is None
    assert out["nothing_to_write"] == 1 and out["written"] == 2


def test_a_photo_that_lost_its_keywords_still_has_the_old_ones_removed(lib):
    keywords.write_all(lib.db, lib.cfg)
    assert _tags(lib, "c.jpg") == {"photocure-cat-food"}
    with get_session(lib.db) as s:
        s.get(ImageRecord, lib.ids["c.jpg"]).category = None
    keywords.write_all(lib.db, lib.cfg)
    assert _tags(lib, "c.jpg") == set()
