"""C4 acceptance: facets, bound-parameter filters, ratings/rejects/favourites and sidecar writes."""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from lxml import etree

from photocure.db import models as M
from photocure.db.session import get_session, init_db
from photocure.scanner.xmp import get_on_disk_rating, update_xmp_sidecar

from .world import build_world

XMP = "{http://ns.adobe.com/xap/1.0/}"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"


@pytest.fixture()
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PCUR_SIDECAR_DEBOUNCE_SECONDS", "600")  # tests flush by hand
    from photocure.ui.app import create_app
    w = build_world(tmp_path, n=6)
    meta = [("Canon R5", 5, "MA", "2023-12-31 23:00:00"), ("Canon R5", None, "MA", "2024-01-01 08:00:00"),
            ("Pixel 8", 3, "FR", "2024-03-10 12:00:00"), ("Pixel 8", 0, "", "2024-03-10 18:30:00"),
            ("", -1, "FR", None), ("Canon R5", 3, "", "2024-06-01 09:00:00")]
    with get_session(w.db) as s:
        for rid, (cam, rating, cc, taken) in zip(w.ids, meta):
            r = s.get(M.ImageRecord, rid)
            r.camera_model, r.rating, r.country_code = cam, rating, cc
            r.taken_at = datetime.fromisoformat(taken).replace(tzinfo=timezone.utc) if taken else None
        s.get(M.ImageRecord, w.ids[0]).is_favorite = True
    app = create_app(str(w.db))
    client = TestClient(app, headers={"X-PCUR-Token": app.state.token})
    return SimpleNamespace(client=client, app=app, ids=w.ids, db=w.db, paths=sorted(w.paths),
                           anon=TestClient(app), tmp=tmp_path)


def ids_of(resp):
    return [i["id"] for i in resp.json()["items"]]


def same(resp, *want):
    """The response holds exactly these photos (default order is by database id, not list order)."""
    assert sorted(ids_of(resp)) == sorted(want)


# ------------------------------------------------------------------- facets
def test_facets_count_cameras_ratings_dates_countries_and_favourites(lib):
    f = lib.client.get("/api/facets").json()
    assert f["total"] == 6 and f["favorites"] == 1
    assert f["cameras"] == [{"name": "Canon R5", "count": 3}, {"name": "Pixel 8", "count": 2}]
    # null counts as 0: ids[1] (None) and ids[3] (0) are both unrated
    assert f["ratings"] == {"-1": 1, "0": 2, "1": 0, "2": 0, "3": 2, "4": 0, "5": 1}
    assert f["countries"] == [{"code": "FR", "count": 2}, {"code": "MA", "count": 2}]
    assert (f["taken_min"], f["taken_max"]) == ("2023-12-31", "2024-06-01")


def test_facets_on_an_empty_library(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_CACHE_DIR", str(tmp_path / "c"))
    from photocure.ui.app import create_app
    db = tmp_path / "e.db"
    init_db(db)
    app = create_app(str(db))
    f = TestClient(app, headers={"X-PCUR-Token": app.state.token}).get("/api/facets").json()
    assert f["total"] == 0 and f["cameras"] == [] and f["taken_min"] is None and f["taken_max"] is None
    assert set(f["ratings"].values()) == {0}


def test_facets_need_auth(lib):
    assert lib.anon.get("/api/facets").status_code == 401


# ------------------------------------------------------------------ filters
def test_filters_each_select_the_right_photos(lib):
    c, i = lib.client, lib.ids
    same(c.get("/api/images?camera=Pixel 8"), i[2], i[3])
    same(c.get("/api/images?rating=3"), i[2], i[5])
    same(c.get("/api/images?rating=0"), i[1], i[3])      # null and 0 are both unrated
    same(c.get("/api/images?rating=-1"), i[4])
    same(c.get("/api/images?min_rating=3"), i[0], i[2], i[5])
    same(c.get("/api/images?country=fr"), i[2], i[4])    # case-insensitive code
    same(c.get("/api/images?favorite=true"), i[0])
    assert len(ids_of(c.get("/api/images?favorite=false"))) == 5


def test_date_range_is_inclusive_of_the_whole_end_day(lib):
    c, i = lib.client, lib.ids
    same(c.get("/api/images?date_from=2024-03-10&date_to=2024-03-10"), i[2], i[3])
    same(c.get("/api/images?date_from=2024-01-01&date_to=2024-03-09"), i[1])
    same(c.get("/api/images?date_from=2024-04-01"), i[5])
    assert i[4] not in ids_of(c.get("/api/images?date_from=1900-01-01"))  # no date: never in a range


def test_sort_by_taken_puts_undated_photos_last(lib):
    got = ids_of(lib.client.get("/api/images?sort=taken"))
    i = lib.ids
    assert got == [i[0], i[1], i[2], i[3], i[5], i[4]]


def test_filters_combine_and_pagination_total_follows(lib):
    r = lib.client.get("/api/images?camera=Canon R5&min_rating=3&limit=1&offset=1").json()
    assert r["total"] == 2 and len(r["items"]) == 1


@pytest.mark.parametrize("param,value", [
    ("camera", "x'; DROP TABLE images; --"), ("camera", "' OR '1'='1"), ("country", "';"),
    ("q", "%' OR 1=1 --"), ("q", "\\"), ("category", "'; DELETE FROM images; --"),
])
def test_malicious_strings_are_literal_text_not_sql(lib, param, value):
    r = lib.client.get("/api/images", params={param: value})
    assert r.status_code in (200, 422)
    if r.status_code == 200:
        assert r.json()["total"] == 0
    with sqlite3.connect(lib.db) as con:
        assert con.execute("select count(*) from images").fetchone()[0] == 6


@pytest.mark.parametrize("qs", ["date_from=notadate", "date_to=2024-13-45", "rating=6", "rating=-2",
                                "min_rating=0", "sort=drop", "country=FRA", "country=F"])
def test_invalid_filter_values_are_rejected(lib, qs):
    assert lib.client.get(f"/api/images?{qs}").status_code == 422


def test_items_expose_the_new_fields(lib):
    item = lib.client.get("/api/images?camera=Canon R5&sort=taken").json()["items"][0]
    assert item["is_favorite"] is True and item["country_code"] == "MA" and item["rating"] == 5


# ---------------------------------------------------------- rating endpoints
def sidecar_for(path: Path) -> Path:
    return path.with_name(path.name + ".xmp")


def rec(lib, rid):
    with get_session(lib.db) as s:
        r = s.get(M.ImageRecord, rid)
        s.expunge(r)
        return r


def test_rating_sets_and_clears_and_returns_the_updated_item(lib):
    rid = lib.ids[3]
    r = lib.client.post(f"/api/images/{rid}/rating", json={"rating": 4}).json()
    assert r["ok"] and r["item"]["rating"] == 4 and rec(lib, rid).rating == 4
    assert lib.client.post(f"/api/images/{rid}/rating", json={"rating": 0}).json()["item"]["rating"] is None
    assert rec(lib, rid).rating is None


@pytest.mark.parametrize("body", [{"rating": 6}, {"rating": -1}, {}, {"rating": "five"}])
def test_rating_rejects_bad_values(lib, body):
    assert lib.client.post(f"/api/images/{lib.ids[0]}/rating", json=body).status_code == 422


def test_reject_and_unreject(lib):
    rid = lib.ids[0]
    assert lib.client.post(f"/api/images/{rid}/reject").json()["item"]["rating"] == -1
    assert lib.client.get("/api/facets").json()["ratings"]["-1"] == 2
    assert lib.client.post(f"/api/images/{rid}/reject", json={"reject": False}).json()["item"]["rating"] is None


def test_favourite_toggles_and_can_be_set_explicitly(lib):
    rid = lib.ids[1]
    assert lib.client.post(f"/api/images/{rid}/favorite").json()["item"]["is_favorite"] is True
    assert lib.client.post(f"/api/images/{rid}/favorite").json()["item"]["is_favorite"] is False
    assert lib.client.post(f"/api/images/{rid}/favorite", json={"favorite": True}).json()["item"]["is_favorite"]
    assert lib.client.post(f"/api/images/{rid}/favorite", json={"favorite": True}).json()["item"]["is_favorite"]


def test_favourites_never_touch_a_sidecar(lib):
    lib.client.post(f"/api/images/{lib.ids[1]}/favorite")
    lib.app.state.sidecars.flush()
    assert not sidecar_for(lib.paths[1]).exists()


def test_unknown_photo_and_missing_auth(lib):
    for path in ("rating", "reject", "favorite"):
        body = {"rating": 3} if path == "rating" else {}
        assert lib.client.post(f"/api/images/99999/{path}", json=body).status_code == 404
        assert lib.anon.post(f"/api/images/{lib.ids[0]}/{path}", json=body).status_code == 401


# ------------------------------------------------------------- sidecar writes
def test_a_rating_reaches_a_new_sidecar_after_the_debounce_flush(lib):
    rid, photo = lib.ids[3], lib.paths[3]
    lib.client.post(f"/api/images/{rid}/rating", json={"rating": 4})
    assert not sidecar_for(photo).exists() and lib.app.state.sidecars.pending == 1  # queued, not blocking
    lib.app.state.sidecars.flush()
    assert get_on_disk_rating(photo) == 4 and rec(lib, rid).xmp_written_rating == 4


def test_many_changes_to_one_photo_are_one_write_with_the_final_value(lib):
    rid, photo = lib.ids[3], lib.paths[3]
    for n in (1, 2, 3, 5, 4):
        lib.client.post(f"/api/images/{rid}/rating", json={"rating": n})
    assert lib.app.state.sidecars.pending == 1
    results = lib.app.state.sidecars.flush()
    assert list(results) == [rid] and get_on_disk_rating(photo) == 4


def test_the_real_debounce_timer_writes_without_a_manual_flush(lib):
    lib.app.state.sidecars.debounce = 0.05
    lib.client.post(f"/api/images/{lib.ids[3]}/rating", json={"rating": 2})
    deadline = time.time() + 5
    while time.time() < deadline and not sidecar_for(lib.paths[3]).exists():
        time.sleep(0.05)
    assert get_on_disk_rating(lib.paths[3]) == 2


def test_reject_writes_the_rejected_marker_and_a_blink_tag_rides_along(lib):
    rid, photo = lib.ids[3], lib.paths[3]
    with get_session(lib.db) as s:
        s.get(M.ImageRecord, rid).has_blinks = True
    lib.client.post(f"/api/images/{rid}/reject")
    lib.app.state.sidecars.flush()
    root = etree.parse(str(sidecar_for(photo))).getroot()
    tags = {li.text for li in root.iter(RDF + "li")}
    assert {"photocure-rejected", "photocure-blink"} <= tags and get_on_disk_rating(photo) == -1
    lib.client.post(f"/api/images/{rid}/reject", json={"reject": False})
    lib.app.state.sidecars.flush()
    tags = {li.text for li in etree.parse(str(sidecar_for(photo))).getroot().iter(RDF + "li")}
    assert "photocure-rejected" not in tags and "photocure-blink" in tags


def test_a_foreign_rating_asks_first_and_changes_nothing_until_confirmed(lib):
    rid, photo = lib.ids[3], lib.paths[3]
    update_xmp_sidecar(photo, rating=2)              # written by some other tool
    r = lib.client.post(f"/api/images/{rid}/rating", json={"rating": 5})
    assert r.status_code == 409 and r.json()["needs_confirmation"] is True
    assert r.json()["reason"] == "existing_foreign_rating"
    assert rec(lib, rid).rating == 0 and get_on_disk_rating(photo) == 2 and lib.app.state.sidecars.pending == 0
    ok = lib.client.post(f"/api/images/{rid}/rating", json={"rating": 5, "overwrite": True})
    assert ok.status_code == 200
    lib.app.state.sidecars.flush()
    assert get_on_disk_rating(photo) == 5 and rec(lib, rid).xmp_written_rating == 5


def test_an_edit_made_elsewhere_after_our_write_also_asks(lib):
    rid, photo = lib.ids[3], lib.paths[3]
    lib.client.post(f"/api/images/{rid}/rating", json={"rating": 3})
    lib.app.state.sidecars.flush()
    update_xmp_sidecar(photo, rating=1)              # another tool changes it later
    r = lib.client.post(f"/api/images/{rid}/rating", json={"rating": 4})
    assert r.status_code == 409 and r.json()["reason"] == "edited_elsewhere"
    assert get_on_disk_rating(photo) == 1


def test_a_request_during_a_background_write_is_not_mistaken_for_a_foreign_edit(lib, monkeypatch):
    """The writer has updated the file but not yet committed its record: the request must wait."""
    import threading

    from photocure import sidecar_queue
    rid = lib.ids[3]
    assert lib.client.post(f"/api/images/{rid}/rating", json={"rating": 3}).status_code == 200
    lib.app.state.sidecars.flush()
    lib.client.post(f"/api/images/{rid}/rating", json={"rating": 4})   # queued again
    inside, release = threading.Event(), threading.Event()
    real = sidecar_queue.sync_sidecar

    def paused(img, **kw):
        res = real(img, **kw)           # the file is now rewritten, the database row is not committed
        inside.set()
        release.wait(5)
        return res
    monkeypatch.setattr(sidecar_queue, "sync_sidecar", paused)
    writer = threading.Thread(target=lib.app.state.sidecars.flush)
    writer.start()
    assert inside.wait(5)
    answer = {}
    caller = threading.Thread(target=lambda: answer.update(
        r=lib.client.post(f"/api/images/{rid}/rating", json={"rating": 5})))
    caller.start()
    caller.join(0.4)
    assert caller.is_alive(), "the request did not wait for the write to finish"
    release.set()
    writer.join(5)
    caller.join(5)
    assert answer["r"].status_code == 200, answer["r"].text  # not a 409 "edited elsewhere"


def test_our_own_previous_write_is_not_mistaken_for_a_foreign_edit(lib):
    rid = lib.ids[3]
    for n in (3, 4):
        assert lib.client.post(f"/api/images/{rid}/rating", json={"rating": n}).status_code == 200
        lib.app.state.sidecars.flush()
    assert get_on_disk_rating(lib.paths[3]) == 4


def test_read_only_mode_keeps_ratings_in_the_database_and_writes_no_files(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PCUR_WRITE_SIDECARS", "false")
    from photocure.ui.app import create_app
    w = build_world(tmp_path, n=2)
    app = create_app(str(w.db))
    c = TestClient(app, headers={"X-PCUR-Token": app.state.token})
    update_xmp_sidecar(sorted(w.paths)[0], rating=2)  # would be "foreign", but nothing is written anyway
    assert c.post(f"/api/images/{w.ids[0]}/rating", json={"rating": 5}).status_code == 200
    assert app.state.sidecars.pending == 0 and get_on_disk_rating(sorted(w.paths)[0]) == 2


def test_a_missing_original_does_not_break_rating(lib):
    rid = lib.ids[3]
    lib.paths[3].unlink()
    assert lib.client.post(f"/api/images/{rid}/rating", json={"rating": 5}).status_code == 200
    res = lib.app.state.sidecars.flush()
    assert rec(lib, rid).rating == 5 and rid in res  # outcome recorded, nothing raised


def test_pending_writes_are_flushed_when_the_server_stops(lib):
    with lib.client:  # runs startup and shutdown hooks
        lib.client.post(f"/api/images/{lib.ids[3]}/rating", json={"rating": 3})
        assert not sidecar_for(lib.paths[3]).exists()
    assert get_on_disk_rating(lib.paths[3]) == 3


# ---------------------------------------------------------------- migration
def test_an_older_database_gains_geo_columns_and_every_declared_index(tmp_path):
    from photocure.db.session import SCHEMA_VERSION
    db = tmp_path / "old.db"
    init_db(db)
    con = sqlite3.connect(db)
    for idx in ("ix_images_country_code", "ix_images_is_favorite", "ix_images_camera_model",
                "ix_images_rating", "ix_images_has_blinks"):
        con.execute(f"DROP INDEX {idx}")
    for col in ("city", "region", "country_code", "is_favorite"):
        con.execute(f"ALTER TABLE images DROP COLUMN {col}")
    con.execute("PRAGMA user_version=5")
    con.commit()
    con.close()
    init_db(db)
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(images)")}
    idx = {r[1] for r in con.execute("PRAGMA index_list(images)")}
    assert {"city", "region", "country_code", "is_favorite"} <= cols
    assert {"ix_images_camera_model", "ix_images_rating", "ix_images_taken_at", "ix_images_country_code",
            "ix_images_is_favorite", "ix_images_has_blinks"} <= idx
    assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_the_images_endpoint_uses_the_indexes_for_filters(lib):
    with sqlite3.connect(lib.db) as con:
        for sql in ("select id from images where camera_model = 'x'",
                    "select id from images where country_code = 'FR'",
                    "select id from images where rating = 3",
                    "select id from images where taken_at >= '2024-01-01'"):
            plan = " ".join(r[3] for r in con.execute("explain query plan " + sql))
            assert "USING INDEX" in plan or "USING COVERING INDEX" in plan, plan

