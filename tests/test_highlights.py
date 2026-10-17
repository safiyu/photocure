"""Highlights ranking, best of each event, year in review, HTML export, API and CLI."""
from __future__ import annotations

import types
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select
from typer.testing import CliRunner

from photocure import highlights
from photocure.db import models as M
from photocure.db.models import EventGroup, ImageRecord, SemanticScore
from photocure.db.session import get_session
from photocure.highlights import HighlightWeights, highlight_score
from photocure.ui.app import create_app

from .conftest import make_image

W = HighlightWeights(rating=0.1, favorite=0.2, preference=0.5)


def _dt(y, m=6, d=1):
    return datetime(y, m, d, 12, tzinfo=timezone.utc)


def _img(session, i, aest=0.5, pref=None, rating=None, fav=False, event=None, taken=None,
         source="exif:dtoriginal", **kw):
    img = ImageRecord(path=kw.pop("path", f"/lib/{i}.jpg"), filename=kw.pop("filename", f"{i}.jpg"),
                      rating=rating, is_favorite=fav, event_id=event, taken_at=taken,
                      taken_at_source=source if taken else "", **kw)
    session.add(img)
    session.flush()
    session.add(SemanticScore(image_id=img.id, aesthetic=aest, preference=pref, backend="clip",
                              top1_category="people"))
    session.flush()
    return img


# ------------------------------------------------------------------- scoring
def test_score_is_the_model_score_without_signals():
    assert highlight_score(0.6, None, None, False, W) == pytest.approx(0.6)


def test_preference_is_blended_by_its_weight():
    assert highlight_score(0.6, 0.2, None, False, W) == pytest.approx(0.5 * 0.6 + 0.5 * 0.2)


def test_stars_lift_or_lower_around_three_and_favourites_add():
    base = highlight_score(0.5, None, None, False, W)
    assert highlight_score(0.5, None, 5, False, W) == pytest.approx(base + 0.1)
    assert highlight_score(0.5, None, 3, False, W) == pytest.approx(base)
    assert highlight_score(0.5, None, 1, False, W) == pytest.approx(base - 0.1)
    assert highlight_score(0.5, None, 0, False, W) == pytest.approx(base)
    assert highlight_score(0.5, None, None, True, W) == pytest.approx(base + 0.2)


def test_rejected_or_unscored_photos_have_no_score():
    assert highlight_score(0.9, None, -1, True, W) is None
    assert highlight_score(None, None, 5, True, W) is None


def test_zero_weights_leave_only_the_model_score():
    z = HighlightWeights(rating=0, favorite=0, preference=0)
    assert highlight_score(0.4, 0.9, 5, True, z) == pytest.approx(0.4)


def test_weights_come_from_the_feature_settings():
    w = HighlightWeights.from_features({"highlight_rating_weight": 0.3,
                                        "highlight_favorite_weight": 0.0,
                                        "highlight_preference_weight": 1.0})
    assert (w.rating, w.favorite, w.preference) == (0.3, 0.0, 1.0)


def test_sql_ranking_matches_the_python_score(session):
    rows = [_img(session, 1, 0.5), _img(session, 2, 0.7, pref=0.1), _img(session, 3, 0.4, rating=5),
            _img(session, 4, 0.4, fav=True), _img(session, 5, 0.9, rating=1, pref=0.8, fav=True)]
    stmt = (select(ImageRecord.id, highlights.highlight_expr(W))
            .join(SemanticScore, SemanticScore.image_id == ImageRecord.id))
    got = dict(session.exec(stmt).all())
    sem = {s.image_id: s for s in session.exec(select(SemanticScore)).all()}
    for r in rows:
        want = highlight_score(sem[r.id].aesthetic, sem[r.id].preference, r.rating, r.is_favorite, W)
        assert got[r.id] == pytest.approx(want)


# --------------------------------------------------------- events and years
def _event(session, label, start):
    ev = EventGroup(start=start, end=start, label=label, image_count=0)
    session.add(ev)
    session.flush()
    return ev


def test_best_of_each_event_picks_the_top_photos_in_time_order(session):
    late = _event(session, "Summer", _dt(2024, 8))
    early = _event(session, "Spring", _dt(2024, 3))
    _img(session, 1, 0.5, event=late.id)
    best_late = _img(session, 2, 0.8, event=late.id)
    best_early = _img(session, 3, 0.6, event=early.id)
    _img(session, 4, 0.9, event=early.id, rating=-1)               # rejected
    _img(session, 5, 0.95, event=early.id, is_duplicate=True)      # duplicate
    _img(session, 6, 0.95, event=early.id, is_alternate=True)      # burst alternate
    _img(session, 7, None, event=early.id)                         # not scored
    out = highlights.best_of_events(session, W, per_event=1)
    assert [e["label"] for e in out] == ["Spring", "Summer"]
    assert [e["items"][0][0] for e in out] == [best_early.id, best_late.id]
    both = highlights.best_of_events(session, W, per_event=5)
    assert len(both[1]["items"]) == 2 and both[1]["items"][0][0] == best_late.id


def test_events_without_a_qualifying_photo_are_left_out(session):
    ev = _event(session, "Dull", _dt(2024, 1))
    _img(session, 1, 0.2, event=ev.id)
    assert highlights.best_of_events(session, W, per_event=1, min_score=0.5) == []


def test_top_of_year_ranks_within_the_year_and_ignores_unreliable_dates(session):
    a = _img(session, 1, 0.7, taken=_dt(2024, 2))
    b = _img(session, 2, 0.9, taken=_dt(2024, 11))
    _img(session, 3, 0.99, taken=_dt(2023, 5))
    _img(session, 4, 0.99, taken=_dt(2024, 5), source="mtime")
    _img(session, 5, 0.98, taken=_dt(2024, 5), rating=-1)
    assert [i for i, _ in highlights.top_of_year(session, W, 2024, 10)] == [b.id, a.id]
    assert [i for i, _ in highlights.top_of_year(session, W, 2024, 1)] == [b.id]
    assert highlights.top_of_year(session, W, 1999, 10) == []


def test_years_with_photos_are_listed_newest_first(session):
    _img(session, 1, 0.5, taken=_dt(2022))
    _img(session, 2, 0.5, taken=_dt(2024))
    _img(session, 3, 0.5, taken=_dt(2024, 9))
    _img(session, 4, 0.5, taken=_dt(2021), source="mtime")
    _img(session, 5, None, taken=_dt(2020))
    assert highlights.years_available(session) == [(2024, 2), (2022, 1)]


# --------------------------------------------------------------- html export
def test_export_html_escapes_text_and_inlines_thumbnails(session, tmp_path):
    path = make_image(tmp_path / "pic.jpg", rng_seed=5)
    img = _img(session, 1, 0.7, taken=_dt(2024), path=str(path), filename="<b>pic</b>.jpg",
               extension=".jpg", width=96, height=128)
    page = highlights.export_html(session, [(img.id, 0.71)], title="Best of <2024>")
    assert "Best of &lt;2024&gt;" in page and "<b>pic</b>" not in page
    assert "&lt;b&gt;pic&lt;/b&gt;.jpg" in page
    assert "data:image/webp;base64," in page
    assert "<script" not in page.lower()


def test_export_html_survives_an_unreadable_file(session):
    img = _img(session, 1, 0.7, path="/nowhere/missing.jpg", filename="missing.jpg")
    page = highlights.export_html(session, [(img.id, 0.5)], title="T")
    assert "missing.jpg" in page


# ----------------------------------------------------------------- API + CLI
@pytest.fixture()
def env(tmp_path):
    app = create_app(str(tmp_path / "h.db"))
    app.state.cfg.cache_dir = tmp_path / "cache"
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    lib = tmp_path / "photos"
    for i in range(4):
        make_image(lib / f"p{i}.jpg", rng_seed=20 + i)
    assert client.post("/api/scan", json={"path": str(lib)}).status_code == 200
    assert client.post("/api/infer", json={"device": "cpu"}).status_code == 200
    with get_session(app.state.db) as s:
        imgs = list(s.exec(select(ImageRecord).order_by(ImageRecord.filename)).all())
        sems = {x.image_id: x for x in s.exec(select(SemanticScore)).all()}
        ev = EventGroup(start=_dt(2024, 4), end=_dt(2024, 4), label="Trip", image_count=4)
        s.add(ev)
        s.flush()
        for n, img in enumerate(imgs):
            sems[img.id].aesthetic = [0.50, 0.60, 0.70, 0.80][n]
            img.taken_at, img.taken_at_source = _dt(2024, 4, 1 + n), "exif:dtoriginal"
            img.event_id = ev.id
        ids = [i.id for i in imgs]
    return types.SimpleNamespace(app=app, client=client, ids=ids, tmp=tmp_path, lib=lib)


def test_highlights_listing_orders_by_the_blended_score(env):
    # Favouriting the weakest photo (+0.15 by default) lifts it past the next one up, not the best.
    with get_session(env.app.state.db) as s:
        s.get(ImageRecord, env.ids[0]).is_favorite = True
        s.get(ImageRecord, env.ids[3]).rating = -1
    r = env.client.get("/api/images", params={"highlights": "true"})
    items = r.json()["items"]
    assert [it["id"] for it in items] == [env.ids[2], env.ids[0], env.ids[1]]
    assert all("highlight" in it for it in items)
    assert items[1]["highlight"] == pytest.approx(0.65, abs=1e-3)
    assert r.json()["total"] == 3


def test_min_aesthetic_still_filters_on_the_model_score(env):
    r = env.client.get("/api/images", params={"highlights": "true", "min_aesthetic": 0.65})
    assert {it["id"] for it in r.json()["items"]} == {env.ids[2], env.ids[3]}


def test_weights_in_settings_change_the_ranking(env):
    with get_session(env.app.state.db) as s:
        s.get(ImageRecord, env.ids[0]).is_favorite = True
    assert env.client.post("/api/features", json={"highlight_favorite_weight": 0.5}).status_code == 200
    first = env.client.get("/api/images", params={"highlights": "true"}).json()["items"][0]
    assert first["id"] == env.ids[0]
    env.client.post("/api/features", json={"highlight_favorite_weight": 0.0})
    first = env.client.get("/api/images", params={"highlights": "true"}).json()["items"][0]
    assert first["id"] == env.ids[3]


def test_features_api_reads_validates_and_requires_a_token(env):
    got = env.client.get("/api/features").json()
    assert got["features"]["highlight_rating_weight"] == 0.10
    assert env.client.post("/api/features", json={"highlight_rating_weight": 9}).status_code == 400
    assert env.client.post("/api/features", json={"bogus": 1}).status_code == 400
    assert TestClient(env.app).post("/api/features", json={}).status_code == 401


def test_best_of_events_endpoint(env):
    r = env.client.get("/api/highlights/events", params={"per_event": 2})
    assert r.status_code == 200
    ev = r.json()["events"][0]
    assert ev["label"] == "Trip" and ev["count"] == 2
    assert [it["id"] for it in ev["items"]] == [env.ids[3], env.ids[2]]
    assert env.client.get("/api/highlights/events", params={"per_event": 0}).status_code == 422


def test_years_endpoint_and_export_download(env):
    assert env.client.get("/api/highlights/years").json()["years"] == [{"year": 2024, "count": 4}]
    r = env.client.get("/api/highlights/export", params={"year": 2024, "n": 2})
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"] and "2024" in r.headers["content-disposition"]
    assert r.headers["content-type"].startswith("text/html")
    assert r.text.count("data:image/webp;base64,") == 2
    assert env.client.get("/api/highlights/export", params={"year": 1999}).status_code == 404
    assert env.client.get("/api/highlights/export", params={"year": 2024, "n": 0}).status_code == 422


def test_cli_highlights_writes_a_page_and_copies_originals(env):
    from photocure import cli
    out = env.tmp / "yearbook"
    res = CliRunner().invoke(cli.app, ["highlights", "--db", str(env.app.state.db),
                                       "--year", "2024", "--top", "2", "--out", str(out), "--copy"],
                             env={"COLUMNS": "300"})
    assert res.exit_code == 0, res.output
    page = (out / "highlights-2024.html").read_text()
    assert page.count("data:image/webp;base64,") == 2
    copies = sorted(p.name for p in (out / "photos").iterdir())
    assert copies == ["01_p3.jpg", "02_p2.jpg"]


def test_cli_highlights_reports_an_empty_year(env):
    from photocure import cli
    res = CliRunner().invoke(cli.app, ["highlights", "--db", str(env.app.state.db),
                                       "--year", "1999", "--out", str(env.tmp / "x")])
    assert res.exit_code == 1 and "No scored photos" in res.output


# ------------------------------------------------------------ review fixes
def test_the_far_future_does_not_overflow_the_year_bounds(session):
    assert highlights.top_of_year(session, W, 9999, 5) == []


def test_api_and_cli_handle_the_largest_year(env):
    from photocure import cli
    assert env.client.get("/api/highlights/export", params={"year": 9999}).status_code == 404
    res = CliRunner().invoke(cli.app, ["highlights", "--db", str(env.app.state.db), "--year", "9999",
                                       "--out", str(env.tmp / "y")])
    assert res.exit_code == 1 and "No scored photos" in res.output


def test_events_minimum_filters_the_model_score_like_the_photo_grid(session):
    ev = _event(session, "Trip", _dt(2024, 5))
    low = _img(session, 1, 0.30, event=ev.id, fav=True)       # a favourite lifts the blended score
    high = _img(session, 2, 0.80, event=ev.id)
    out = highlights.best_of_events(session, W, per_event=5, min_score=0.5)
    assert [i for i, _ in out[0]["items"]] == [high.id]
    assert low.id not in [i for e in out for i, _ in e["items"]]


def test_taste_is_not_added_on_top_of_a_photos_own_rating_signal():
    # A favourite or a good rating is the label the taste model learned from, so counting both double counts.
    assert highlight_score(0.6, 0.2, 5, False, W) == pytest.approx(0.6 + 0.1)
    assert highlight_score(0.6, 0.2, None, True, W) == pytest.approx(0.6 + 0.2)
    assert highlight_score(0.6, 0.2, 1, False, W) == pytest.approx(0.6 - 0.1)
    assert highlight_score(0.6, 0.2, 3, False, W) == pytest.approx(0.5 * 0.6 + 0.5 * 0.2)   # three stars is neutral
    # a favourite with a bad rating contradicts itself, so it stays unlabelled and taste still counts
    assert highlight_score(0.6, 0.2, 1, True, W) == pytest.approx(0.5 * 0.6 + 0.5 * 0.2 - 0.1 + 0.2)


def test_sql_ranking_ignores_taste_for_labelled_photos_like_the_python_score(session):
    rows = [_img(session, 1, 0.5, pref=0.9, rating=5), _img(session, 2, 0.5, pref=0.9, fav=True),
            _img(session, 3, 0.5, pref=0.9, rating=2), _img(session, 4, 0.5, pref=0.9, rating=3),
            _img(session, 5, 0.5, pref=0.9, rating=1, fav=True), _img(session, 6, 0.5, pref=0.9)]
    stmt = (select(ImageRecord.id, highlights.highlight_expr(W))
            .join(SemanticScore, SemanticScore.image_id == ImageRecord.id))
    got = dict(session.exec(stmt).all())
    for r in rows:
        want = highlight_score(0.5, 0.9, r.rating, r.is_favorite, W)
        assert got[r.id] == pytest.approx(want), r.filename
