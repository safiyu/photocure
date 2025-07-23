"""Taste learning: a personal score trained from favourites, stars and rejects."""
from __future__ import annotations

import types

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from photocure.db.models import ImageRecord, PreferenceModel, SemanticScore
from photocure.db.repository import dequantize_embedding, quantize_embedding
from photocure.db.session import get_session
from photocure.inference import preference as pref
from photocure.ui.app import create_app

DIR = np.random.default_rng(7).standard_normal(512).astype(np.float32)
DIR /= np.linalg.norm(DIR)


def _emb(seed: int, side: float) -> np.ndarray:
    """A unit vector that leans along (+1) or against (-1) the hidden taste direction."""
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    v = v + 0.9 * side * DIR
    return v / np.linalg.norm(v)


def _row(session, i, emb, *, rating=None, fav=False, backend="clip", **kw):
    img = ImageRecord(path=f"/lib/{i}.jpg", filename=f"{i}.jpg", rating=rating, is_favorite=fav, **kw)
    session.add(img)
    session.flush()
    session.add(SemanticScore(image_id=img.id, backend=backend, aesthetic=0.5,
                              embedding=quantize_embedding(emb.tolist())))
    session.flush()
    return img


def _labelled(session, n_each=20, start=0):
    pos = [_row(session, start + i, _emb(start + i, +1), fav=True) for i in range(n_each)]
    neg = [_row(session, start + 1000 + i, _emb(start + 1000 + i, -1), rating=-1) for i in range(n_each)]
    return pos, neg


def _prefs(session):
    return {s.image_id: s.preference for s in session.exec(select(SemanticScore)).all()}


def test_examples_are_favourites_and_good_stars_against_rejects_and_low_stars(session):
    _row(session, 1, _emb(1, 1), fav=True)
    _row(session, 2, _emb(2, 1), rating=4)
    _row(session, 3, _emb(3, 1), rating=5)
    _row(session, 4, _emb(4, -1), rating=-1)
    _row(session, 5, _emb(5, -1), rating=1)
    _row(session, 6, _emb(6, -1), rating=2)
    _row(session, 7, _emb(7, 0), rating=3)                        # neutral, never used
    _row(session, 8, _emb(8, 0))                                  # unrated
    _row(session, 9, _emb(9, 1), fav=True, rating=-1)             # contradicts itself
    _row(session, 10, _emb(10, 1), fav=True, backend="stub")      # placeholder embedding
    ids, X, y = pref.collect_examples(session)
    assert sorted(y.tolist()) == [0, 0, 0, 1, 1, 1] and X.shape == (6, 512) and len(ids) == 6


def test_training_learns_the_hidden_direction_and_scores_every_photo(session):
    _labelled(session, 20)
    unlabeled_good = _row(session, 5000, _emb(5000, +1))
    unlabeled_bad = _row(session, 5001, _emb(5001, -1))
    res = pref.train_preferences(session)
    assert res.status == "trained" and res.n_positive == 20 and res.n_negative == 20
    assert res.cv_accuracy > 0.85
    scores = _prefs(session)
    assert scores[unlabeled_good.id] > 0.6 > 0.4 > scores[unlabeled_bad.id]
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert len(session.exec(select(PreferenceModel)).all()) == 1


def test_too_few_examples_train_nothing(session):
    for i in range(5):
        _row(session, i, _emb(i, 1), fav=True)
        _row(session, 100 + i, _emb(100 + i, -1), rating=-1)
    res = pref.train_preferences(session)
    assert res.status == "not_enough"
    assert session.exec(select(PreferenceModel)).all() == []
    assert set(_prefs(session).values()) == {None}


def test_a_signal_that_does_not_generalise_is_not_applied(session):
    rng = np.random.default_rng(3)
    for i in range(40):
        e = rng.standard_normal(512).astype(np.float32)
        e /= np.linalg.norm(e)
        _row(session, i, e, fav=(i % 2 == 0), rating=None if i % 2 == 0 else -1)
    res = pref.train_preferences(session)
    assert res.status == "weak" and res.cv_accuracy < 0.7
    assert session.exec(select(PreferenceModel)).all() == []
    assert set(_prefs(session).values()) == {None}


def test_retraining_replaces_the_model(session):
    _labelled(session, 12)
    pref.train_preferences(session)
    _labelled(session, 12, start=300)
    res = pref.train_preferences(session)
    assert res.status == "trained"
    rows = session.exec(select(PreferenceModel)).all()
    assert len(rows) == 1 and rows[0].n_positive == 24


def test_apply_fills_new_rows_without_touching_scored_ones(session):
    _labelled(session, 15)
    pref.train_preferences(session)
    kept = session.exec(select(SemanticScore)).first()
    kept.preference = 0.123
    newcomer = _row(session, 9000, _emb(9000, +1))
    assert pref.apply_preferences(session, only_missing=True) == 1
    scores = _prefs(session)
    assert scores[kept.image_id] == 0.123 and scores[newcomer.id] is not None
    assert pref.apply_preferences(session) == 1                     # recompute fixes only the edited row
    assert _prefs(session)[kept.image_id] != 0.123
    assert pref.apply_preferences(session) == 0                     # and is idempotent


def test_apply_without_a_model_clears_stale_scores(session):
    img = _row(session, 1, _emb(1, 1))
    session.exec(select(SemanticScore)).first().preference = 0.9
    assert pref.apply_preferences(session) == 1
    assert _prefs(session)[img.id] is None


def test_reset_removes_the_model_and_the_scores(session):
    _labelled(session, 15)
    pref.train_preferences(session)
    pref.reset_preferences(session)
    assert session.exec(select(PreferenceModel)).all() == []
    assert set(_prefs(session).values()) == {None}


def test_status_reports_examples_and_the_active_model(session):
    _labelled(session, 9)
    st = pref.status(session)
    assert st["n_positive"] == 9 and st["n_negative"] == 9 and st["trained"] is False
    assert st["can_train"] is True and st["min_per_class"] == pref.MIN_PER_CLASS
    pref.train_preferences(session)
    st = pref.status(session)
    assert st["trained"] is True and st["cv_accuracy"] > 0.8 and st["trained_at"]


def test_stored_vectors_of_another_size_are_ignored(session):
    _labelled(session, 15)
    pref.train_preferences(session)
    odd = _row(session, 7777, np.ones(64, dtype=np.float32) / 8.0)
    pref.apply_preferences(session)
    assert _prefs(session)[odd.id] is None


def test_logistic_fit_separates_a_simple_problem():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(1, 1, (60, 8)), rng.normal(-1, 1, (60, 8))])
    y = np.array([1] * 60 + [0] * 60)
    w, b = pref.fit_logistic(X, y, lam=1.0)
    acc = float((((X @ w + b) > 0).astype(int) == y).mean())
    assert acc > 0.9


# --------------------------------------------------------------- API / pipeline
@pytest.fixture()
def env(tmp_path):
    app = create_app(str(tmp_path / "p.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    with get_session(app.state.db) as s:
        _labelled(s, 14)
    return types.SimpleNamespace(app=app, client=client)


def test_api_status_learn_and_reset(env):
    st = env.client.get("/api/preferences").json()
    assert st["trained"] is False and st["n_positive"] == 14 and st["can_train"] is True
    r = env.client.post("/api/preferences/learn")
    assert r.status_code == 200 and r.json()["status"] == "trained"
    assert env.client.get("/api/preferences").json()["trained"] is True
    with get_session(env.app.state.db) as s:
        assert None not in set(_prefs(s).values())
    assert env.client.post("/api/preferences/reset").status_code == 200
    assert env.client.get("/api/preferences").json()["trained"] is False


def test_api_learn_reports_when_there_is_not_enough_to_learn_from(tmp_path):
    app = create_app(str(tmp_path / "q.db"))
    c = TestClient(app)
    c.headers["X-PCUR-Token"] = app.state.token
    r = c.post("/api/preferences/learn")
    assert r.status_code == 200 and r.json()["status"] == "not_enough"


def test_api_preference_endpoints_need_a_token(env):
    anon = TestClient(env.app)
    assert anon.post("/api/preferences/learn").status_code == 401
    assert anon.post("/api/preferences/reset").status_code == 401
    assert anon.get("/api/preferences").status_code == 401


def test_learning_changes_the_highlight_ranking(env):
    with get_session(env.app.state.db) as s:
        good = _row(s, 8001, _emb(8001, +1))
        bad = _row(s, 8002, _emb(8002, -1))
        # The model rates the bad one slightly higher; taste should overturn that.
        s.exec(select(SemanticScore).where(SemanticScore.image_id == good.id)).one().aesthetic = 0.50
        s.exec(select(SemanticScore).where(SemanticScore.image_id == bad.id)).one().aesthetic = 0.55
        good_id, bad_id = good.id, bad.id
    env.client.post("/api/features", json={"highlight_preference_weight": 1.0})
    env.client.post("/api/preferences/learn")
    order = [it["id"] for it in env.client.get("/api/images", params={"highlights": "true"}).json()["items"]]
    assert order.index(good_id) < order.index(bad_id)


def test_the_pipeline_scores_new_photos_with_the_active_model(tmp_path, monkeypatch):
    from photocure.inference import pipeline
    called = []
    monkeypatch.setattr(pipeline, "apply_preferences",
                        lambda session, only_missing=False: called.append(only_missing) or 0)
    from photocure.config import CuratorConfig
    from photocure.db.session import init_db
    from photocure.scanner.file_scanner import FileScanner
    from .conftest import make_image
    db = tmp_path / "pp.db"
    cfg = CuratorConfig()
    cfg.db_path, cfg.device.device = db, "cpu"
    init_db(db, drop=True)
    make_image(tmp_path / "lib" / "a.jpg", rng_seed=1)
    FileScanner(cfg).scan_directory(tmp_path / "lib", db_path=db, show_progress=False)
    assert pipeline.InferencePipeline(cfg).run(db_path=db, show_progress=False).processed == 1
    assert called == [True]
    assert dequantize_embedding("") is None
