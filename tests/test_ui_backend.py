"""Backend banner info, uncertainty explanations and the learn job."""
from __future__ import annotations

import time
import types

import pytest
from fastapi.testclient import TestClient

from photocure.db import models as M
from photocure.db.session import get_session
from photocure.services import Planner
from photocure.ui import app as ui_app
from photocure.ui.app import create_app

from .conftest import make_image


@pytest.fixture(autouse=True)
def _fresh_backend():
    """Detect the backend afresh for each test and leave nothing cached behind."""
    from photocure.inference import backends as bk
    bk.reset_backend_cache()
    yield
    bk.reset_backend_cache()


@pytest.fixture()
def env(tmp_path):
    app = create_app(str(tmp_path / "ui.db"))
    app.state.cfg.cache_dir = tmp_path / "cache"
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    lib = tmp_path / "photos"
    for i in range(1, 4):
        make_image(lib / f"p{i}.jpg", rng_seed=i)
    assert client.post("/api/scan", json={"path": str(lib)}).status_code == 200
    assert client.post("/api/infer", json={"device": "cpu"}).status_code == 200
    return types.SimpleNamespace(app=app, client=client, db=app.state.db)


def wait_job(client, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def seed_uncertain(env, image_id, top1, s1, top2, s2, dist=None):
    import json
    with get_session(env.db) as s:
        rec = s.get(M.ImageRecord, image_id)
        rec.category = "uncertain"
        rec.manual_bucket = None
        sem = rec.semantic
        sem.top1_category, sem.top1_score = top1, s1
        sem.top2_category, sem.top2_score = top2, s2
        sem.score_margin = s1 - s2
        sem.distribution = json.dumps(dist or {top1: s1, top2: s2, "scenery": 0.1})


# ------------------------------------------------------------------ config
def test_config_reports_backend_and_thresholds(env):
    cfg = env.client.get("/api/config").json()
    b = cfg["backend"]
    for key in ("device", "semantic_backend", "face_backend", "semantic_is_stub", "forced_by_env",
                "face_is_fallback", "hint", "stub_rows"):
        assert key in b
    assert b["semantic_is_stub"] is True, b
    assert b["semantic_backend"] == "stub", b
    assert b["hint"], b
    t = cfg["thresholds"]
    assert 0 < t["min_confidence"] < 1 and 0 < t["min_margin"] < 1


def test_config_uses_planner_backend_info(env, monkeypatch):
    fake = {"device": "mps", "semantic_backend": "clip", "face_backend": "insightface",
            "semantic_is_stub": False, "face_is_fallback": False, "hint": "x", "stub_rows": 7}
    monkeypatch.setattr(Planner, "backend_info", lambda self: dict(fake), raising=False)
    b = env.client.get("/api/config").json()["backend"]
    assert b["semantic_is_stub"] is False and b["stub_rows"] == 7 and b["device"] == "mps"


def test_config_survives_backend_info_failure(env, monkeypatch):
    def boom(self):
        raise RuntimeError("no backend")
    monkeypatch.setattr(Planner, "backend_info", boom, raising=False)
    b = env.client.get("/api/config").json()["backend"]
    assert "semantic_is_stub" in b and "hint" in b


def test_new_categories_are_listed(env):
    vals = {c["value"] for c in env.client.get("/api/config").json()["categories"]}
    assert {"screenshot", "document", "uncertain"} <= vals


# ------------------------------------------------------------------- infer
def test_infer_accepts_mps_and_rejects_unknown_device(env):
    assert env.client.post("/api/infer", json={"device": "mps"}).status_code == 200
    assert env.client.post("/api/infer", json={"device": "tpu"}).status_code == 422


# ------------------------------------------------------------------- items
def test_items_carry_top2_fields(env):
    it = env.client.get("/api/images").json()["items"][0]
    assert "top2" in it and "top2_score" in it and "margin" in it
    assert "uncertain_reason" not in it


def test_uncertain_item_has_reason_and_guesses(env):
    seed_uncertain(env, 1, "food", 0.38, "scenery", 0.35,
                   {"food": 0.38, "scenery": 0.35, "people": 0.2, "animal": 0.07})
    it = next(i for i in env.client.get("/api/images?category=uncertain").json()["items"]
              if i["id"] == 1)
    assert it["uncertain_reason"]["text"]
    assert it["uncertain_reason"]["kind"]
    names = [g["name"] for g in it["guesses"]]
    assert names == ["food", "scenery", "people"]
    assert it["guesses"][0]["score"] == pytest.approx(0.38)
    assert it["top2"] == "scenery"


def test_manual_choice_removes_uncertain_reason(env):
    seed_uncertain(env, 2, "food", 0.3, "scenery", 0.29)
    env.client.post("/api/images/2/reassign", json={"bucket": "food"})
    it = next(i for i in env.client.get("/api/images").json()["items"] if i["id"] == 2)
    assert "uncertain_reason" not in it


def test_explain_fallback_sentences():
    sem = types.SimpleNamespace(top1_category="food", top1_score=0.31, top2_category="scenery",
                                top2_score=0.30, score_margin=0.01, distribution="{}")
    low = ui_app._explain_uncertain(sem, None)
    assert low["kind"] == "low_confidence" and "Food" in low["text"] and "31%" in low["text"]
    sem.top1_score, sem.score_margin = 0.5, 0.02
    close = ui_app._explain_uncertain(sem, None)
    assert close["kind"] == "close_call"
    assert "Food" in close["text"] and "Scenery" in close["text"]
    assert ui_app._explain_uncertain(None, None)["kind"] == "no_data"


def test_guesses_fall_back_when_distribution_empty():
    sem = types.SimpleNamespace(top1_category="a_b", top1_score=0.4, top2_category="c",
                                top2_score=0.3, distribution="not json")
    assert [g["name"] for g in ui_app._guesses(sem)] == ["a_b", "c"]


# ---------------------------------------------------------------- overview
def test_overview_counts_manual_choices(env):
    assert env.client.get("/api/overview").json()["manual_count"] == 0
    env.client.post("/api/images/1/reassign", json={"bucket": "food"})
    env.client.post("/api/images/2/reassign", json={"bucket": "people"})
    assert env.client.get("/api/overview").json()["manual_count"] == 2


# ------------------------------------------------------------------- learn
def test_learn_runs_as_a_job(env, monkeypatch):
    calls = []

    def fake(self):
        calls.append(1)
        return {"updated": 5, "classes_used": 2, "examples": 4, "moved_out_of_uncertain": 3}
    monkeypatch.setattr(Planner, "learn_from_corrections", fake, raising=False)
    r = env.client.post("/api/learn")
    assert r.status_code == 202
    job = wait_job(env.client, r.json()["job_id"])
    assert job["status"] == "done"
    assert job["result"]["moved_out_of_uncertain"] == 3 and job["result"]["ok"] is True
    assert calls == [1]
    assert env.client.get("/api/progress").json()["busy"] is False


def test_learn_501_when_unavailable(env, monkeypatch):
    monkeypatch.delattr(Planner, "learn_from_corrections", raising=False)
    assert env.client.post("/api/learn").status_code == 501


def test_learn_needs_token_and_respects_busy_lock(env, monkeypatch):
    monkeypatch.setattr(Planner, "learn_from_corrections", lambda self: {}, raising=False)
    assert TestClient(env.app).post("/api/learn").status_code == 401
    assert ui_app._BUSY.acquire(blocking=False)
    try:
        assert env.client.post("/api/learn").status_code == 409
    finally:
        ui_app._BUSY.release()


def test_learn_failure_is_reported_and_releases_lock(env, monkeypatch):
    def boom(self):
        raise RuntimeError("model missing")
    monkeypatch.setattr(Planner, "learn_from_corrections", boom, raising=False)
    job = wait_job(env.client, env.client.post("/api/learn").json()["job_id"])
    assert job["status"] == "error" and "model missing" in job["error"]
    assert env.client.get("/api/progress").json()["busy"] is False
