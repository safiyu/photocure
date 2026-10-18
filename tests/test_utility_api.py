# ruff: noqa: E501
"""Utility vault API: the group filter, the config flag and the re-check endpoint."""
from __future__ import annotations

import time
import types

import pytest
from fastapi.testclient import TestClient

from photocure.services import Planner
from photocure.ui.app import create_app

from .conftest import make_image


@pytest.fixture()
def env(tmp_path):
    app = create_app(str(tmp_path / "ui.db"))
    app.state.cfg.cache_dir = tmp_path / "cache"
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    lib = tmp_path / "photos"
    for i in range(1, 5):
        make_image(lib / f"p{i}.jpg", rng_seed=i)
    assert client.post("/api/scan", json={"path": str(lib)}).status_code == 200
    assert client.post("/api/infer", json={"device": "cpu"}).status_code == 200
    return types.SimpleNamespace(app=app, client=client)


def _ids(env):
    return [i["id"] for i in env.client.get("/api/images").json()["items"]]


def test_config_flags_the_utility_family(env):
    cats = {c["value"]: c for c in env.client.get("/api/config").json()["categories"]}
    for name in ("utility", "screenshot", "document", "slide"):
        assert cats[name]["utility"] is True and cats[name]["junk"] is False, name
    assert cats["food"]["utility"] is False and cats["graphic"]["junk"] is True


def test_utility_group_lists_the_family_and_junk_no_longer_does(env):
    a, b, c, d = _ids(env)
    for image_id, bucket in ((a, "document"), (b, "utility"), (c, "slide"), (d, "graphic")):
        assert env.client.post(f"/api/images/{image_id}/reassign", json={"bucket": bucket}).status_code == 200
    vault = {i["id"] for i in env.client.get("/api/images?group=utility").json()["items"]}
    junk = {i["id"] for i in env.client.get("/api/images?group=junk").json()["items"]}
    assert vault == {a, b, c}
    assert junk == {d}


def test_unknown_group_is_still_rejected(env):
    assert env.client.get("/api/images?group=bogus").status_code == 422


def _wait_job(client, job_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.05)
    raise TimeoutError


def test_rescore_with_the_placeholder_model_explains_itself(env):
    r = env.client.post("/api/rescore")
    assert r.status_code == 200
    body = r.json()
    assert body["updated"] == 0 and "AI model" in body["note"]


def test_rescore_runs_as_a_background_job_when_the_real_model_is_active(env, monkeypatch):
    fake = {"device": "cpu", "semantic_backend": "clip", "face_backend": "yunet",
            "semantic_is_stub": False, "face_is_fallback": False, "hint": "x", "stub_rows": 0}
    monkeypatch.setattr(Planner, "backend_info", lambda self: dict(fake), raising=False)
    monkeypatch.setattr("photocure.inference.semantic_engine.rescore_from_embeddings",
                        lambda session, cfg=None, backend=None: 7)
    r = env.client.post("/api/rescore")
    assert r.status_code == 202
    job = _wait_job(env.client, r.json()["job_id"])
    assert job["status"] == "done" and job["result"] == {"ok": True, "updated": 7}


def test_rescore_refuses_while_another_task_runs(env, monkeypatch):
    fake = {"device": "cpu", "semantic_backend": "clip", "face_backend": "yunet",
            "semantic_is_stub": False, "face_is_fallback": False, "hint": "x", "stub_rows": 0}
    monkeypatch.setattr(Planner, "backend_info", lambda self: dict(fake), raising=False)
    from photocure.ui import app as ui_app
    assert ui_app._BUSY.acquire(blocking=False)
    try:
        assert env.client.post("/api/rescore").status_code == 409
    finally:
        ui_app._BUSY.release()
