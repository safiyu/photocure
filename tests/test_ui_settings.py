"""GET and POST /api/settings, and saved settings reaching scan, infer and ingest."""
from __future__ import annotations

import types
from typing import get_args

import pytest
from fastapi.testclient import TestClient

from photocure.config import DEVICE_CHOICES
from photocure.inference import backends as bk
from photocure.services import settings_path
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
    for i in range(1, 3):
        make_image(lib / f"p{i}.jpg", rng_seed=i)
    return types.SimpleNamespace(app=app, client=client, db=app.state.db, lib=lib)


def fake_pipeline(monkeypatch, seen):
    class Fake:
        def __init__(self, cfg, backend=None):
            seen.append(cfg)

        def run(self, **kw):
            return types.SimpleNamespace(processed=0, faces_total=0, quality_computed=0,
                                         semantic_computed=0, failures=0, duration_s=0.0)
    import photocure.inference.pipeline as pl
    monkeypatch.setattr(pl, "InferencePipeline", Fake)


def fake_scanner(monkeypatch, seen):
    class Fake:
        def __init__(self, cfg):
            seen.append(cfg)

        def scan_directory(self, *a, **kw):
            return types.SimpleNamespace(scanned=0, new=0, duplicates=0, skipped_known=0,
                                         errors=0, duration_s=0.0)
    import photocure.scanner.file_scanner as fs
    monkeypatch.setattr(fs, "FileScanner", Fake)


def test_device_names_match_config():
    assert set(get_args(ui_app.DeviceName)) == set(DEVICE_CHOICES)


# --------------------------------------------------------------------- GET
def test_get_settings_shape(env):
    j = env.client.get("/api/settings").json()
    assert j["settings"] == {}
    eff = j["effective"]
    for k in ("scan_workers", "decode_workers", "torch_threads"):
        assert isinstance(eff[k], int) and eff[k] >= 1
    assert eff["device"] == j["selected"]
    ids = [a["id"] for a in j["accelerators"]]
    assert "cpu" in ids
    for a in j["accelerators"]:
        assert {"id", "label", "available", "reason", "install_hint"} <= set(a)
    assert "fallback_reason" in j and "requested" in j


def test_get_settings_needs_no_token(env):
    assert TestClient(env.app).get("/api/settings").status_code == 200


# -------------------------------------------------------------------- POST
def test_post_saves_and_get_reflects(env):
    r = env.client.post("/api/settings", json={
        "device": "cpu", "device_index": 0, "scan_workers": 4,
        "decode_workers": 3, "torch_threads": 0})
    assert r.status_code == 200
    j = r.json()
    assert j["settings"]["device"] == "cpu" and j["settings"]["decode_workers"] == 3
    assert j["effective"]["decode_workers"] == 3 and j["effective"]["scan_workers"] == 4
    assert j["effective"]["torch_threads"] >= 1
    assert settings_path(env.db).exists()
    again = env.client.get("/api/settings").json()
    assert again["settings"] == j["settings"]
    assert again["effective"]["device"] == "cpu"


def test_post_null_clears_a_key(env):
    env.client.post("/api/settings", json={"decode_workers": 6, "scan_workers": 2})
    j = env.client.post("/api/settings", json={"decode_workers": None}).json()
    assert "decode_workers" not in j["settings"] and j["settings"]["scan_workers"] == 2


@pytest.mark.parametrize(("body", "needle"), [
    ({"device": "quantum"}, "Unknown device"),
    ({"decode_workers": 999}, "between 0 and 128"),
    ({"scan_workers": -1}, "between 0 and 128"),
    ({"torch_threads": "abc"}, "whole number"),
    ({"scan_workers": True}, "whole number"),
    ({"scan_workers": 2.5}, "whole number"),
    ({"colour": "red"}, "Unknown setting"),
])
def test_post_rejects_bad_values_with_plain_message(env, body, needle):
    r = env.client.post("/api/settings", json=body)
    assert r.status_code == 400
    assert needle in r.json()["detail"]
    assert not settings_path(env.db).exists()


def test_post_needs_token(env):
    anon = TestClient(env.app)
    assert anon.post("/api/settings", json={"device": "cpu"}).status_code == 403
    assert anon.post("/api/settings", json={"device": "cpu"},
                     headers={"X-PCUR-Token": "nope"}).status_code == 403


def test_post_rejects_non_object_body(env):
    assert env.client.post("/api/settings", json=[1, 2]).status_code == 422


def test_post_redetects_backend_when_idle(env, monkeypatch):
    calls = []
    monkeypatch.setattr(bk, "reset_backend_cache", lambda: calls.append("reset"))
    env.client.post("/api/settings", json={"device": "cpu"})
    assert calls == ["reset"]
    calls.clear()
    assert ui_app._BUSY.acquire(blocking=False)
    try:
        assert env.client.post("/api/settings", json={"device": "auto"}).status_code == 200
    finally:
        ui_app._BUSY.release()
    assert calls == []


# ------------------------------------------------------ saved values applied
def test_infer_receives_saved_device_and_workers(env, monkeypatch):
    seen: list = []
    fake_pipeline(monkeypatch, seen)
    env.client.post("/api/settings", json={
        "device": "cpu", "decode_workers": 3, "torch_threads": 5})
    assert env.client.post("/api/infer", json={}).status_code == 200
    c = seen[-1]
    assert c.device.device == "cpu"
    assert c.device.decode_workers == 3 and c.device.torch_threads == 5


def test_infer_body_overrides_saved_without_changing_them(env, monkeypatch):
    seen: list = []
    fake_pipeline(monkeypatch, seen)
    env.client.post("/api/settings", json={"device": "cpu", "decode_workers": 3})
    r = env.client.post("/api/infer", json={
        "device": "mps", "decode_workers": 5, "device_index": 1, "torch_threads": 2})
    assert r.status_code == 200
    c = seen[-1]
    assert (c.device.device, c.device.decode_workers, c.device.device_index,
            c.device.torch_threads) == ("mps", 5, 1, 2)
    assert env.client.get("/api/settings").json()["settings"] == {
        "device": "cpu", "decode_workers": 3}


def test_infer_without_saved_settings_keeps_environment_defaults(env, monkeypatch):
    seen: list = []
    fake_pipeline(monkeypatch, seen)
    assert env.client.post("/api/infer", json={}).status_code == 200
    assert seen[-1].device.device == env.app.state.cfg.device.device


def test_infer_rejects_bad_override(env):
    assert env.client.post("/api/infer", json={"device": "tpu"}).status_code == 422
    assert env.client.post("/api/infer", json={"decode_workers": 500}).status_code == 422


def test_scan_receives_saved_workers_and_explicit_override_wins(env, monkeypatch):
    seen: list = []
    fake_scanner(monkeypatch, seen)
    env.client.post("/api/settings", json={"scan_workers": 4})
    assert env.client.post("/api/scan", json={"path": str(env.lib)}).status_code == 200
    assert seen[-1].scan_workers == 4
    env.client.post("/api/scan", json={"path": str(env.lib), "scan_workers": 9})
    assert seen[-1].scan_workers == 9
    env.client.post("/api/scan", json={"path": str(env.lib), "workers": 6})
    assert seen[-1].scan_workers == 6


def test_scan_auto_workers_resolves_to_a_positive_number(env, monkeypatch):
    seen: list = []
    fake_scanner(monkeypatch, seen)
    env.client.post("/api/settings", json={"scan_workers": 0})
    env.client.post("/api/scan", json={"path": str(env.lib)})
    assert seen[-1].scan_workers >= 1


def test_ingest_applies_saved_settings_to_every_step(env, monkeypatch):
    infer_seen: list = []
    scan_seen: list = []
    fake_pipeline(monkeypatch, infer_seen)
    fake_scanner(monkeypatch, scan_seen)
    env.client.post("/api/settings", json={"device": "cpu", "scan_workers": 3,
                                           "decode_workers": 2})
    r = env.client.post("/api/ingest", json={"path": str(env.lib)})
    assert r.status_code == 200
    assert scan_seen[-1].scan_workers == 3
    assert infer_seen[-1].device.device == "cpu" and infer_seen[-1].device.decode_workers == 2


def test_corrupt_saved_settings_do_not_break_runs(env, monkeypatch):
    seen: list = []
    fake_pipeline(monkeypatch, seen)
    settings_path(env.db).write_text("{not json", encoding="utf-8")
    assert env.client.post("/api/infer", json={}).status_code == 200
    assert env.client.get("/api/settings").json()["settings"] == {}


def test_changed_device_triggers_redetection(env):
    bk.reset_backend_cache()
    assert env.client.post("/api/infer", json={"device": "cpu"}).status_code == 200
    assert bk.detect_backend().requested == "cpu"
    assert env.client.post("/api/infer", json={"device": "auto"}).status_code == 200
    assert bk.detect_backend().requested == "auto"
