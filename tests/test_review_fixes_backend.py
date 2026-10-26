"""Fixes from the final review of the HTTP layer."""
from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

from photocure.db import models as M
from photocure.db.session import get_session
from photocure.ui.app import create_app

from .test_ui_api import env, wait_job  # noqa: F401  (shared fixture)


def test_similar_for_a_photo_without_an_embedding_is_not_a_server_error(env):
    with get_session(env.db) as s:
        r = M.ImageRecord(path="/p/new.jpg", filename="new.jpg", infer_status="pending")
        s.add(r)
        s.commit()
        iid = r.id
    assert env.client.get(f"/api/images/{iid}/similar").status_code == 409


def test_watch_folder_must_stay_inside_the_allowed_folders(tmp_path, monkeypatch):
    allowed = tmp_path / "photos"
    allowed.mkdir()
    monkeypatch.setenv("PCUR_BROWSE_ROOTS", str(allowed))
    app = create_app(str(tmp_path / "ui.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    r = client.post("/api/features", json={"watch_enabled": True, "watch_path": str(tmp_path)})
    assert r.status_code == 403
    assert client.post("/api/features", json={"watch_path": str(allowed)}).status_code == 200


def test_plan_summary_reports_the_library_folder(env):
    assert env.client.get("/api/plan").json()["organize_root"] == str(env.out.resolve())


def test_odd_inputs_are_client_errors_not_500s(env):
    assert env.client.get("/api/images", params={"date_to": "9999-12-31"}).status_code in (200, 422)
    many = ",".join(["people"] * 5000)
    assert env.client.get("/api/images", params={"categories": many}).status_code in (200, 422)
    assert env.client.get(f"/api/faces/{'9' * 5000}.jpg").status_code == 404


def _fake_scanner(monkeypatch, fail_ids):
    import photocure.clean_export as ce

    class Fake:
        def __init__(self, *a, **k):
            pass

        def scan_image(self, iid, bgr):
            if iid in fail_ids:
                raise RuntimeError("cannot decode")
            return None

        def release(self):
            pass

    monkeypatch.setattr(ce, "PiiScanner", Fake)


def test_a_photo_whose_privacy_scan_failed_is_not_exported_as_clean(env, monkeypatch):
    ids = sorted(i["id"] for i in env.client.get("/api/images").json()["items"])
    _fake_scanner(monkeypatch, {ids[0]})
    r = env.client.post("/api/export/clean/scan", json={"ids": ids[:2]})
    j = wait_job(env.client, r.json()["job_id"])
    assert j["result"]["ok"] is True and j["result"]["failed"] == [ids[0]]
    assert ids[0] not in env.app.state.pii and env.app.state.pii[ids[1]] is False
    body = {"ids": [ids[0]], "output_dir": str(env.tmp / "share"), "policy": "blur"}
    assert env.client.post("/api/export/clean", json=body).status_code == 409


def test_reset_forgets_scan_results_and_the_last_analysis(env):
    env.app.state.pii[1] = False
    env.app.state.last_analyzed_ids = {1, 2}
    assert env.client.post("/api/reset").status_code == 200
    assert env.app.state.pii == {} and not env.app.state.last_analyzed_ids
