"""AI runs on New photos only, and in Trash when asked; never on finished photos."""
from __future__ import annotations

from pathlib import Path

import pytest

from photocure.db import models as M
from photocure.db.session import get_session

from .test_ui_api import env  # noqa: F401  (shared fixture)

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "photocure" / "ui" / "static"


def _setup(env):
    ids = sorted(i["id"] for i in env.client.get("/api/images?limit=50").json()["items"])
    a, b, c = ids[:3]
    with get_session(env.db) as s:
        for iid in (a, b, c):
            s.get(M.ImageRecord, iid).infer_status = "pending"
        s.get(M.ImageRecord, b).stage = "library"
        s.get(M.ImageRecord, b).route_status = "executed"
        t = s.get(M.ImageRecord, c)
        t.stage, t.route_status, t.scan_status, t.delete_approved = "trash", "trashed", "missing", True
        s.commit()
    return a, b, c


def _status(env, iid):
    with get_session(env.db) as s:
        r = s.get(M.ImageRecord, iid)
        return r.infer_status, r.route_status


def test_infer_default_touches_only_new_rows(env):
    a, b, c = _setup(env)
    r = env.client.post("/api/infer", json={"device": "cpu"})
    assert r.status_code == 200
    assert _status(env, a)[0] == "inferred"
    assert _status(env, b)[0] == "pending"  # library photo untouched
    assert _status(env, c)[0] == "pending"  # trash untouched unless asked


def test_infer_trash_stage_reanalyses_trashed_rows_only(env):
    a, b, c = _setup(env)
    with get_session(env.db) as s:
        s.get(M.ImageRecord, c).infer_status = "inferred"
        s.commit()
    r = env.client.post("/api/infer", json={"device": "cpu", "stage": "trash"})
    assert r.status_code == 200
    assert r.json()["processed"] == 1
    assert _status(env, a)[0] == "pending"  # a new photo is not touched by a trash run
    assert _status(env, b)[0] == "pending"
    assert _status(env, c) == ("inferred", "trashed")


def test_infer_library_stage_rejected(env):
    assert env.client.post("/api/infer", json={"device": "cpu", "stage": "library"}).status_code == 422


def test_rescore_endpoint_is_gone(env):
    assert env.client.post("/api/rescore", json={}).status_code in (404, 405)


def test_trash_reanalyse_retires_a_file_removed_by_hand(env):
    a, b, c = _setup(env)
    with get_session(env.db) as s:
        s.get(M.ImageRecord, c).infer_status = "inferred"
        path = s.get(M.ImageRecord, c).path
        s.commit()
    Path(path).unlink()
    r = env.client.post("/api/infer", json={"device": "cpu", "stage": "trash"})
    assert r.status_code == 200
    assert r.json()["processed"] == 0 and r.json()["missing"] == 1
    assert _status(env, c)[1] == "purged"


def test_ui_has_no_reanalyse_buttons():
    html = (STATIC / "index.html").read_text()
    js = (STATIC / "app.js").read_text()
    for needle in ("btn-recheck", "btn-reanalyse-all", "btn-rescore", "data-reanalyze", "/api/rescore"):
        assert needle not in html and needle not in js, needle
