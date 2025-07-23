# ruff: noqa: E501
"""The burst detail endpoint reports the group matrix."""
from __future__ import annotations

import types
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from photocure.db import models as M
from photocure.db.session import get_session
from photocure.ui.app import create_app

from .conftest import make_image


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


def seed_burst(env, frames):
    """frames[i] is a list of (x, eye_status) faces for the i-th image of the burst."""
    with get_session(env.db) as s:
        recs = s.query(M.ImageRecord).order_by(M.ImageRecord.id).all()
        bg = M.BurstGroup(name="Burst_test", image_count=len(frames), best_shot_id=recs[0].id)
        s.add(bg)
        s.flush()
        ids = []
        for i, faces in enumerate(frames):
            rec = recs[i]
            rec.burst_id = bg.id
            rec.taken_at = datetime(2024, 1, 1, 12, 0, i, tzinfo=timezone.utc)
            ids.append(rec.id)
            for x, status in faces:
                s.add(M.FaceDetection(
                    image_id=rec.id, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1, confidence=0.9,
                    x1=x, y1=0.3, x2=x + 0.1, y2=0.5, eye_status=status,
                    eye_closed_score=0.9 if status == "closed" else 0.1))
        s.commit()
        return bg.id, ids


def test_everyone_open_burst_reports_a_perfect_frame(env):
    bid, ids = seed_burst(env, [[(0.1, "open"), (0.5, "open")]] * 3)
    g = env.client.get(f"/api/burst/{bid}").json()["group"]
    assert g["status"] == "perfect" and g["subject_count"] == 2
    assert g["best_id"] in ids and g["partner_id"] is None
    assert [f["id"] for f in g["frames"]] == ids
    assert all(0 <= f["min_quality"] <= 1 and len(f["subjects"]) == 2 for f in g["frames"])


def test_complementary_blinks_report_a_composite_pair(env):
    frames = [[(0.1, "closed"), (0.5, "open")], [(0.1, "open"), (0.5, "closed")], [(0.1, "closed"), (0.5, "closed")]]
    bid, ids = seed_burst(env, frames)
    g = env.client.get(f"/api/burst/{bid}").json()["group"]
    assert g["status"] == "composite"
    assert g["best_id"] in ids and g["partner_id"] in ids and g["best_id"] != g["partner_id"]


def test_burst_without_faces_is_not_a_group(env):
    bid, _ = seed_burst(env, [[], [], []])
    g = env.client.get(f"/api/burst/{bid}").json()["group"]
    assert g == {"status": "not_group", "subject_count": 0, "best_id": None, "partner_id": None, "frames": []}


def test_unknown_burst_is_still_a_404(env):
    assert env.client.get("/api/burst/99999").status_code == 404


def test_burst_returns_face_matrix_with_consistent_subject_columns(env):
    frames = [
        [(0.1, "open"), (0.5, "closed")],
        [(0.1, "closed"), (0.5, "open")],
        [(0.1, "open"), (0.5, "open")],
    ]
    bid, ids = seed_burst(env, frames)
    data = env.client.get(f"/api/burst/{bid}").json()
    assert "face_matrix" in data
    fm = data["face_matrix"]
    assert len(fm["subjects"]) == 2
    assert fm["subjects"][0]["name"] == "Person 1"
    assert fm["subjects"][1]["name"] == "Person 2"
    assert len(fm["rows"]) == 3
    for idx, row in enumerate(fm["rows"]):
        assert row["id"] == ids[idx]
        assert "is_best_shot" in row
        assert "delete_approved" in row
        assert len(row["faces"]) == 2
        p1 = row["faces"][0]
        p2 = row["faces"][1]
        assert p1["subject_idx"] == 0
        assert p2["subject_idx"] == 1
        assert p1["url"].startswith("/api/faces/")
        assert p2["url"].startswith("/api/faces/")
