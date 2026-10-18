# ruff: noqa: E501
"""Event stories: scale tags come from stored embeddings, and selection balances them."""
from __future__ import annotations

import types
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from photocure.db.models import EventGroup, ImageRecord, SemanticScore
from photocure.db.repository import quantize_embedding
from photocure.db.session import get_session
from photocure.narrative import SCALE_PROMPTS, SCALES
from photocure.search import SearchUnavailable
from photocure.ui.app import create_app

from .conftest import make_image

DIM = 8


def _dt(d):
    return datetime(2024, 4, d, 12, tzinfo=timezone.utc)


class FakeEncoder:
    """Maps every scale prompt to the basis vector of its scale."""

    def __init__(self):
        self.index = {p: i for i, s in enumerate(SCALES) for p in SCALE_PROMPTS[s]}

    def encode(self, text):
        v = np.zeros(DIM, dtype=np.float32)
        v[self.index[text]] = 1.0
        return v


class BrokenEncoder:
    def encode(self, text):
        raise SearchUnavailable


@pytest.fixture()
def env(tmp_path):
    app = create_app(str(tmp_path / "s.db"))
    app.state.cfg.cache_dir = tmp_path / "cache"
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    lib = tmp_path / "photos"
    for i in range(4):
        make_image(lib / f"p{i}.jpg", rng_seed=40 + i)
    assert client.post("/api/scan", json={"path": str(lib)}).status_code == 200
    assert client.post("/api/infer", json={"device": "cpu"}).status_code == 200
    shapes = ["wide", "closeup", "medium", "medium"]
    scores = [0.5, 0.6, 0.9, 0.8]
    with get_session(app.state.db) as s:
        imgs = list(s.exec(select(ImageRecord).order_by(ImageRecord.filename)).all())
        sems = {x.image_id: x for x in s.exec(select(SemanticScore)).all()}
        ev = EventGroup(start=_dt(1), end=_dt(4), label="Trip", image_count=4)
        s.add(ev)
        s.flush()
        for n, img in enumerate(imgs):
            sem = sems[img.id]
            vec = np.zeros(DIM, dtype=np.float32)
            vec[SCALES.index(shapes[n])] = 1.0
            sem.embedding, sem.backend, sem.aesthetic = quantize_embedding(vec.tolist()), "clip", scores[n]
            img.taken_at, img.taken_at_source, img.event_id = _dt(1 + n), "exif:dtoriginal", ev.id
        ids = [i.id for i in imgs]
    app.state.text_encoder = FakeEncoder()
    return types.SimpleNamespace(app=app, client=client, ids=ids)


def test_balanced_story_covers_the_scales_and_reads_in_time_order(env):
    r = env.client.get("/api/highlights/events", params={"per_event": 3, "balanced": "true"}).json()
    assert r["balanced"] is True and r["note"] == ""
    ev = r["events"][0]
    assert [i["id"] for i in ev["items"]] == [env.ids[0], env.ids[1], env.ids[2]]
    assert [i["scale"] for i in ev["items"]] == ["wide", "closeup", "medium"]
    assert ev["coverage"] == {"wide": 1, "closeup": 1, "medium": 1}


def test_default_view_is_unchanged(env):
    r = env.client.get("/api/highlights/events", params={"per_event": 3}).json()
    assert r["balanced"] is False
    ev = r["events"][0]
    assert [i["id"] for i in ev["items"]] == [env.ids[2], env.ids[3], env.ids[1]]
    assert "scale" not in ev["items"][0] and "coverage" not in ev


def test_placeholder_embeddings_never_get_scale_tags(env):
    with get_session(env.app.state.db) as s:
        for sem in s.exec(select(SemanticScore)).all():
            sem.backend = "stub"
    r = env.client.get("/api/highlights/events", params={"per_event": 3, "balanced": "true"}).json()
    assert r["balanced"] is False and "AI model" in r["note"]
    assert [i["id"] for i in r["events"][0]["items"]] == [env.ids[2], env.ids[3], env.ids[1]]


def test_missing_model_falls_back_with_a_note(env):
    env.app.state.text_encoder = BrokenEncoder()
    r = env.client.get("/api/highlights/events", params={"per_event": 3, "balanced": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["balanced"] is False and "AI model" in body["note"]
    assert len(body["events"][0]["items"]) == 3


def test_slots_beyond_the_pool_return_the_whole_event(env):
    r = env.client.get("/api/highlights/events", params={"per_event": 12, "balanced": "true"}).json()
    assert len(r["events"][0]["items"]) == 4


def test_photos_without_a_usable_embedding_get_no_scale_tag(env):
    with get_session(env.app.state.db) as s:
        sem = s.exec(select(SemanticScore).where(SemanticScore.image_id == env.ids[2])).one()
        sem.backend = "stub"
    r = env.client.get("/api/highlights/events", params={"per_event": 3, "balanced": "true"}).json()
    ev = r["events"][0]
    assert r["balanced"] is True
    by_id = {i["id"]: i["scale"] for i in ev["items"]}
    assert by_id[env.ids[2]] is None, "a placeholder photo must not be labelled medium"
    assert by_id[env.ids[0]] == "wide" and by_id[env.ids[1]] == "closeup"
    assert ev["coverage"] == {"wide": 1, "closeup": 1}


def test_scale_labelling_runs_outside_the_encoder_lock():
    import inspect

    from photocure.ui import app as ui_app
    src = inspect.getsource(ui_app.create_app).splitlines()
    start = next(i for i, ln in enumerate(src) if "def scale_of(ids):" in ln)
    block = src[start:start + 8]
    with_indent = next(len(ln) - len(ln.lstrip()) for ln in block if "with encoder_lock:" in ln)
    ret_indent = next(len(ln) - len(ln.lstrip()) for ln in block if "narrative.scale_labels" in ln)
    assert ret_indent < with_indent + 4 and ret_indent == with_indent, "labelling must not hold the lock"
