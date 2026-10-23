"""Analyze only costs the new photos; Reanalyse re-runs the AI on purpose."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from photocure.db import models as M
from photocure.db.session import get_session
from photocure.ui.app import create_app

from .conftest import make_image

STATIC = Path(__file__).resolve().parent.parent / "photocure" / "ui" / "static"
T0 = datetime(2024, 5, 1, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def env(tmp_path):
    app = create_app(str(tmp_path / "inc.db"))
    app.state.cfg.cache_dir = tmp_path / "cache"
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    lib = tmp_path / "photos"
    for i in range(4):
        make_image(lib / f"p{i}.jpg", rng_seed=i + 1)
    assert client.post("/api/scan", json={"path": str(lib)}).status_code == 200
    return app, client, lib


def infer(client, **kw):
    r = client.post("/api/infer", json={"device": "cpu", **kw})
    assert r.status_code == 200
    return r.json()


def rows(app):
    with get_session(app.state.db) as s:
        return [SimpleNamespace(id=r.id, infer_status=r.infer_status, delete_approved=r.delete_approved)
                for r in s.exec(select(M.ImageRecord).order_by(M.ImageRecord.id)).all()]


def test_second_infer_processes_nothing_then_only_the_new_photo(env):
    app, client, lib = env
    first = infer(client)
    assert first["processed"] == 4 and first["new_ids_count"] == 4
    assert "ids" not in first
    second = infer(client)
    assert second["processed"] == 0 and second["new_ids_count"] == 0
    assert app.state.last_analyzed_ids == set()
    make_image(lib / "new.jpg", rng_seed=99)
    client.post("/api/scan", json={"path": str(lib)})
    third = infer(client)
    assert third["processed"] == 1 and len(app.state.last_analyzed_ids) == 1


def _set_times(app, times):
    with get_session(app.state.db) as s:
        for rec in s.exec(select(M.ImageRecord).order_by(M.ImageRecord.id)).all():
            rec.taken_at = times.get(rec.id)
            rec.taken_at_source = "exif"


def _bursts(app):
    with get_session(app.state.db) as s:
        return {b.id: (b.start_time, b.image_count, b.best_shot_id)
                for b in s.exec(select(M.BurstGroup)).all()}


def test_only_new_cluster_keeps_distant_group_untouched(env, monkeypatch):
    app, client, lib = env
    # Identical embeddings so close frames always cluster.
    from photocure.clustering import burst_detector as bd
    monkeypatch.setattr(bd.BurstDetector, "cluster", _time_only_cluster)
    for i in range(4, 8):
        make_image(lib / f"p{i}.jpg", rng_seed=i + 1)
    client.post("/api/scan", json={"path": str(lib)})
    infer(client)
    ids = [r.id for r in rows(app)]
    far = timedelta(hours=5)
    _set_times(app, {ids[0]: T0, ids[1]: T0 + timedelta(seconds=5),
                     ids[2]: T0 + far, ids[3]: T0 + far + timedelta(seconds=5)})
    assert client.post("/api/cluster", json={}).json()["burst_groups"] == 2
    before = _bursts(app)
    assert len(before) == 2
    group_a = next(k for k, v in before.items() if v[0] == T0)
    # a new frame near group B
    make_image(lib / "z.jpg", rng_seed=77)
    client.post("/api/scan", json={"path": str(lib)})
    app.state.last_analyzed_ids = set()
    infer(client)
    with get_session(app.state.db) as s:
        new = s.exec(select(M.ImageRecord).where(M.ImageRecord.id.in_(app.state.last_analyzed_ids))).all()
        assert len(new) == 1
        new[0].taken_at = T0 + far + timedelta(seconds=10)
        new[0].taken_at_source = "exif"
        new_id = new[0].id
    # drop the other rows from "new" so only z is the neighbourhood seed
    r = client.post("/api/cluster", json={"only_new": True})
    assert r.status_code == 200
    after = _bursts(app)
    assert group_a in after and after[group_a] == before[group_a]
    group_b = [v for k, v in after.items() if k != group_a]
    assert len(after) == 2 and group_b[0][1] == 3
    with get_session(app.state.db) as s:
        assert s.get(M.ImageRecord, new_id).burst_id is not None
        assert s.get(M.ImageRecord, new_id).burst_id != group_a


def _time_only_cluster(self, items):
    items = sorted((i for i in items if i.taken_at is not None), key=lambda i: i.taken_at)
    out, cur = [], []
    for it in items:
        if cur and (it.taken_at - cur[-1].taken_at).total_seconds() > self.t.window_seconds:
            out.append(cur)
            cur = []
        cur.append(it)
    out.append(cur)
    return [[i.image_id for i in g] for g in out if len(g) >= self.t.min_group_size]


def test_only_new_without_memory_falls_back_to_full(env):
    app, client, _ = env
    infer(client)
    app.state.last_analyzed_ids = set()
    r = client.post("/api/cluster", json={"only_new": True})
    assert r.status_code == 200 and r.json()["ok"] is True
    del app.state.last_analyzed_ids
    assert client.post("/api/cluster", json={"only_new": True}).status_code == 200


def test_approve_only_new_touches_only_last_analyzed_junk(env):
    app, client, lib = env
    infer(client)
    old = rows(app)
    with get_session(app.state.db) as s:
        s.get(M.ImageRecord, old[0].id).manual_bucket = "blurry"
    make_image(lib / "new.jpg", rng_seed=50)
    client.post("/api/scan", json={"path": str(lib)})
    infer(client)
    new_id = next(iter(app.state.last_analyzed_ids))
    from photocure.ui.app import JUNK_VALUES
    junk = sorted(JUNK_VALUES)[0]
    with get_session(app.state.db) as s:
        s.get(M.ImageRecord, old[0].id).manual_bucket = junk
        s.get(M.ImageRecord, new_id).manual_bucket = junk
    r = client.post("/api/deletes/approve", json={"only_new": True}).json()
    assert r["changed"] == 1
    got = {x.id: x.delete_approved for x in rows(app)}
    assert got[new_id] is True and got[old[0].id] is False


def test_reanalyze_reprocesses_everything_and_keeps_rows(env):
    app, client, _ = env
    infer(client)
    n = len(rows(app))
    again = infer(client, reanalyze=True)
    assert again["processed"] == n
    assert len(rows(app)) == n
    assert all(r.infer_status == "inferred" for r in rows(app))
    assert infer(client)["processed"] == 0


def test_ui_has_reanalyse_buttons_and_skips_when_nothing_new():
    html = (STATIC / "index.html").read_text()
    js = (STATIC / "app.js").read_text()
    for bid in ("btn-recheck", "btn-reanalyse-all"):
        assert f'id="{bid}" data-busy class="btn"' in html
    assert "Nothing new to analyze. Use Reanalyse to re-check photos you already analyzed." in js
    assert '{ only_new: true }' in js and "reanalyze: true" in js
    seg = js[js.index('$("#btn-recheck").onclick'):js.index('$("#btn-reanalyse-all").onclick')]
    assert "/api/cluster" not in seg and "approveProposedDeletes" not in seg
    seg = js[js.index('$("#btn-reanalyse-all").onclick'):js.index('$("#btn-ingest-all").onclick')]
    assert "showConfirmDialog" in seg and "approveProposedDeletes" not in seg
