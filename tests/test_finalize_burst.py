from __future__ import annotations

import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from photocure.burst_actions import finalize_all_bursts, finalize_burst
from photocure.db import models as M
from photocure.db.session import get_session
from photocure.services import Planner
from photocure.ui.app import create_app
from .conftest import make_image


@pytest.fixture()
def burst_env(tmp_path):
    app = create_app(str(tmp_path / "ui.db"))
    app.state.cfg.cache_dir = tmp_path / "cache"
    lib_root = tmp_path / "library"
    lib_root.mkdir()
    app.state.cfg.library_root = lib_root
    from photocure.services import save_settings
    save_settings(app.state.db, {"library_root": str(lib_root)})

    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    for i in range(1, 4):
        make_image(photos_dir / f"frame_{i}.jpg", rng_seed=i)

    assert client.post("/api/scan", json={"path": str(photos_dir)}).status_code == 200
    assert client.post("/api/infer", json={"device": "cpu"}).status_code == 200

    # Seed a burst group with 3 members
    with get_session(app.state.db) as s:
        recs = s.exec(select(M.ImageRecord).order_by(M.ImageRecord.id)).all()
        bg = M.BurstGroup(name="Burst_20240101_0001", image_count=3, best_shot_id=recs[0].id)
        s.add(bg)
        s.flush()
        recs[0].burst_id = bg.id
        recs[0].is_best_shot = True
        recs[0].is_alternate = False
        recs[0].delete_approved = False

        recs[1].burst_id = bg.id
        recs[1].is_best_shot = False
        recs[1].is_alternate = True
        recs[1].delete_approved = True  # marked for delete

        recs[2].burst_id = bg.id
        recs[2].is_best_shot = False
        recs[2].is_alternate = True
        recs[2].delete_approved = False  # kept alternate

        rec_ids = [r.id for r in recs]
        burst_id = bg.id
        s.commit()

    return types.SimpleNamespace(
        app=app, client=client, db=app.state.db, lib_root=lib_root,
        burst_id=burst_id, rec_ids=rec_ids,
    )


def test_finalize_burst_moves_marked_to_trash_and_kept_to_library(burst_env):
    res = finalize_burst(burst_env.db, burst_env.app.state.cfg, burst_env.burst_id)
    assert res["ok"] is True
    assert res["trashed"] == 1
    assert res["finalized"] == 2
    assert res["transaction_id"] != ""

    with get_session(burst_env.db) as s:
        r0 = s.get(M.ImageRecord, burst_env.rec_ids[0])
        r1 = s.get(M.ImageRecord, burst_env.rec_ids[1])
        r2 = s.get(M.ImageRecord, burst_env.rec_ids[2])

        # Best shot is in library
        assert r0.stage == M.STAGE_LIBRARY
        assert r0.route_status == "executed"
        assert Path(r0.path).exists()
        assert burst_env.lib_root in Path(r0.path).parents

        # Marked for delete frame is in trash
        assert r1.stage == M.STAGE_TRASH
        assert r1.route_status == "trashed"
        assert r1.scan_status == "missing"
        assert Path(r1.path).exists()
        assert (burst_env.lib_root / ".trash") in Path(r1.path).parents

        # Kept alternate is in library
        assert r2.stage == M.STAGE_LIBRARY
        assert r2.route_status == "executed"
        assert Path(r2.path).exists()
        assert burst_env.lib_root in Path(r2.path).parents

        # Burst group still exists with 2 members
        bg = s.get(M.BurstGroup, burst_env.burst_id)
        assert bg is not None
        assert bg.image_count == 2


def test_finalize_burst_prunes_burst_group_when_only_one_kept(burst_env):
    # Mark both alternates for delete
    with get_session(burst_env.db) as s:
        r2 = s.get(M.ImageRecord, burst_env.rec_ids[2])
        r2.delete_approved = True
        s.commit()

    res = finalize_burst(burst_env.db, burst_env.app.state.cfg, burst_env.burst_id)
    assert res["ok"] is True
    assert res["trashed"] == 2
    assert res["finalized"] == 1
    assert res["pruned"] is True

    with get_session(burst_env.db) as s:
        # Best shot is standalone in library now
        r0 = s.get(M.ImageRecord, burst_env.rec_ids[0])
        assert r0.stage == M.STAGE_LIBRARY
        assert r0.burst_id is None
        assert r0.is_best_shot is False

        # Burst group was pruned
        bg = s.get(M.BurstGroup, burst_env.burst_id)
        assert bg is None


def test_finalize_burst_endpoint(burst_env):
    r = burst_env.client.post(f"/api/bursts/{burst_env.burst_id}/finalize")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["trashed"] == 1
    assert data["finalized"] == 2


def test_overview_burst_count_matches_stage(burst_env):
    # Before finalize: 1 burst in new, 0 in library
    ov_new = burst_env.client.get("/api/overview?stage=new").json()
    ov_lib = burst_env.client.get("/api/overview?stage=library").json()
    assert ov_new["stats"]["burst_groups"] == 1
    assert ov_lib["stats"]["burst_groups"] == 0

    # Finalize burst (2 kept members stay as burst in library)
    r = burst_env.client.post(f"/api/bursts/{burst_env.burst_id}/finalize")
    assert r.status_code == 200

    # After finalize: 0 bursts in new, 1 in library
    ov_new = burst_env.client.get("/api/overview?stage=new").json()
    ov_lib = burst_env.client.get("/api/overview?stage=library").json()
    assert ov_new["stats"]["burst_groups"] == 0
    assert ov_lib["stats"]["burst_groups"] == 1


def test_finalize_burst_can_be_undone(burst_env):
    res = finalize_burst(burst_env.db, burst_env.app.state.cfg, burst_env.burst_id)
    txn = res["transaction_id"]
    assert txn != ""

    planner = Planner(burst_env.app.state.cfg, db_path=burst_env.db)
    rollback_rep = planner.undo(txn)
    assert rollback_rep.reversed == 3

    with get_session(burst_env.db) as s:
        r0 = s.get(M.ImageRecord, burst_env.rec_ids[0])
        r1 = s.get(M.ImageRecord, burst_env.rec_ids[1])
        r2 = s.get(M.ImageRecord, burst_env.rec_ids[2])

        # Stages restored to new
        assert r0.stage == M.STAGE_NEW
        assert r1.stage == M.STAGE_NEW
        assert r2.stage == M.STAGE_NEW


def test_ui_has_finalize_burst_button():
    static_dir = Path(__file__).resolve().parent.parent / "photocure" / "ui" / "static"
    html = (static_dir / "index.html").read_text()
    js = (static_dir / "app.js").read_text()

    assert 'id="bulk-finalize-burst"' in html
    assert 'id="btn-finalize-all-bursts"' in html
    assert 'id="bursts-toolbar"' in html
    assert 'btn-finalize-burst' in js
    assert 'btn-finalize-all-pill' not in js
    assert 'Finalize burst' in js
    assert 'Finalize all' in js
    assert 'finalizeAllBursts' in js
    assert 'api/bursts/' in js and '/finalize' in js
    assert 'api/bursts/finalize-all' in js
    assert 'selectAllInSection' in js
    for sec in ("categories", "search", "people", "utility", "junk", "uncertain", "highlights", "events"):
        assert f'id="btn-select-all-{sec}"' in html
    assert 'id="btn-select-all-bursts"' not in html


@pytest.fixture()
def multi_burst_env(tmp_path):
    app = create_app(str(tmp_path / "ui_multi.db"))
    app.state.cfg.cache_dir = tmp_path / "cache"
    lib_root = tmp_path / "library"
    lib_root.mkdir()
    app.state.cfg.library_root = lib_root
    from photocure.services import save_settings
    save_settings(app.state.db, {"library_root": str(lib_root)})

    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    for i in range(1, 7):
        make_image(photos_dir / f"frame_{i}.jpg", rng_seed=i)

    assert client.post("/api/scan", json={"path": str(photos_dir)}).status_code == 200
    assert client.post("/api/infer", json={"device": "cpu"}).status_code == 200

    # Seed 2 burst groups with 3 members each
    with get_session(app.state.db) as s:
        recs = s.exec(select(M.ImageRecord).order_by(M.ImageRecord.id)).all()
        # Burst A: recs[0..2]
        bg1 = M.BurstGroup(name="Burst_A", image_count=3, best_shot_id=recs[0].id)
        s.add(bg1)
        s.flush()
        recs[0].burst_id = bg1.id
        recs[0].is_best_shot = True
        recs[0].is_alternate = False
        recs[0].delete_approved = False

        recs[1].burst_id = bg1.id
        recs[1].is_best_shot = False
        recs[1].is_alternate = True
        recs[1].delete_approved = True  # marked for delete

        recs[2].burst_id = bg1.id
        recs[2].is_best_shot = False
        recs[2].is_alternate = True
        recs[2].delete_approved = False  # kept alternate

        # Burst B: recs[3..5]
        bg2 = M.BurstGroup(name="Burst_B", image_count=3, best_shot_id=recs[3].id)
        s.add(bg2)
        s.flush()
        recs[3].burst_id = bg2.id
        recs[3].is_best_shot = True
        recs[3].is_alternate = False
        recs[3].delete_approved = False

        recs[4].burst_id = bg2.id
        recs[4].is_best_shot = False
        recs[4].is_alternate = True
        recs[4].delete_approved = True  # marked for delete

        recs[5].burst_id = bg2.id
        recs[5].is_best_shot = False
        recs[5].is_alternate = True
        recs[5].delete_approved = True  # marked for delete

        rec_ids = [r.id for r in recs]
        burst_ids = [bg1.id, bg2.id]
        s.commit()

    return types.SimpleNamespace(
        app=app, client=client, db=app.state.db, lib_root=lib_root,
        burst_ids=burst_ids, rec_ids=rec_ids,
    )


def test_finalize_all_bursts_moves_marked_to_trash_and_kept_to_library(multi_burst_env):
    res = finalize_all_bursts(multi_burst_env.db, multi_burst_env.app.state.cfg)
    assert res["ok"] is True
    assert res["bursts"] == 2
    assert res["trashed"] == 3  # rec 1, 4, 5
    assert res["finalized"] == 3  # rec 0, 2, 3

    with get_session(multi_burst_env.db) as s:
        # Check marked records moved to trash
        for idx in [1, 4, 5]:
            rec = s.get(M.ImageRecord, multi_burst_env.rec_ids[idx])
            assert rec.stage == M.STAGE_TRASH
            assert rec.route_status == "trashed"
            assert rec.scan_status == "missing"
            assert (multi_burst_env.lib_root / ".trash") in Path(rec.path).parents

        # Check kept records in library
        for idx in [0, 2, 3]:
            rec = s.get(M.ImageRecord, multi_burst_env.rec_ids[idx])
            assert rec.stage == M.STAGE_LIBRARY
            assert rec.route_status == "executed"
            assert multi_burst_env.lib_root in Path(rec.path).parents

        # Burst A had 2 kept members (0 and 2), so it still exists
        bg1 = s.get(M.BurstGroup, multi_burst_env.burst_ids[0])
        assert bg1 is not None
        assert bg1.image_count == 2

        # Burst B had only 1 kept member (3), so it was pruned
        bg2 = s.get(M.BurstGroup, multi_burst_env.burst_ids[1])
        assert bg2 is None
        r3 = s.get(M.ImageRecord, multi_burst_env.rec_ids[3])
        assert r3.burst_id is None
        assert r3.is_best_shot is False


def test_finalize_all_bursts_endpoint(multi_burst_env):
    r = multi_burst_env.client.post("/api/bursts/finalize-all")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["bursts"] == 2
    assert data["trashed"] == 3
    assert data["finalized"] == 3


def test_finalize_all_bursts_can_be_undone(multi_burst_env):
    res = finalize_all_bursts(multi_burst_env.db, multi_burst_env.app.state.cfg)
    txn = res["transaction_id"]
    assert txn != ""

    planner = Planner(multi_burst_env.app.state.cfg, db_path=multi_burst_env.db)
    rollback_rep = planner.undo(txn)
    assert rollback_rep.reversed == 6

    with get_session(multi_burst_env.db) as s:
        for rid in multi_burst_env.rec_ids:
            rec = s.get(M.ImageRecord, rid)
            assert rec.stage == M.STAGE_NEW


def test_cluster_stage_isolation(burst_env, tmp_path):
    import numpy as np
    from photocure.db.repository import quantize_embedding

    rng = np.random.default_rng(42)
    vec = [float(x) for x in (rng.normal(size=512) / np.linalg.norm(rng.normal(size=512)))]
    emb = quantize_embedding(vec)

    # Give the 2 kept photos in library exif timestamps and identical embeddings
    with get_session(burst_env.db) as s:
        recs = s.exec(select(M.ImageRecord).order_by(M.ImageRecord.id)).all()
        for i, r in enumerate(recs):
            r.taken_at = datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc)
            r.taken_at_source = "exif:dtoriginal"
            r.phash = "0000000000000000"
            if not r.semantic:
                s.add(M.SemanticScore(image_id=r.id, embedding=emb))
            else:
                r.semantic.embedding = emb
        s.commit()

    # 1. Finalize the initial burst so it is in library
    r = burst_env.client.post(f"/api/bursts/{burst_env.burst_id}/finalize")
    assert r.status_code == 200
    ov_lib = burst_env.client.get("/api/overview?stage=library").json()
    assert ov_lib["stats"]["burst_groups"] == 1

    # 2. Add two new photos to new stage and form a new burst in new
    new_dir = tmp_path / "new_photos"
    new_dir.mkdir()
    for i in (10, 11):
        make_image(new_dir / f"new_{i}.jpg", rng_seed=i)
    assert burst_env.client.post("/api/scan", json={"path": str(new_dir)}).status_code == 200
    assert burst_env.client.post("/api/infer", json={"device": "cpu"}).status_code == 200

    with get_session(burst_env.db) as s:
        new_recs = s.exec(select(M.ImageRecord).where(M.ImageRecord.stage == "new").order_by(M.ImageRecord.id)).all()
        assert len(new_recs) == 2
        for i, nr in enumerate(new_recs):
            nr.taken_at = datetime(2024, 2, 2, 10, 0, i, tzinfo=timezone.utc)
            nr.taken_at_source = "exif:dtoriginal"
            nr.phash = "0000000000000000"
            if not nr.semantic:
                s.add(M.SemanticScore(image_id=nr.id, embedding=emb))
            else:
                nr.semantic.embedding = emb
        bg_new = M.BurstGroup(name="Burst_20240202_0001", image_count=2, best_shot_id=new_recs[0].id)
        s.add(bg_new)
        s.flush()
        new_recs[0].burst_id = bg_new.id
        new_recs[0].is_best_shot = True
        new_recs[1].burst_id = bg_new.id
        new_recs[1].is_alternate = True
        s.commit()
        new_bid = bg_new.id

    # Verify both stages have 1 burst group
    assert burst_env.client.get("/api/overview?stage=new").json()["stats"]["burst_groups"] == 1
    assert burst_env.client.get("/api/overview?stage=library").json()["stats"]["burst_groups"] == 1

    # 3. Cluster on stage=library only
    res = burst_env.client.post("/api/cluster", json={"stage": "library"}).json()
    assert res["ok"] is True

    # The burst in stage=new must still be completely intact!
    with get_session(burst_env.db) as s:
        new_burst = s.get(M.BurstGroup, new_bid)
        assert new_burst is not None
        recs_in_new = s.exec(select(M.ImageRecord).where(M.ImageRecord.burst_id == new_bid)).all()
        assert len(recs_in_new) == 2
        for r in recs_in_new:
            assert r.stage == "new"

    assert burst_env.client.get("/api/overview?stage=new").json()["stats"]["burst_groups"] == 1
    assert burst_env.client.get("/api/overview?stage=library").json()["stats"]["burst_groups"] == 1

    # 4. Cluster on stage=new only
    res = burst_env.client.post("/api/cluster", json={"stage": "new"}).json()
    assert res["ok"] is True

    # The burst in stage=library must still be intact!
    assert burst_env.client.get("/api/overview?stage=library").json()["stats"]["burst_groups"] == 1
    assert burst_env.client.get("/api/overview?stage=new").json()["stats"]["burst_groups"] == 1


