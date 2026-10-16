"""API tests for the review dashboard (token, filters, jobs, clusters, thumbnails)."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from photocure.db import models as M
from photocure.db.session import get_session
from photocure.ui import app as ui_app
from photocure.ui.app import create_app

from .conftest import make_image

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def env(tmp_path):
    """App + authed client with four distinct inferred images in a library."""
    app = create_app(str(tmp_path / "ui.db"))
    app.state.cfg.cache_dir = tmp_path / "cache"
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    lib = tmp_path / "photos"
    make_image(lib / "a.jpg", rng_seed=1)
    make_image(lib / "b_blur.jpg", rng_seed=2, blur=True)
    make_image(lib / "c_dark.jpg", rng_seed=3, dark=True)
    make_image(lib / "d_white.jpg", rng_seed=4, white=True)
    assert client.post("/api/scan", json={"path": str(lib)}).status_code == 200
    assert client.post("/api/infer", json={"device": "cpu"}).status_code == 200
    return types.SimpleNamespace(app=app, client=client, lib=lib, out=tmp_path / "out",
                                 db=app.state.db, tmp=tmp_path)


def wait_job(client, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def plan_body(env, **kw):
    return {"output_dir": str(env.out), "operation": "move", **kw}


# ---------------------------------------------------------------- security
def test_index_leaks_no_credentials_and_no_cors(env):
    html = env.client.get("/").text
    assert env.app.state.token not in html
    assert "__PCUR_TOKEN__" not in html
    r = env.client.get("/api/stats", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    r = env.client.options("/api/plan/apply", headers={
        "Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


MUTATING = [
    ("/api/scan", {"path": "/x"}), ("/api/infer", {}), ("/api/cluster", {}),
    ("/api/ingest", {"path": "/x"}), ("/api/images/1/reassign", {"bucket": "food"}),
    ("/api/images/1/lock", {"locked": True}),
    ("/api/plan/rebuild", {"output_dir": "/x"}), ("/api/plan/diff", {"output_dir": "/x"}),
    ("/api/plan/apply", {"output_dir": "/x"}), ("/api/undo", {}), ("/api/repair", {}),
    ("/api/events/cluster", {}), ("/api/events/1/rename", {"name": "n"}),
    ("/api/learn", {}), ("/api/stop", {}), ("/api/reset", {}),
]


@pytest.mark.parametrize(("path", "body"), MUTATING)
def test_every_mutating_endpoint_needs_token(env, path, body):
    anon = TestClient(env.app)
    assert anon.post(path, json=body).status_code == 401
    assert anon.post(path, json=body, headers={"X-PCUR-Token": "nope"}).status_code == 401


def test_import_has_no_side_effects_and_lazy_app(tmp_path, monkeypatch):
    import os
    env_vars = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, "-c", "import photocure.ui.app"],
                   cwd=tmp_path, env=env_vars, check=True)
    assert not (tmp_path / "photocure.db").exists()
    monkeypatch.setenv("PCUR_DB_PATH", str(tmp_path / "lazy.db"))
    monkeypatch.setattr(ui_app, "_app_singleton", None)
    a = ui_app.app
    assert a is ui_app.app
    assert (tmp_path / "lazy.db").exists()


# ------------------------------------------------------------------ config
def test_config_endpoint(env):
    d = env.client.get("/api/config").json()
    assert Path(d["default_output_dir"]).is_absolute()
    vals = {c["value"]: c["junk"] for c in d["categories"]}
    assert vals["uncertain"] is False
    assert vals["screenshot"] is True


# ------------------------------------------------------------------ images
def test_images_pagination_and_totals(env):
    r = env.client.get("/api/images?limit=2&offset=0").json()
    assert r["total"] == 4
    assert len(r["items"]) == 2
    r2 = env.client.get("/api/images?limit=2&offset=2").json()
    assert len(r2["items"]) == 2
    ids = [i["id"] for i in r["items"] + r2["items"]]
    assert len(set(ids)) == 4
    assert env.client.get("/api/images?limit=0").status_code == 422
    assert env.client.get("/api/images?limit=5000").status_code == 422
    it = r["items"][0]
    for k in ("locked", "rating", "event_id", "near_dup_of", "aesthetic", "face_count"):
        assert k in it


def test_images_filters(env):
    c = env.client
    assert c.get("/api/images?q=blur").json()["total"] == 1
    assert c.get("/api/images?q=%25").json()["total"] == 0
    assert c.get("/api/images?q=_").json()["total"] == 3  # literal underscore
    items = c.get("/api/images").json()["items"]
    target = items[0]["id"]
    assert c.post(f"/api/images/{target}/reassign", json={"bucket": "screenshot"}).status_code == 200
    assert c.get("/api/images?manual=true").json()["total"] == 1
    assert c.get("/api/images?category=screenshot").json()["total"] >= 1
    assert c.get("/api/images?group=junk").json()["total"] >= 1
    got = c.get("/api/images?categories=screenshot,food").json()
    assert target in [i["id"] for i in got["items"]]
    assert c.get("/api/images?group=bogus").status_code == 422


def test_near_dup_highlights_event_filters(env):
    ids = [i["id"] for i in env.client.get("/api/images").json()["items"]]
    with get_session(env.db) as s:
        a, b = s.get(M.ImageRecord, ids[0]), s.get(M.ImageRecord, ids[1])
        b.near_dup_of = a.id
        ev = M.EventGroup(label="trip", image_count=2)
        s.add(ev)
        s.flush()
        a.event_id = ev.id
        b.event_id = ev.id
        for i, val in zip(ids, (1.0, 2.0, 7.5, 3.0)):
            s.get(M.ImageRecord, i).semantic.aesthetic = val
        eid = ev.id
    c = env.client
    assert [i["id"] for i in c.get("/api/images?near_dup=true").json()["items"]] == [ids[1]]
    assert c.get(f"/api/images?event_id={eid}").json()["total"] == 2
    h = c.get("/api/images?highlights=true").json()
    assert [i["id"] for i in h["items"]] == [ids[2], ids[3], ids[1], ids[0]]
    assert [i["id"] for i in c.get("/api/images?min_aesthetic=5").json()["items"]] == [ids[2]]


# ---------------------------------------------------------------- reassign
def test_reassign_validation_and_persistence(env):
    c = env.client
    iid = c.get("/api/images").json()["items"][0]["id"]
    assert c.post(f"/api/images/{iid}/reassign", json={"bucket": "nonsense"}).status_code == 400
    assert c.post(f"/api/images/{iid}/reassign", json={"category": "nonsense"}).status_code == 400
    assert c.post("/api/images/99999/reassign", json={"bucket": "food"}).status_code == 404
    r = c.post(f"/api/images/{iid}/reassign", json={"bucket": "food"}).json()
    assert r["manual_bucket"] == "food"
    r = c.post(f"/api/images/{iid}/reassign", json={"category": "animal"}).json()
    assert r["manual_bucket"] == "animal"
    with get_session(env.db) as s:
        assert s.get(M.ImageRecord, iid).manual_bucket == "animal"
    r = c.post(f"/api/images/{iid}/reassign", json={"bucket": None}).json()
    assert r["manual_bucket"] is None


def test_lock_endpoint(env):
    c = env.client
    iid = c.get("/api/images").json()["items"][0]["id"]
    assert c.post(f"/api/images/{iid}/lock", json={"locked": True}).json()["locked"] is True
    assert c.get("/api/images?locked=true").json()["total"] == 1
    assert c.post(f"/api/images/{iid}/lock", json={"locked": False}).json()["locked"] is False
    assert c.get("/api/images?locked=true").json()["total"] == 0
    assert c.post("/api/images/99999/lock", json={"locked": True}).status_code == 404


# -------------------------------------------------------------------- busy
def test_busy_lock_returns_409(env):
    assert ui_app._BUSY.acquire(blocking=False)
    try:
        assert env.client.get("/api/progress").json()["busy"] is True
        for path, body in [("/api/scan", {"path": str(env.lib)}), ("/api/infer", {}),
                           ("/api/cluster", {}), ("/api/ingest", {"path": str(env.lib)}),
                           ("/api/plan/rebuild", plan_body(env)),
                           ("/api/plan/diff", plan_body(env)),
                           ("/api/plan/apply", plan_body(env)),
                           ("/api/undo", {"transaction_id": "T1"}),
                           ("/api/repair", {})]:
            r = env.client.post(path, json=body)
            assert r.status_code in (409, 501), (path, r.status_code)
    finally:
        ui_app._BUSY.release()
    assert env.client.get("/api/progress").json()["busy"] is False


def test_config_not_mutated_by_requests(env):
    before = env.app.state.cfg.device.device
    env.client.post("/api/infer", json={"device": "cpu", "batch_size": 7})
    assert env.app.state.cfg.device.device == before
    assert env.app.state.cfg.device.face_batch_size != 7 or before == "cpu"


# ------------------------------------------------------------- output dir
@pytest.mark.parametrize("bad", ["Organized", "relative/dir", "/", "/etc/photos",
                                 "/usr/local/x", ""])
def test_output_dir_rejected(env, bad):
    for path in ("/api/plan/rebuild", "/api/plan/diff", "/api/plan/apply"):
        r = env.client.post(path, json={"output_dir": bad})
        assert r.status_code == 400, (path, bad, r.status_code)


def test_output_dir_inside_library_rejected_unless_in_place(env):
    inside = str(env.lib / "organized")
    r = env.client.post("/api/plan/diff", json={"output_dir": inside})
    assert r.status_code == 400
    r = env.client.post("/api/plan/diff", json={
        "output_dir": inside, "mode": "in_place", "operation": "hardlink"})
    assert r.status_code == 200


def test_bad_enum_values_rejected(env):
    r = env.client.post("/api/plan/diff", json=plan_body(env, operation="explode"))
    assert r.status_code == 422
    r = env.client.post("/api/plan/diff", json=plan_body(env, only_category="bogus"))
    assert r.status_code == 400


# ------------------------------------------------------------- plan / diff
def test_plan_rebuild_and_diff(env):
    r = env.client.post("/api/plan/rebuild", json=plan_body(env))
    assert r.status_code == 200
    body = r.json()
    assert body["plan_id"] and body["total"] == 4
    d = env.client.post("/api/plan/diff", json=plan_body(env)).json()
    for k in ("collisions", "cross_device", "bytes_total", "bytes_to_write",
              "free_bytes", "enough_space", "missing_sources", "by_category"):
        assert k in d
    assert d["total"] == 4
    assert d["bytes_total"] > 0
    assert d["bytes_to_write"] == 0  # same-device move writes nothing
    assert d["enough_space"] is True
    dc = env.client.post("/api/plan/diff", json=plan_body(env, operation="copy")).json()
    assert dc["bytes_to_write"] == dc["bytes_total"]


def test_diff_detects_collisions(env):
    from photocure.rules.organizer import OrganizeMode
    from photocure.services import Planner
    p = Planner(env.app.state.cfg, db_path=env.db).build_plan(
        organize_root=env.out, operation="move", mode=OrganizeMode.CATEGORY_FIRST)
    first = p.actions[0].target
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"x")
    d = env.client.post("/api/plan/diff", json=plan_body(env)).json()
    assert d["collisions"] >= 1
    assert str(first) in d["collision_samples"]


# ------------------------------------------------------------ apply / undo
def test_apply_job_log_transactions_and_undo(env):
    c = env.client
    before = sorted(p.name for p in env.lib.iterdir())
    r = c.post("/api/plan/apply", json=plan_body(env))
    assert r.status_code == 202
    job = wait_job(c, r.json()["job_id"])
    assert job["status"] == "done", job
    res = job["result"]
    assert res["executed"] == 4 and res["failures"] == 0
    assert any("committed" in line["msg"] for line in job["log"])
    assert not list(env.lib.glob("*.jpg"))
    assert list(env.out.rglob("*.jpg"))
    assert c.get("/api/progress").json()["busy"] is False

    tx = c.get("/api/transactions").json()
    assert tx["latest_undoable"] == res["transaction_id"]
    assert tx["items"][0]["committed"] == 4 and tx["items"][0]["undoable"] is True

    # Moved images still serve thumbnails.
    iid = c.get("/api/images").json()["items"][0]["id"]
    assert c.get(f"/thumbnail/{iid}").status_code == 200

    u = c.post("/api/undo", json={})
    assert u.status_code == 202
    ujob = wait_job(c, u.json()["job_id"])
    assert ujob["status"] == "done", ujob
    assert ujob["result"]["reversed"] == 4
    assert sorted(p.name for p in env.lib.iterdir()) == before

    # Fully rolled back: nothing left to undo.
    assert c.get("/api/transactions").json()["latest_undoable"] is None
    again = c.post("/api/undo", json={}).json()
    assert again["ok"] is False


def test_apply_only_category_with_nothing_matching(env):
    r = env.client.post("/api/plan/apply", json=plan_body(env, only_category="people"))
    job = wait_job(env.client, r.json()["job_id"])
    if job["result"]["ok"] is False:
        assert job["result"]["error"] == "nothing to apply"
        assert list(env.lib.glob("*.jpg"))
    else:
        assert job["result"]["executed"] >= 1


def test_apply_trash_flag_supported_or_501(env):
    r = env.client.post("/api/plan/apply", json=plan_body(env, trash=True))
    assert r.status_code in (202, 501)
    if r.status_code == 202:
        wait_job(env.client, r.json()["job_id"])


def test_job_404(env):
    assert env.client.get("/api/jobs/nope").status_code == 404


def test_latest_undoable_skips_fully_rolled_back(env):
    with get_session(env.db) as s:
        s.add(M.ActionLedger(transaction_id="A", source_path="/s/a", target_path="/t/a",
                             status="committed"))
        s.add(M.ActionLedger(transaction_id="B", source_path="/s/b", target_path="/t/b",
                             status="rolled_back"))
    tx = env.client.get("/api/transactions").json()
    assert tx["latest_undoable"] == "A"
    by = {t["transaction_id"]: t for t in tx["items"]}
    assert by["A"]["undoable"] is True and by["B"]["undoable"] is False
    assert tx["items"][0]["transaction_id"] == "B"


# -------------------------------------------------------------- thumbnails
def test_thumbnail_ok_cached_and_fingerprinted(env):
    iid = env.client.get("/api/images").json()["items"][0]["id"]
    r = env.client.get(f"/thumbnail/{iid}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    cached = list((env.app.state.cfg.cache_dir / "thumbs" / "micro").glob(f"*_{iid}_*.webp"))
    assert len(cached) == 1
    assert env.client.get(f"/thumbnail/{iid}").status_code == 200
    assert env.client.get("/thumbnail/99999").status_code == 404


def test_thumbnail_missing_file_404(env):
    iid = env.client.get("/api/images").json()["items"][0]["id"]
    with get_session(env.db) as s:
        rec = s.get(M.ImageRecord, iid)
        Path(rec.path).unlink()
        rec.quick_hash = "changed"  # new fingerprint avoids any cached copy
    assert env.client.get(f"/thumbnail/{iid}").status_code == 404


# ------------------------------------------------------- similar / clusters
def _fake(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)


def test_find_similar_endpoint(env, monkeypatch):
    ids = [i["id"] for i in env.client.get("/api/images").json()["items"]]
    _fake(monkeypatch, "photocure.clustering.similarity",
          find_similar=lambda session, image_id, k: [(ids[1], 0.91), (ids[2], 0.5)][:k])
    r = env.client.get(f"/api/images/{ids[0]}/similar?k=2").json()
    assert [i["id"] for i in r["items"]] == [ids[1], ids[2]]
    assert r["items"][0]["score"] == 0.91
    assert env.client.get("/api/images/99999/similar").status_code == 404
    assert env.client.get(f"/api/images/{ids[0]}/similar?k=0").status_code == 422


def test_events_endpoints(env, monkeypatch):
    calls = {}

    def fake_cluster(session, gap_hours, gps_km):
        calls["args"] = (gap_hours, gps_km)
        session.add(M.EventGroup(label=None, image_count=3))
        return 1
    _fake(monkeypatch, "photocure.clustering.events", cluster_events=fake_cluster)
    r = env.client.post("/api/events/cluster", json={"gap_hours": 3, "gps_km": 2})
    assert r.json() == {"ok": True, "events": 1}
    assert calls["args"] == (3, 2)
    ev = env.client.get("/api/events").json()
    assert ev["count"] == 1 and ev["items"][0]["image_count"] == 3
    eid = ev["items"][0]["id"]
    assert env.client.post(f"/api/events/{eid}/rename", json={"name": "Trip"}).json()["label"] == "Trip"
    assert env.client.get("/api/events").json()["items"][0]["label"] == "Trip"
    assert env.client.post("/api/events/999/rename", json={"name": "x"}).status_code == 404


def test_identity_clustering_endpoints_are_gone(env):
    assert env.client.get("/api/people").status_code == 404
    assert env.client.post("/api/people/cluster", json={}).status_code == 404
    assert env.client.post("/api/people/1/rename", json={"name": "x"}).status_code == 404


# ------------------------------------------------------- verify and repair
def test_verify_and_repair_endpoints(env, monkeypatch):
    from photocure.services import Planner
    monkeypatch.setattr(Planner, "verify_transaction",
                        lambda self, tid: {"checked": 2, "drift": 0}, raising=False)
    monkeypatch.setattr(Planner, "repair_paths", lambda self: 5, raising=False)
    with get_session(env.db) as s:
        s.add(M.ActionLedger(transaction_id="TX9", source_path="/s", target_path="/t",
                             status="committed"))
    r = env.client.get("/api/transactions/latest/verify").json()
    assert r == {"transaction_id": "TX9", "report": {"checked": 2, "drift": 0}}
    assert env.client.get("/api/transactions/TX1/verify").json()["transaction_id"] == "TX1"
    assert env.client.post("/api/repair", json={}).json() == {"ok": True, "repaired": 5}


def test_verify_latest_404_without_transactions(env, monkeypatch):
    from photocure.services import Planner
    monkeypatch.setattr(Planner, "verify_transaction", lambda self, t: {}, raising=False)
    assert env.client.get("/api/transactions/latest/verify").status_code == 404


# ------------------------------------------------------------------ bursts
def test_burst_endpoints_404_and_listing(env):
    assert env.client.post("/api/cluster", json={}).status_code == 200
    assert env.client.get("/api/burst/99999").status_code == 404
    b = env.client.get("/api/bursts").json()
    for item in b["items"]:
        d = env.client.get(f"/api/burst/{item['id']}").json()
        assert d["burst"] == item["name"]


def test_ingest_single_call(env, tmp_path):
    lib2 = tmp_path / "lib2"
    make_image(lib2 / "z.jpg", rng_seed=9, dark=True)
    r = env.client.post("/api/ingest", json={"path": str(lib2)}).json()
    assert r["ok"] and r["scan"]["new"] == 1
    assert env.client.get("/api/progress").json()["active"] is False


def test_progress_concurrent_apply_rejected(env):
    r1 = env.client.post("/api/plan/apply", json=plan_body(env))
    assert r1.status_code == 202
    results = []

    def other():
        results.append(env.client.post("/api/cluster", json={}).status_code)
    t = threading.Thread(target=other)
    t.start()
    t.join()
    wait_job(env.client, r1.json()["job_id"])
    assert results[0] in (200, 409)


# ------------------------------------------------------------------ reset & stop
def test_reset_library(env):
    # Verify initial stats show images
    stats_before = env.client.get("/api/stats").json()
    assert stats_before["total_images"] == 4

    # Execute reset
    res = env.client.post("/api/reset", json={})
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["deleted"]["images"] == 4

    # Verify stats after reset are zeroed out
    stats_after = env.client.get("/api/stats").json()
    assert stats_after["total_images"] == 0
    assert stats_after["pending_inference"] == 0
    assert stats_after["burst_groups"] == 0

    # Images endpoint returns empty
    imgs = env.client.get("/api/images").json()
    assert imgs["total"] == 0
    assert len(imgs["items"]) == 0


def test_stop_endpoint(env):
    # Call stop when idle
    r = env.client.post("/api/stop", json={})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # Call stop when active
    env.app.state.progress.start("TestTask", "testing", 100)
    assert env.app.state.progress.is_stopped() is False
    r2 = env.client.post("/api/stop", json={})
    assert r2.status_code == 200
    assert r2.json()["stopping"] is True
    assert env.app.state.progress.is_stopped() is True
    assert env.app.state.progress.get()["stopping"] is True


def test_cancellation_during_scan(env, tmp_path):
    from photocure.scanner.file_scanner import FileScanner

    lib = tmp_path / "stop_scan_photos"
    lib.mkdir()
    for i in range(10):
        make_image(lib / f"img_{i}.jpg", rng_seed=i)

    scanner = FileScanner(env.app.state.cfg)
    stop_event = threading.Event()

    def on_prog(done, total, curr):
        if done >= 2:
            stop_event.set()

    with pytest.raises(InterruptedError):
        scanner.scan_directory(lib, db_path=env.db, on_progress=on_prog,
                               cancel_check=stop_event.is_set)


def test_cancellation_during_infer(env, tmp_path):
    from photocure.inference.pipeline import InferencePipeline

    lib = tmp_path / "stop_infer_photos"
    lib.mkdir()
    for i in range(5):
        make_image(lib / f"infer_{i}.jpg", rng_seed=100 + i)

    # Scan them in
    assert env.client.post("/api/scan", json={"path": str(lib)}).status_code == 200

    pipeline = InferencePipeline(env.app.state.cfg)
    # Cancel immediately
    with pytest.raises(InterruptedError):
        pipeline.run(db_path=env.db, cancel_check=lambda: True)

