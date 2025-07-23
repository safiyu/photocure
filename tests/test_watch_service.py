"""Watch mode inside the app: new photos are scanned, analysed and filed for review."""
from __future__ import annotations

import threading
import time
import types

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from photocure import features
from photocure.db.models import ImageRecord
from photocure.db.session import get_session, init_db
from photocure.services import Planner
from photocure.ui.app import create_app
from photocure.watch_service import WatchService, default_process

from .conftest import make_image


def _wait(pred, timeout=25.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _rows(db):
    with get_session(db) as s:
        return [(r.filename, r.infer_status, r.category) for r in s.exec(select(ImageRecord)).all()]


def _filed(db, n):
    rows = _rows(db)
    return len(rows) == n and all(st == "inferred" and cat is not None for _, st, cat in rows)


@pytest.fixture()
def svc(cfg, tmp_db):
    init_db(tmp_db)
    made = []

    def make(busy=None, process=None, on_cancel=None, stop_timeout=8.0):
        s = WatchService(tmp_db, lambda: cfg, busy=busy or threading.Lock(),
                         process_fn=process or (lambda c: default_process(tmp_db, c)),
                         debounce_s=0.1, poll_interval=0.1, use_watchdog=False, retry_s=0.1,
                         on_cancel=on_cancel, stop_timeout=stop_timeout)
        made.append(s)
        return s

    yield make
    for s in made:
        s.stop()


def test_a_missing_folder_is_refused(svc, tmp_path):
    with pytest.raises(FileNotFoundError):
        svc().start(tmp_path / "nope")


def test_new_photos_are_scanned_analysed_and_filed(svc, tmp_db, tmp_path):
    root = tmp_path / "inbox"
    make_image(root / "a.jpg", rng_seed=1)
    s = svc()
    s.start(root)
    assert s.status()["running"] and s.status()["path"] == str(root.resolve())
    assert _wait(lambda: _filed(tmp_db, 1))
    make_image(root / "sub" / "b.jpg", rng_seed=2)
    assert _wait(lambda: _filed(tmp_db, 2))
    assert _wait(lambda: s.status()["state"] == "idle")
    st = s.status()
    assert st["photos_added"] == 2 and st["batches"] >= 2 and st["last_error"] is None
    assert st["last_event"] is not None


def test_stopping_ends_the_watch_and_ignores_later_files(svc, tmp_db, tmp_path):
    root = tmp_path / "inbox"
    make_image(root / "a.jpg", rng_seed=1)
    s = svc()
    s.start(root)
    assert _wait(lambda: _filed(tmp_db, 1))
    s.stop()
    assert s.status()["running"] is False and s.status()["state"] == "stopped"
    make_image(root / "late.jpg", rng_seed=3)
    time.sleep(0.8)
    assert len(_rows(tmp_db)) == 1


def test_scanning_and_analysis_both_wait_while_another_task_runs(svc, tmp_db, tmp_path):
    root = tmp_path / "inbox"
    root.mkdir()
    busy = threading.Lock()
    s = svc(busy=busy)
    busy.acquire()
    s.start(root)
    make_image(root / "a.jpg", rng_seed=1)
    assert _wait(lambda: s.status()["state"] == "waiting")
    time.sleep(0.6)
    assert _rows(tmp_db) == []                       # not even scanned while an apply or scan may be running
    busy.release()
    assert _wait(lambda: _filed(tmp_db, 1))


def test_stopping_cancels_a_running_analysis_and_frees_the_lock(svc, tmp_db, tmp_path):
    root = tmp_path / "inbox"
    make_image(root / "a.jpg", rng_seed=1)
    started, release = threading.Event(), threading.Event()

    def slow(c):
        started.set()
        release.wait(30)
        return {}

    busy = threading.Lock()
    s = svc(busy=busy, process=slow, on_cancel=release.set)
    s.start(root)
    assert started.wait(20)
    t0 = time.monotonic()
    assert s.stop() is True
    assert time.monotonic() - t0 < 5
    assert s.status()["running"] is False
    assert busy.acquire(blocking=False)              # the watch no longer holds the shared lock
    busy.release()


def test_a_new_watch_is_refused_while_the_old_analysis_cannot_be_stopped(svc, tmp_db, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_image(a / "a.jpg", rng_seed=1)
    b.mkdir()
    started, release = threading.Event(), threading.Event()

    def stuck(c):
        started.set()
        release.wait(30)
        return {}

    s = svc(process=stuck, stop_timeout=0.3)
    s.start(a)
    assert started.wait(20)
    assert s.stop() is False
    assert s.status()["running"] is True and s.status()["state"] == "stopping"
    with pytest.raises(RuntimeError):
        s.start(b)
    release.set()
    assert _wait(lambda: s.stop() is True)
    s.start(b)                                       # fine once the old one has finished
    assert s.status()["path"] == str(b.resolve())


def test_a_failed_batch_is_reported_and_the_next_one_recovers(svc, tmp_db, tmp_path):
    root = tmp_path / "inbox"
    root.mkdir()
    calls = []

    def flaky(c):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("model fell over")
        return default_process(tmp_db, c)

    s = svc(process=flaky)
    s.start(root)
    make_image(root / "a.jpg", rng_seed=1)
    assert _wait(lambda: s.status()["last_error"] is not None)
    assert "model fell over" in s.status()["last_error"] and s.status()["running"]
    make_image(root / "b.jpg", rng_seed=2)
    assert _wait(lambda: _filed(tmp_db, 2))
    assert s.status()["last_error"] is None


def test_starting_again_replaces_the_watch(svc, tmp_db, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    s = svc()
    s.start(a)
    s.start(b)
    assert s.status()["path"] == str(b.resolve())
    assert [t.name for t in threading.enumerate()].count("photocure-watch") == 1
    make_image(a / "ignored.jpg", rng_seed=1)
    make_image(b / "seen.jpg", rng_seed=2)
    assert _wait(lambda: _filed(tmp_db, 1))
    assert _rows(tmp_db)[0][0] == "seen.jpg"


# ----------------------------------------------------------------- routing
def test_route_new_files_only_photos_that_have_no_category_yet(cfg, tmp_db):
    init_db(tmp_db, drop=True)
    with get_session(tmp_db) as s:
        fresh = ImageRecord(path="/l/a.jpg", filename="a.jpg", infer_status="inferred")
        done = ImageRecord(path="/l/b.jpg", filename="b.jpg", infer_status="inferred", category="food")
        waiting = ImageRecord(path="/l/c.jpg", filename="c.jpg", infer_status="pending")
        hand = ImageRecord(path="/l/d.jpg", filename="d.jpg", infer_status="inferred", manual_bucket="people")
        dup = ImageRecord(path="/l/e.jpg", filename="e.jpg", infer_status="inferred", is_duplicate=True)
        gone = ImageRecord(path="/l/f.jpg", filename="f.jpg", infer_status="inferred", scan_status="missing")
        s.add_all([fresh, done, waiting, hand, dup, gone])
    assert Planner(cfg, db_path=tmp_db).route_new() == 2
    by = {n: c for n, _, c in _rows(tmp_db)}
    assert by["a.jpg"] is not None and by["e.jpg"] == "duplicate"
    assert by["b.jpg"] == "food" and by["c.jpg"] is None and by["d.jpg"] is None and by["f.jpg"] is None
    assert Planner(cfg, db_path=tmp_db).route_new() == 0


# --------------------------------------------------------------------- API
@pytest.fixture()
def env(tmp_path):
    app = create_app(str(tmp_path / "w.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    lib = tmp_path / "inbox"
    lib.mkdir()
    yield types.SimpleNamespace(app=app, client=client, lib=lib, tmp=tmp_path)
    app.state.watch.stop()


def test_api_status_when_nothing_is_watched(env):
    st = env.client.get("/api/watch").json()
    assert st["running"] is False and st["state"] == "stopped" and st["path"] == ""


def test_api_reports_a_watch_that_is_still_stopping(env):
    started, release = threading.Event(), threading.Event()
    make_image(env.lib / "a.jpg", rng_seed=1)

    def stuck(c):
        started.set()
        release.wait(30)
        return {}

    env.app.state.watch._process = stuck
    env.app.state.watch._stop_timeout = 0.3
    env.app.state.watch._on_cancel = None
    assert env.client.post("/api/watch", json={"enabled": True, "path": str(env.lib)}).status_code == 200
    assert started.wait(20)
    other = env.tmp / "other"
    other.mkdir()
    r = env.client.post("/api/watch", json={"enabled": True, "path": str(other)})
    assert r.status_code == 409
    release.set()


def test_api_start_persists_the_choice_and_stop_keeps_the_folder(env):
    r = env.client.post("/api/watch", json={"enabled": True, "path": str(env.lib)})
    assert r.status_code == 200 and r.json()["running"] is True
    f = features.load_features(env.app.state.db)
    assert f["watch_enabled"] is True and f["watch_path"] == str(env.lib.resolve())
    r = env.client.post("/api/watch", json={"enabled": False})
    assert r.json()["running"] is False
    f = features.load_features(env.app.state.db)
    assert f["watch_enabled"] is False and f["watch_path"] == str(env.lib.resolve())


def test_api_refuses_a_missing_folder_and_a_start_without_one(env):
    bad = env.client.post("/api/watch", json={"enabled": True, "path": str(env.tmp / "nope")})
    assert bad.status_code == 400
    assert env.client.post("/api/watch", json={"enabled": True}).status_code == 400
    assert features.load_features(env.app.state.db)["watch_enabled"] is False


def test_api_watch_needs_a_session(env):
    anon = TestClient(env.app)
    assert anon.get("/api/watch").status_code == 401
    assert anon.post("/api/watch", json={"enabled": False}).status_code == 401


def test_the_saved_watch_resumes_when_the_app_starts(tmp_path):
    db = tmp_path / "r.db"
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    features.save_features(db, {"watch_enabled": True, "watch_path": str(inbox)})
    app = create_app(str(db))
    with TestClient(app) as c:
        c.headers["X-PCUR-Token"] = app.state.token
        assert c.get("/api/watch").json()["running"] is True
    assert app.state.watch.status()["running"] is False


def test_a_saved_folder_that_has_gone_does_not_stop_the_app_starting(tmp_path):
    db = tmp_path / "r.db"
    features.save_features(db, {"watch_enabled": True, "watch_path": str(tmp_path / "gone")})
    app = create_app(str(db))
    with TestClient(app) as c:
        c.headers["X-PCUR-Token"] = app.state.token
        st = c.get("/api/watch").json()
        assert st["running"] is False and "gone" in (st["last_error"] or "")
