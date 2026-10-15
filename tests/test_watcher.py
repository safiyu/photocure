"""Watch mode: debounced ingestion via polling and (when installed) watchdog."""
from __future__ import annotations

import importlib.util
import threading
import time

import pytest

from photocure.scanner.file_scanner import FileScanner
from photocure.scanner.watcher import watch
from tests.conftest import make_image


def _count(db):
    from photocure.db.repository import PhotoRepository
    from photocure.db.session import get_session
    with get_session(db) as s:
        return len(PhotoRepository(s).all_images())


def _run(cfg, root, db, **kw):
    from photocure.db.session import init_db
    init_db(db)  # create the schema before threads race on create_all
    stop = threading.Event()
    out = {}
    t = threading.Thread(
        target=lambda: out.setdefault("r", watch(root, db, cfg=cfg, stop_event=stop, **kw)))
    t.start()
    return stop, t, out


def _wait(pred, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_poll_mode_ingests_new_files(cfg, tmp_db, tmp_path):
    root = tmp_path / "inbox"
    make_image(root / "first.jpg", rng_seed=1)
    stop, t, out = _run(cfg, root, tmp_db, debounce_s=0.2, poll_interval=0.2,
                        use_watchdog=False)
    try:
        assert _wait(lambda: _count(tmp_db) == 1)
        time.sleep(0.3)
        make_image(root / "sub" / "second.jpg", rng_seed=2)
        assert _wait(lambda: _count(tmp_db) == 2)
    finally:
        stop.set()
        t.join(timeout=10)
    assert not t.is_alive()
    assert out["r"].new == 2


def test_poll_mode_defers_files_still_being_written(cfg, tmp_db, tmp_path):
    root = tmp_path / "inbox"
    root.mkdir()
    stop, t, _ = _run(cfg, root, tmp_db, debounce_s=1.5, poll_interval=0.2,
                      use_watchdog=False, initial_scan=False)
    try:
        make_image(root / "fresh.jpg", rng_seed=3)
        time.sleep(0.7)
        assert _count(tmp_db) == 0  # still inside the debounce window
        assert _wait(lambda: _count(tmp_db) == 1)
    finally:
        stop.set()
        t.join(timeout=10)


def test_watch_requires_existing_target(cfg, tmp_db, tmp_path):
    with pytest.raises(FileNotFoundError):
        watch(tmp_path / "nope", tmp_db, cfg=cfg, stop_event=threading.Event())


def test_scan_paths_only_touches_given_files(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    a = make_image(root / "a.jpg", rng_seed=1)
    make_image(root / "b.jpg", rng_seed=2)
    res = FileScanner(cfg).scan_paths(root, [a, root / "notes.txt", tmp_path / "x.jpg"],
                                      db_path=tmp_db)
    assert res.new == 1 and _count(tmp_db) == 1


@pytest.mark.skipif(importlib.util.find_spec("watchdog") is None,
                    reason="watchdog not installed")
def test_watchdog_mode_ingests_new_files(cfg, tmp_db, tmp_path):
    root = tmp_path / "inbox"
    root.mkdir()
    stop, t, _ = _run(cfg, root, tmp_db, debounce_s=0.3, use_watchdog=True,
                      initial_scan=False)
    try:
        time.sleep(0.5)
        make_image(root / "evt.jpg", rng_seed=4)
        assert _wait(lambda: _count(tmp_db) == 1)
    finally:
        stop.set()
        t.join(timeout=10)
