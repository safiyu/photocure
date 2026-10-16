"""Regression tests for the scanner/db review findings (XMP, sync, watcher, mtime, migrate)."""
from __future__ import annotations

import os
import sqlite3
import threading
import time

from sqlmodel import select

from photocure.db.models import ImageRecord
from photocure.db.session import close_engine, get_session, init_db, migrate
from photocure.scanner.file_scanner import FileScanner
from photocure.scanner.xmp import (get_on_disk_owned_tags, get_on_disk_rating, parse_xmp,
                                   update_xmp_sidecar)
from photocure.sidecar_sync import EDITED_ELSEWHERE, EXISTING_FOREIGN, sync_sidecar
from tests.conftest import make_image
from tests.test_watcher import _run, _wait

SPLIT = ('<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
         'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
         'xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
         'xmlns:dc="http://purl.org/dc/elements/1.1/">'
         '<rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
         'crs:Exposure2012="+0.35"/>'
         '<rdf:Description rdf:about="" xmp:Rating="2" xmp:Label="Red">'
         '<dc:subject><rdf:Bag><rdf:li>photocure-old</rdf:li></rdf:Bag></dc:subject>'
         '</rdf:Description></rdf:RDF></x:xmpmeta>')


# ------------------------------------------------------------ xmp writer
def test_writer_updates_rating_label_and_tags_in_later_description(tmp_path):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"x")
    sc = tmp_path / "a.jpg.xmp"
    sc.write_text(SPLIT)
    ok, _ = update_xmp_sidecar(photo, rating=5, label="Blue",
                               add_tags=("photocure-new",), remove_tags=("photocure-old",))
    assert ok
    text = sc.read_text()
    assert text.count("Rating=") + text.count("<xmp:Rating") == 1  # no duplicate rating
    assert 'xmp:Rating="2"' not in text
    assert text.count("<dc:subject") == 1
    info = parse_xmp(sc.read_bytes())
    assert info.rating == 5 and info.label == "Blue"
    assert info.keywords == ["photocure-new"]
    assert "+0.35" in text


# ----------------------------------------------------------- sidecar sync
def test_tag_only_write_over_foreign_rating_does_not_lock_future_syncs(tmp_path):
    photo = tmp_path / "IMG.JPG"
    photo.write_bytes(b"x")
    (tmp_path / "IMG.JPG.xmp").write_text(SPLIT.replace("photocure-old", "other"))
    img = ImageRecord(path=str(photo), filename="IMG.JPG")
    assert sync_sidecar(img, rating=3, tags=("photocure-best-shot",)).status == EXISTING_FOREIGN
    assert img.xmp_written_rating is None
    again = sync_sidecar(img, rating=3, tags=("photocure-blink",))
    assert again.status != EDITED_ELSEWHERE
    assert get_on_disk_owned_tags(photo) == {"photocure-blink"}
    assert get_on_disk_rating(photo) == 2  # foreign rating still preserved


def test_dual_sidecars_read_what_the_scanner_reads(tmp_path):
    photo = tmp_path / "IMG.JPG"
    photo.write_bytes(b"x")
    (tmp_path / "IMG.xmp").write_text(SPLIT)
    (tmp_path / "IMG.JPG.xmp").write_text(SPLIT.replace('Rating="2"', 'Rating="1"'))
    assert get_on_disk_rating(photo) == 2  # same file read_xmp (IMG.xmp) reports
    assert get_on_disk_owned_tags(photo) == {"photocure-old"}


# ----------------------------------------------------------------- mtime
def test_same_size_replacement_with_older_mtime_is_rescanned(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    p = make_image(root / "a.jpg", rng_seed=1)
    old = time.time() - 3600
    os.utime(p, (old, old))
    sc = FileScanner(cfg)
    assert sc.scan_directory(root, db_path=tmp_db, show_progress=False).new == 1
    size = p.stat().st_size
    raw = bytearray(p.read_bytes())
    raw[-3] ^= 0x01  # same size, different bytes
    p.write_bytes(bytes(raw))
    assert p.stat().st_size == size
    older = old - 7200
    os.utime(p, (older, older))
    res = sc.scan_directory(root, db_path=tmp_db, show_progress=False)
    assert res.modified == 1 and res.skipped_known == 0
    res = sc.scan_directory(root, db_path=tmp_db, show_progress=False)
    assert res.modified == 0 and res.skipped_known == 1


def test_legacy_rows_without_mtime_fall_back_to_created_at(cfg, tmp_db, tmp_path):
    root = tmp_path / "lib"
    p = make_image(root / "a.jpg", rng_seed=1)
    sc = FileScanner(cfg)
    sc.scan_directory(root, db_path=tmp_db, show_progress=False)
    with get_session(tmp_db) as s:
        for r in s.exec(select(ImageRecord)).all():
            r.file_mtime = None
    res = sc.scan_directory(root, db_path=tmp_db, show_progress=False)
    assert res.modified == 0 and res.skipped_known == 1
    future = time.time() + 600
    os.utime(p, (future, future))
    assert sc.scan_directory(root, db_path=tmp_db, show_progress=False).modified == 1


# --------------------------------------------------------------- migrate
def test_migrate_adds_file_mtime_and_not_null_callable_default(tmp_path):
    db = tmp_path / "old.db"
    init_db(db)
    close_engine(db)
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE images DROP COLUMN file_mtime")
    con.close()
    init_db(db)
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(images)")}
    con.close()
    assert "file_mtime" in cols

    from sqlalchemy import Column, DateTime, MetaData, Table
    from sqlmodel import SQLModel
    from photocure.db.models import utcnow
    t = Table("fix_probe", SQLModel.metadata, Column("id", DateTime, primary_key=True),
              extend_existing=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE fix_probe (id TEXT PRIMARY KEY)")
    con.execute("INSERT INTO fix_probe VALUES ('x')")
    con.commit()
    con.close()
    t.append_column(Column("stamp", DateTime, nullable=False, default=utcnow))
    try:
        from photocure.db.session import get_engine
        assert migrate(get_engine(db)) >= 1
        con = sqlite3.connect(db)
        assert con.execute("SELECT stamp FROM fix_probe").fetchone()[0]
        con.close()
    finally:
        SQLModel.metadata.remove(t)
        close_engine(db)


# --------------------------------------------------------------- watcher
def _paths(db):
    with get_session(db) as s:
        return {r.path: r.scan_status for r in s.exec(select(ImageRecord)).all()}


def test_watchdog_marks_deleted_and_moved_away_files_missing(cfg, tmp_db, tmp_path):
    root = tmp_path / "inbox"
    a = make_image(root / "a.jpg", rng_seed=1)
    b = make_image(root / "b.jpg", rng_seed=2)
    old = time.time() - 60
    for f in (a, b):
        os.utime(f, (old, old))
    stop, t, _ = _run(cfg, root, tmp_db, debounce_s=0.3, use_watchdog=True,
                      prune_missing=True)
    try:
        assert _wait(lambda: len(_paths(tmp_db)) == 2)
        time.sleep(0.5)
        a.unlink()
        b.rename(tmp_path / "elsewhere.jpg")
        assert _wait(lambda: set(_paths(tmp_db).values()) == {"missing"})
    finally:
        stop.set()
        t.join(timeout=10)


def test_watchdog_ingests_folder_moved_into_root(cfg, tmp_db, tmp_path):
    root = tmp_path / "inbox"
    root.mkdir()
    outside = tmp_path / "outside" / "trip"
    make_image(outside / "x.jpg", rng_seed=5)
    make_image(outside / "y.jpg", rng_seed=6)
    old = time.time() - 60
    for f in outside.iterdir():
        os.utime(f, (old, old))
    stop, t, _ = _run(cfg, root, tmp_db, debounce_s=0.3, use_watchdog=True,
                      initial_scan=False)
    try:
        time.sleep(0.5)
        outside.rename(root / "trip")
        assert _wait(lambda: len(_paths(tmp_db)) == 2)
    finally:
        stop.set()
        t.join(timeout=10)
