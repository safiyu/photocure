"""C4: offline reverse geocoding, batched, optional."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from sqlmodel import select
from typer.testing import CliRunner

from photocure import cli as cli_mod
from photocure.config import CuratorConfig
from photocure.db import models as M
from photocure.db.session import get_session, init_db
from photocure.scanner import geo
from photocure.scanner.file_scanner import FileScanner

runner = CliRunner()


def fake_search(calls):
    def search(coords):
        calls.append(len(coords))
        return [{"name": f"City{round(lat)}", "admin1": "Region", "cc": "ma" if lon < 0 else "fr"} for lat, lon in coords]
    return search


def add_photos(db, n, with_gps=True):
    init_db(db, drop=True)
    with get_session(db) as s:
        for i in range(n):
            s.add(M.ImageRecord(path=f"/p/{i}.jpg", filename=f"{i}.jpg",
                                gps_lat=30.0 + i * 0.001 if with_gps else None,
                                gps_lon=-7.0 if i % 2 else 2.0 if with_gps else None))


def test_lookups_run_in_batches_of_500(tmp_path):
    db = tmp_path / "g.db"
    add_photos(db, 1200)
    calls = []
    with get_session(db) as s:
        assert geo.geocode_pending(s, batch=500, search=fake_search(calls)) == 1200
    assert calls == [500, 500, 200]
    with get_session(db) as s:
        rows = s.exec(select(M.ImageRecord)).all()
        assert all(r.country_code in ("MA", "FR") and r.region == "Region" and r.city.startswith("City") for r in rows)


def test_only_photos_with_gps_and_no_place_yet_are_looked_up(tmp_path):
    db = tmp_path / "g.db"
    init_db(db, drop=True)
    with get_session(db) as s:
        s.add(M.ImageRecord(path="/a", filename="a", gps_lat=33.5, gps_lon=-7.6))                       # to do
        s.add(M.ImageRecord(path="/b", filename="b"))                                                    # no GPS
        s.add(M.ImageRecord(path="/c", filename="c", gps_lat=48.8, gps_lon=2.3, country_code="FR"))      # done
    calls = []
    with get_session(db) as s:
        assert geo.geocode_pending(s, search=fake_search(calls)) == 1
    assert calls == [1]
    with get_session(db) as s:
        assert geo.geocode_pending(s, search=fake_search(calls)) == 0   # nothing left
    assert calls == [1]


def test_a_failing_batch_is_skipped_and_the_rest_still_done(tmp_path):
    db = tmp_path / "g.db"
    add_photos(db, 30)
    n = {"i": 0}

    def flaky(coords):
        n["i"] += 1
        if n["i"] == 2:
            raise ValueError("bad coordinate")
        return [{"name": "X", "admin1": "Y", "cc": "MA"} for _ in coords]
    with get_session(db) as s:
        assert geo.geocode_pending(s, batch=10, search=flaky) == 20
    with get_session(db) as s:
        assert len([r for r in s.exec(select(M.ImageRecord)).all() if r.country_code]) == 20


def test_a_hit_without_a_country_is_left_unfilled(tmp_path):
    db = tmp_path / "g.db"
    add_photos(db, 3)
    with get_session(db) as s:
        assert geo.geocode_pending(s, search=lambda c: [{"name": "?", "admin1": "", "cc": ""}] * len(c)) == 0


def test_without_the_package_nothing_happens(tmp_path, monkeypatch):
    db = tmp_path / "g.db"
    add_photos(db, 3)
    monkeypatch.setattr(geo, "available", lambda: False)
    with get_session(db) as s:
        assert geo.geocode_pending(s) == 0


def gps_jpeg(path: Path, lat: float, lon: float) -> Path:
    exif = Image.Exif()
    gps = exif.get_ifd(0x8825)
    gps[1], gps[3] = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    for tag, v in ((2, abs(lat)), (4, abs(lon))):
        d = int(v)
        m = int((v - d) * 60)
        gps[tag] = (float(d), float(m), round((v - d - m / 60) * 3600, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), (120, 120, 120)).save(path, exif=exif.tobytes())
    return path


def scan(db, root):
    cfg = CuratorConfig()
    cfg.db_path = db
    return FileScanner(cfg).scan_directory(root, db_path=db, show_progress=False)


def test_a_scan_fills_places_and_a_changed_position_is_looked_up_again(tmp_path, monkeypatch):
    db, root = tmp_path / "s.db", tmp_path / "lib"
    init_db(db, drop=True)
    calls = []
    monkeypatch.setattr(geo, "_default_search", lambda: fake_search(calls))
    monkeypatch.setattr(geo, "available", lambda: True)
    gps_jpeg(root / "a.jpg", 33.5731, -7.5898)
    res = scan(db, root)
    assert res.geocoded == 1
    with get_session(db) as s:
        rec = s.exec(select(M.ImageRecord)).one()
        assert abs(rec.gps_lat - 33.5731) < 0.01 and rec.country_code == "MA" and rec.city == "City34"
    # the photo is replaced by one taken elsewhere: its place must be looked up again
    import os
    import time
    gps_jpeg(root / "a.jpg", 48.8566, 2.3522)
    os.utime(root / "a.jpg", (time.time() + 5, time.time() + 5))
    scan(db, root)
    with get_session(db) as s:
        assert s.exec(select(M.ImageRecord)).one().country_code == "FR"


def test_geocoding_can_be_turned_off(tmp_path, monkeypatch):
    db, root = tmp_path / "s.db", tmp_path / "lib"
    init_db(db, drop=True)
    calls = []
    monkeypatch.setattr(geo, "_default_search", lambda: fake_search(calls))
    monkeypatch.setattr(geo, "available", lambda: True)
    gps_jpeg(root / "a.jpg", 33.5, -7.5)
    cfg = CuratorConfig()
    cfg.db_path, cfg.geocode = db, False
    FileScanner(cfg).scan_directory(root, db_path=db, show_progress=False)
    assert calls == []


def test_reverse_geocoder_is_a_required_dependency():
    import tomllib
    data = tomllib.loads(Path("pyproject.toml").read_text())
    assert any(d.startswith("reverse_geocoder") for d in data["project"]["dependencies"])
    assert "geo" not in data["project"]["optional-dependencies"]


def test_geocode_command_reports_a_missing_package_without_a_stale_extra_name(tmp_path, monkeypatch):
    monkeypatch.setattr(geo, "available", lambda: False)
    r = runner.invoke(cli_mod.app, ["geocode", "--db", str(tmp_path / "x.db")])
    out = " ".join(r.output.split())
    assert r.exit_code == 1 and "photocure[geo]" not in out and "pip install" in out


def test_geocode_command_fills_an_existing_library(tmp_path, monkeypatch):
    db = tmp_path / "c.db"
    add_photos(db, 4)
    monkeypatch.setattr(geo, "available", lambda: True)
    monkeypatch.setattr(geo, "_default_search", lambda: fake_search([]))
    r = runner.invoke(cli_mod.app, ["geocode", "--db", str(db)])
    assert r.exit_code == 0 and "Found a place for 4 photos" in " ".join(r.output.split())


def test_the_real_geocoder_when_installed():
    pytest.importorskip("reverse_geocoder")
    search = geo._default_search()
    casa, paris = search([(33.5731, -7.5898), (48.8566, 2.3522)])
    assert (casa["cc"], casa["name"]) == ("MA", "Casablanca") and paris["cc"] == "FR"
    assert len(search([(35.0, -5.0)] * 500)) == 500
