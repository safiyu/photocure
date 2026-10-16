"""The app is called photocure; older names and data keep working."""
from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from photocure import config as cfgmod

ROOT = Path(__file__).resolve().parent.parent


def test_package_and_entry_points():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert meta["project"]["name"] == "photocure"
    scripts = meta["project"]["scripts"]
    assert scripts["photocure"] == "photocure.cli:cli"
    assert scripts["photo-curator"] == "photocure.cli:cli"      # old command still works
    assert meta["tool"]["setuptools"]["package-data"] == {"photocure": ["ui/static/*", "models/*"]}
    assert not (ROOT / "photo_curator").exists()


def test_no_old_name_left_in_source_files():
    bad = []
    for base in ("photocure", "tests", "docker"):
        for p in (ROOT / base).rglob("*"):
            if not p.is_file() or p.suffix in (".pyc", ".png", ".db") or "__pycache__" in p.parts:
                continue
            if p.name in ("test_rename.py", "entrypoint.sh", "config.py"):
                continue  # these mention the old name on purpose, for old databases
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "photo_curator" in text or "photo-curator" in text or "Photo Curator" in text:
                bad.append(str(p.relative_to(ROOT)))
    assert bad == []


def test_ui_uses_the_new_display_name():
    html = (ROOT / "photocure" / "ui" / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "photocure" / "ui" / "static" / "app.js").read_text(encoding="utf-8")
    assert "<title>Photocure</title>" in html and "<b>Photocure</b>" in html
    assert "<span>Intelligent image curator</span>" in html
    assert "Works fully offline" not in html
    assert "Photo Curator" not in html + js


# ------------------------------------------------------- existing databases keep working
def test_default_db_prefers_the_new_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cfgmod.default_db_path() == Path("photocure.db")                  # nothing yet
    (tmp_path / "photo_curator.db").write_text("x")
    assert cfgmod.default_db_path() == Path("photo_curator.db")              # only the old one
    (tmp_path / "photocure.db").write_text("x")
    assert cfgmod.default_db_path() == Path("photocure.db")                  # both: new wins


def test_config_default_follows_the_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PCUR_DB_PATH", raising=False)
    (tmp_path / "photo_curator.db").write_text("x")
    assert cfgmod.CuratorConfig().db_path == Path("photo_curator.db")


def test_cli_without_db_flag_keeps_using_an_old_database(tmp_path):
    """Run the real command in a folder that only has a database from the old name."""
    import sqlite3
    old = tmp_path / "photo_curator.db"
    sqlite3.connect(old).close()
    env = {**os.environ, "PCUR_BACKEND": "stub"}
    r = subprocess.run([sys.executable, "-m", "photocure.cli", "stats"], cwd=tmp_path, env={**env, "PYTHONPATH": str(ROOT)},
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-300:]
    assert "photo_curator.db" in r.stdout.replace("\n", "")
    assert not (tmp_path / "photocure.db").exists()


# ------------------------------------------------------------------ Docker volume migration
def run_entrypoint(tmp_path, files):
    data, bin_ = tmp_path / "data", tmp_path / "bin"
    data.mkdir()
    bin_.mkdir()
    for name in files:
        (data / name).write_text(name)
    (bin_ / "python").write_text(f"#!/bin/sh\necho {tmp_path}/site\n")
    (bin_ / "photocure").write_text(f'#!/bin/sh\necho "$@" > {tmp_path}/args\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "PCUR_DATA_DIR": str(data)}
    env.pop("PYTHONUSERBASE", None)
    subprocess.run(["sh", str(ROOT / "docker" / "entrypoint.sh"), "auto"], env=env, check=True)
    return data, (tmp_path / "args").read_text()


def test_entrypoint_migrates_an_old_volume(tmp_path):
    data, args = run_entrypoint(tmp_path, ["photo_curator.db", "photo_curator.db-wal",
                                           "photo_curator.db-shm", "photo_curator.settings.json"])
    assert sorted(p.name for p in data.iterdir()) == [
        "photocure.db", "photocure.db-shm", "photocure.db-wal", "photocure.settings.json"]
    assert (data / "photocure.db").read_text() == "photo_curator.db"      # same content, new name
    assert f"--db {data}/photocure.db" in args and args.startswith("setup --yes")


def test_entrypoint_never_overwrites_a_new_database(tmp_path):
    data, _ = run_entrypoint(tmp_path, ["photo_curator.db", "photocure.db"])
    assert (data / "photocure.db").read_text() == "photocure.db"
    assert (data / "photo_curator.db").exists()


def test_entrypoint_on_a_fresh_volume(tmp_path):
    data, args = run_entrypoint(tmp_path, [])
    assert list(data.iterdir()) == [] and "setup --yes" in args


# ------------------------------------------------------------------------------ logo
LOGO_FILES = {"logo.png": (192, 192), "favicon.png": (64, 64), "apple-touch-icon.png": (180, 180)}


def test_logo_assets_are_transparent_squares():
    from PIL import Image

    static = ROOT / "photocure" / "ui" / "static"
    for name, size in LOGO_FILES.items():
        im = Image.open(static / name)
        assert im.size == size, name
        if name == "apple-touch-icon.png":
            # iOS fills transparency with black, so this one is a flat white tile on purpose
            assert im.mode == "RGB" and im.getpixel((0, 0)) == (255, 255, 255)
            continue
        im = im.convert("RGBA")
        assert im.getpixel((0, 0))[3] == 0, f"{name} corner should be transparent"
        assert im.getpixel((size[0] // 2, size[1] // 2))[3] == 255, f"{name} centre should be opaque"
    docs = Image.open(ROOT / "docs" / "logo.png")
    assert docs.size[0] == docs.size[1] and docs.size[0] >= 512


def test_page_links_and_serves_the_logo(tmp_path):
    from fastapi.testclient import TestClient

    from photocure.ui.app import create_app

    app = create_app(str(tmp_path / "logo.db"))
    c = TestClient(app)
    html = c.get("/").text
    for ref in ("/static/logo.png", "/static/favicon.png", "/static/apple-touch-icon.png"):
        assert ref in html
        r = c.get(ref)
        assert r.status_code == 200 and r.headers["content-type"] == "image/png", ref
    assert "i-aperture" not in html            # the old drawn mark is gone


def test_readme_shows_the_logo():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'src="docs/logo.png"' in readme and (ROOT / "docs" / "logo.png").exists()
