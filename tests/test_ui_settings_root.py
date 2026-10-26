"""Settings page: the library folder (set once), where it came from, and the default operation."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from photocure.ui.app import create_app

STATIC = Path(__file__).resolve().parent.parent / "photocure" / "ui" / "static"


def _html() -> str:
    return (STATIC / "index.html").read_text()


def test_settings_markup_has_library_folder_controls_first():
    html = _html()
    panel = re.search(r'<section id="tab-settings"[^>]*>(.*?)</section>', html, re.S).group(1)
    for needle in ('id="set-library-root"', 'id="btn-browse-library"', 'id="library-root-source"',
                   'id="btn-move-library"', 'id="set-default-op"'):
        assert needle in panel, needle
    assert re.search(r'id="set-library-root"[^>]*readonly', panel)
    first_sect = re.search(r'<div class="sect"[^>]*>(.*?)</div>\s*</div>', panel, re.S).group(1)
    assert "Library folder" in first_sect
    assert panel.index("Library folder") < panel.index("Processing device")
    ops = re.search(r'<select id="set-default-op">(.*?)</select>', panel, re.S).group(1)
    assert 'value="move"' in ops and 'value="copy"' in ops and "hardlink" not in ops


def test_settings_js_wires_the_folder_and_default_operation():
    js = (STATIC / "app.js").read_text()
    assert "library_root_source" in js and "default_operation" in js
    save = js.split("async function saveSettings", 1)[1].split("\n  }\n", 1)[0]
    assert "default_operation" in save and "library_root" in save
    assert "btn-browse-library" in js


def test_settings_payload_reports_root_source_and_default_operation(tmp_path):
    app = create_app(str(tmp_path / "ui.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    j = client.get("/api/settings").json()
    assert j["library_root"] is None and j["library_root_source"] == "none"
    assert j["default_operation"] == "move"
    r = client.post("/api/settings", json={"library_root": str(tmp_path / "Lib"), "default_operation": "copy"})
    assert r.status_code == 200
    j = client.get("/api/settings").json()
    assert j["library_root"] == str((tmp_path / "Lib").resolve())
    assert j["library_root_source"] == "settings"
    assert j["default_operation"] == "copy"
    assert j["settings"]["default_operation"] == "copy"
