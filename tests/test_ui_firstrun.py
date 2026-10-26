"""First run: a blocking popup asks for the library folder, then the device, unless the environment set the root."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from photocure.ui.app import create_app

STATIC = Path(__file__).resolve().parent.parent / "photocure" / "ui" / "static"


def test_firstrun_markup_is_a_two_step_modal_with_no_way_out():
    html = (STATIC / "index.html").read_text()
    modal = re.search(r'<div id="firstrun" class="modal" hidden>(.*?)\n  </div>\n\n', html, re.S).group(1)
    for needle in ('id="firstrun-root"', 'id="firstrun-browse"', 'id="firstrun-accel"',
                   'id="firstrun-gpu-field"', 'id="firstrun-next"', 'id="firstrun-back"', 'id="firstrun-finish"'):
        assert needle in modal, needle
    assert 'id="firstrun-close"' not in modal and "icon-btn" not in modal  # nothing to dismiss it with
    assert html.index('id="firstrun"') < html.index('id="picker"')  # the folder picker opens above it


def test_firstrun_js_is_gated_on_the_root_and_its_source():
    js = (STATIC / "app.js").read_text()
    assert "function maybeFirstRun" in js
    gate = js.split("function maybeFirstRun", 1)[1].split("\n  }\n", 1)[0]
    assert "library_root_source" in gate and "environment" in gate and "library_root" in gate
    assert "trapFocus(" in gate  # Escape and Tab stay inside
    assert "/api/settings" in gate and "device_index" in gate and "library_root:" in gate.replace(" ", " ")
    assert "maybeFirstRun()" in js.split("async function init", 1)[1]
    assert "function renderAccelList" in js  # shared with the Settings page


def test_config_source_is_environment_when_the_env_var_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_LIBRARY_ROOT", str(tmp_path / "Lib"))
    app = create_app(str(tmp_path / "ui.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    j = client.get("/api/settings").json()
    assert j["library_root"] == str(tmp_path / "Lib") and j["library_root_source"] == "environment"
    assert client.get("/api/config").json()["library_root_source"] == "environment"


def test_one_settings_call_stores_folder_and_device_together(tmp_path):
    app = create_app(str(tmp_path / "ui.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    r = client.post("/api/settings", json={"library_root": str(tmp_path / "Lib"), "device": "cpu", "device_index": 0})
    assert r.status_code == 200
    j = client.get("/api/settings").json()
    assert j["library_root_source"] == "settings" and j["settings"]["device"] == "cpu"
