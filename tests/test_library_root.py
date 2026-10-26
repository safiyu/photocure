"""The single library root: where it comes from, when it locks, and what depends on it."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from photocure import library
from photocure.config import CuratorConfig
from photocure.services import load_settings, save_settings
from photocure.ui.app import create_app


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch):
    monkeypatch.delenv("PCUR_LIBRARY_ROOT", raising=False)


def test_settings_file_beats_environment(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_LIBRARY_ROOT", str(tmp_path / "from_env"))
    save_settings(tmp_db, {"library_root": str(tmp_path / "from_settings")})
    assert library.library_root(tmp_db) == (tmp_path / "from_settings").resolve()
    assert library.library_root_source(tmp_db) == "settings"


def test_environment_used_when_no_setting(tmp_db, tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_LIBRARY_ROOT", str(tmp_path / "from_env"))
    assert library.library_root(tmp_db) == tmp_path / "from_env"
    assert library.library_root_source(tmp_db) == "environment"


def test_require_raises_when_unset(tmp_db):
    assert library.library_root(tmp_db) is None
    assert library.library_root_source(tmp_db) == "none"
    with pytest.raises(library.LibraryRootMissing, match="library folder not set"):
        library.require_library_root(tmp_db)


def test_library_root_cannot_change_once_set(tmp_db, tmp_path):
    save_settings(tmp_db, {"library_root": str(tmp_path / "a")})
    save_settings(tmp_db, {"library_root": str(tmp_path / "a")})  # same value is fine
    with pytest.raises(ValueError, match="already set"):
        save_settings(tmp_db, {"library_root": str(tmp_path / "b")})
    with pytest.raises(ValueError, match="already set"):
        save_settings(tmp_db, {"library_root": None})
    save_settings(tmp_db, {"library_root": str(tmp_path / "b")}, force=True)
    assert load_settings(tmp_db)["library_root"] == str((tmp_path / "b").resolve())


def test_default_operation_validated(tmp_db):
    save_settings(tmp_db, {"default_operation": "copy"})
    assert load_settings(tmp_db)["default_operation"] == "copy"
    for bad in ("hardlink", "symlink", "", "MOVE ME"):
        with pytest.raises(ValueError):
            save_settings(tmp_db, {"default_operation": bad})


def test_trash_dir_is_under_the_root(tmp_path):
    assert library.trash_dir(tmp_path / "Lib") == tmp_path / "Lib" / ".trash"


@pytest.mark.parametrize("bad", ["relative/dir", "/", "/etc/photos", "/usr/local/x", ""])
def test_validate_library_root_rejects(tmp_db, bad):
    with pytest.raises(library.LibraryRootError) as e:
        library.validate_library_root(bad, tmp_db)
    assert e.value.status == 400


def test_validate_library_root_accepts_a_fresh_folder(tmp_db, tmp_path):
    assert library.validate_library_root(str(tmp_path / "Library"), tmp_db) == (tmp_path / "Library").resolve()


def test_validate_library_root_confinement_is_403(tmp_db, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(library.LibraryRootError) as e:
        library.validate_library_root(str(tmp_path / "x" / "y"), tmp_db, browse_roots=[other])
    assert e.value.status == 403


def test_category_folder_uses_taxonomy_folder(cfg):
    assert library.category_folder(cfg, "people") == "People"
    assert library.category_folder(cfg, "made_up_class") == "made_up_class"


def test_library_target_is_root_folder_filename(cfg, tmp_path):
    class Rec:
        filename = "IMG_1.jpg"
    assert library.library_target(cfg, tmp_path / "Lib", Rec, "people") == tmp_path / "Lib" / "People" / "IMG_1.jpg"


def test_config_endpoint_reports_root_and_source(tmp_path):
    app = create_app(str(tmp_path / "ui.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    j = client.get("/api/config").json()
    assert j["library_root"] is None
    assert j["library_root_source"] == "none"
    assert j["default_operation"] == "move"
    assert "default_output_dir" not in j
    assert client.post("/api/settings", json={"library_root": str(tmp_path / "Lib")}).status_code == 200
    j = client.get("/api/config").json()
    assert j["library_root"] == str((tmp_path / "Lib").resolve())
    assert j["library_root_source"] == "settings"


def test_settings_endpoint_refuses_to_change_a_set_root(tmp_path):
    app = create_app(str(tmp_path / "ui.db"))
    client = TestClient(app)
    client.headers["X-PCUR-Token"] = app.state.token
    assert client.post("/api/settings", json={"library_root": str(tmp_path / "a")}).status_code == 200
    r = client.post("/api/settings", json={"library_root": str(tmp_path / "b")})
    assert r.status_code == 400
    assert "already set" in r.json()["detail"]


def test_trash_retention_setting_is_gone():
    assert not hasattr(CuratorConfig(), "trash_retention_days")
