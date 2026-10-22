"""Face overlay generalized to standalone (non-burst) photos, with inline reassign."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATIC = ROOT / "photocure" / "ui" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _fn(js: str, name: str) -> str:
    return js.split(f"function {name}", 1)[1].split("\n  }\n", 1)[0]


def test_open_projection_skips_the_burst_fetch_when_burst_id_is_falsy():
    js = _read("app.js")
    body = _fn(js, "openBiometricProjection")
    assert "if (burstId)" in body or "burstId ?" in body


def test_reassign_face_function_posts_to_the_reassign_endpoint():
    js = _read("app.js")
    assert "function reassignFace" in js
    body = js.split("function reassignFace", 1)[1].split("\n  }\n", 1)[0]
    assert "/reassign" in body


def test_empty_faces_shows_the_no_faces_message():
    js = _read("app.js")
    body = _fn(js, "openBiometricProjection")
    assert "No faces were found in this photo" in body


def test_burst_only_actions_are_guarded_when_there_is_no_burst():
    js = _read("app.js")
    body = _fn(js, "renderBiometricOverlay")
    assert "activeOverlayBurstId ?" in body or "if (activeOverlayBurstId)" in body


def test_reassign_face_is_exposed_to_inline_handlers():
    js = _read("app.js")
    assert "window.reassignFace = reassignFace" in js
