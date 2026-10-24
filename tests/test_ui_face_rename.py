from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "photocure" / "ui" / "static"


def _js():
    return (STATIC / "app.js").read_text(encoding="utf-8")


def test_hover_no_longer_selects_a_face():
    js = _js()
    body = js[js.index("function setHoveredFace"):js.index("function selectFaceInOverlay")]
    assert "selectedFaceId" not in body


def test_click_opens_rename_popover_with_tick_button():
    js = _js()
    assert js.count("selectFaceInOverlay(${f.id}, this)") == 2
    body = js[js.index("function openFaceRename"):js.index("_faceRenameBound")]
    assert 'type: "submit"' in body and 'icon("check"' in body and "new_name" in body


def test_bottom_reassign_input_is_gone():
    assert "overlay-reassign-input" not in _js()


def test_escape_cancels_rename_before_closing_panel():
    js = _js()
    assert 'if (!closeFaceRename()) closeBiometricProjection()' in js


def test_person_photo_face_avatar_zooms_on_hover():
    js = _js()
    assert "showPpoFaceZoom(a)" in js and "hidePpoFaceZoom()" in js
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    block = css[css.index(".ppo-thumb-face {"):css.index(".ppo-face-zoom {")]
    assert "pointer-events: none" not in block


def test_face_rename_wires_autocomplete_with_tab_and_enter():
    js = _js()
    body = js[js.index("function openFaceRename"):js.index("_faceRenameBound")]
    assert "setupNameAutocomplete" in body
    assert "commitRename" in body

