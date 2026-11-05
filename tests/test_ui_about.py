"""The version badge opens the About section, and About is a styled card, not inline-styled text."""
import re
from pathlib import Path

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_version_badge_targets_the_about_section():
    html = _read("index.html")
    badge = re.search(r'<button[^>]*id="sidebar-version"[^>]*>', html).group(0)
    assert 'data-goto="settings"' in badge and 'data-anchor="sect-about"' in badge
    assert re.search(r'id="sect-about"[^>]*tabindex="-1"', html)


def test_goto_handlers_scroll_to_the_anchor():
    js = _read("app.js")
    assert re.search(r"function gotoTarget\(", js)
    assert js.count("gotoTarget(") >= 3, "both goto handlers must use gotoTarget"
    assert "dataset.anchor" in js.split("function gotoTarget(", 1)[1].split("\n  }\n", 1)[0]


def test_about_card_uses_classes_not_inline_styles():
    js = _read("app.js")
    body = js.split('const aboutCard = $("#about-card");', 1)[1].split("} catch (e)", 1)[0]
    assert "style=" not in body
    for cls in ("about-name", "about-ver", "about-facts", "about-links"):
        assert cls in body, cls


def test_about_styles_exist_in_both_themes():
    css = re.sub(r"/\*.*?\*/", "", _read("style.css"), flags=re.S)
    for sel in (".about {", ".about-facts {", ".about-ver {", ".about-chip {"):
        assert sel in css, sel
    assert ':root[data-theme="dark"] .about-chip' in css


def test_notice_declares_every_model_in_the_manifest():
    import json
    root = Path(__file__).parent.parent
    notice = (root / "NOTICE").read_text(encoding="utf-8")
    manifest = json.loads((root / "photocure/models/MANIFEST.json").read_text(encoding="utf-8"))
    for m in manifest["models"]:
        assert m["file"] in notice, f"NOTICE is missing {m['file']}"
    for needle in ("rembg", "BiRefNet", "U2-Net", "RapidOCR", "ESPCN", "SFace"):
        assert needle in notice, needle


def test_downloaded_models_are_listed_only_when_present(tmp_path, monkeypatch):
    import json
    from photocure import models
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"models": [], "downloaded": [
        {"name": "A", "license": "MIT", "cache_file": "a.onnx"},
        {"name": "B", "license": "MIT", "cache_file": "b.onnx"},
        {"name": "C", "license": "MIT", "module": "no_such_module_xyz"}]}))
    home = tmp_path / "u2"
    home.mkdir()
    (home / "a.onnx").write_bytes(b"x")
    monkeypatch.setenv("U2NET_HOME", str(home))
    assert [m["name"] for m in models.downloaded_models(tmp_path)] == ["A"]


def test_readme_does_not_claim_unshipped_models():
    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    for word in ("SigLIP", "TOPIQ", "LIQE", "multilingual"):
        assert word not in text, word


def test_manifest_and_notice_have_no_siglip_entry():
    root = Path(__file__).resolve().parent.parent
    assert "siglip" not in (root / "photocure/models/MANIFEST.json").read_text(encoding="utf-8").lower()
    assert "SigLIP" not in (root / "NOTICE").read_text(encoding="utf-8")


def test_model_path_has_no_developer_paths():
    root = Path(__file__).resolve().parent.parent
    assert "/home/safiyu" not in (root / "photocure/models/__init__.py").read_text(encoding="utf-8")
