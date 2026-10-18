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
