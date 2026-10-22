"""People tab: nav entry, panel markup, and renderPeople() wiring."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATIC = ROOT / "photocure" / "ui" / "static"


def _index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_people_tab_exists_in_nav_and_markup():
    html = _index()
    assert 'data-tab="people"' in html and 'id="tab-people"' in html


def test_render_people_fetches_the_people_endpoint():
    js = _read("app.js")
    body = js.split("function renderPeople", 1)[1].split("\n  }\n", 1)[0]
    assert "/api/people" in body


def test_render_people_offers_rename_via_the_people_endpoint():
    js = _read("app.js")
    body = js.split("function renderPeople", 1)[1].split("\n  }\n", 1)[0]
    assert "/rename" in body


def test_people_tab_is_wired_into_show_tab():
    js = _read("app.js")
    body = js.split("function showTab", 1)[1].split("\n  }\n", 1)[0]
    assert '"people"' in body and "renderPeople" in body
