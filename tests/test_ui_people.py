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
    assert 'data-count="people"' in html


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


def test_people_count_wired_in_render_counts():
    js = _read("app.js")
    assert 'set("people", s.people);' in js
    assert "setPeopleCount" in js


def test_people_size_slider_exists_in_toolbar():
    html = _index()
    assert 'id="people-density"' in html
    assert 'class="people-density-slider"' in html
    assert '--people-card-min' in _read("style.css")


def test_people_size_slider_wired_in_app_js():
    js = _read("app.js")
    assert "setPeopleDens" in js
    assert "--people-card-min" in js
    assert "pcur-people-density" in js


def test_people_name_autocomplete_wired_with_tab_and_enter():
    js = _read("app.js")
    assert "function setupNameAutocomplete" in js
    body = js[js.index("function setupNameAutocomplete"):js.index("function _peopleFP")]
    assert 'e.key === "Tab"' in body
    assert 'e.key === "Enter"' in body
    assert 'e.key === "ArrowDown"' in body
    assert 'e.key === "ArrowUp"' in body
    assert "name-autocomplete-menu" in body
    assert "setupNameAutocomplete(input, getKnownPeopleNames" in js
    css = _read("style.css")
    assert ".name-autocomplete-menu" in css
    assert ".name-autocomplete-item.active" in css

