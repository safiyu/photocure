# ruff: noqa: E501
"""Duplicates page: every photo says Keep or Duplicate, one sentence says what Confirm does."""
import re
from pathlib import Path

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _dup_js() -> str:
    js = _read("app.js")
    return js.split("function dupMemberNode", 1)[1].split("function dupSummaryText", 1)[0]


def test_page_explains_itself_and_the_bulk_button_is_clear():
    html = _read("index.html")
    tab = html.split('id="tab-duplicates"', 1)[1].split("</section>", 1)[0]
    intro = re.search(r'<p class="dup-intro"[^>]*>(.*?)</p>', tab, re.S)
    assert intro and "Nothing is deleted or moved" in intro.group(1)
    assert re.search(r'id="btn-dup-all"[^>]*>\s*Accept all suggestions', tab)


def test_tiles_are_labelled_keep_or_duplicate_and_only_others_get_a_button():
    js = _dup_js()
    assert 'dup-badge' in js and '"Keep"' in js and '"Duplicate"' in js
    assert "Keep this one instead" in js
    assert "Keeping this one" not in js and "Keep this one," not in js


def test_group_footer_states_the_outcome_with_a_single_confirm_button():
    js = _dup_js()
    assert "dup-plan" in js and "as a duplicate" in js
    assert 'text: "Confirm"' in js
    assert "discard ${others.length}" not in js


def test_bulk_dialog_uses_the_same_vocabulary():
    js = _read("app.js")
    body = js.split("async function dupResolveAll", 1)[1].split("function bindDuplicates", 1)[0]
    assert "Accept all" in body and "Resolve all" not in body


def test_badge_and_plan_styles_exist_for_both_themes():
    css = re.sub(r"/\*.*?\*/", "", _read("style.css"), flags=re.S)
    for sel in (".dup-badge {", ".dup-badge.keep {", ".dup-badge.out {", ".dup-plan {", ".dup-intro {"):
        assert sel in css, sel
    assert "position: relative" in re.search(r"\.dup-thumb\s*\{([^}]*)\}", css).group(1)
