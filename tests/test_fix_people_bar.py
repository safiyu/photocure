import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "photocure/ui/static"
JS = (STATIC / "app.js").read_text()
HTML = (STATIC / "index.html").read_text()
CSS = (STATIC / "style.css").read_text()


def test_people_buttons_in_bar():
    for bid in ("bulk-merge", "bulk-name", "bulk-del-people"):
        m = re.search(r'<button[^>]*id="%s"[^>]*>' % bid, HTML)
        assert m and "people-only" in m.group(0)
    assert re.search(r'id="bulk-merge"[^>]*disabled', HTML)


def test_people_ctx_css():
    assert ".bulk.people-ctx .people-only" in CSS
    assert ".bulk.people-ctx .lib-only" in CSS and ".bulk.people-ctx .burst-only" in CSS
    assert ".people-card.picked" in CSS and ".people-card-select" in CSS


def test_apply_picks_handles_people():
    body = JS.split("function applyPicks()")[1].split("function syncPeopleSel")[0]
    assert "peoplePicks" in body and "people-ctx" in body
    assert "will become one person" in body and "select 2 or more to merge" in body


def test_select_toggle_and_clearing():
    assert 'aria-label="Select ' in JS and "aria-pressed" in JS
    assert "state.peoplePicks.clear()" in JS.split("function clearPicks()")[1][:120]
    assert "live.has(id)" in JS


def test_merge_target_rule_and_reuse():
    fn = JS.split("function pickMergeTarget")[1].split("window.pickMergeTarget")[0]
    assert "named.length === 1" in fn and "face_count" in fn and "a.id - b.id" in fn
    merge = JS.split("async function bulkMergePeople")[1].split("async function bulkDeletePeople")[0]
    assert "merge-into" in merge and "rippleDeleteCard" in merge and "showConfirmDialog" in merge
    assert "Photos stay in your library" in merge
