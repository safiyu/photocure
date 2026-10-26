"""The Trash stage in the UI: bar actions, toolbar, confirms and plain-words results."""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "photocure" / "ui" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text()


def test_trash_controls_are_in_the_markup():
    html = _read("index.html")
    for needle in ('id="bulk-restore"', 'id="bulk-return"', 'id="bulk-purge"', 'id="trash-toolbar"',
                   'id="btn-trash-reanalyse"', 'id="btn-trash-return-all"', 'id="btn-trash-empty"'):
        assert needle in html, needle
    for bid in ("bulk-restore", "bulk-return", "bulk-purge"):
        tag = re.search(r'<button[^>]*id="%s"[^>]*>' % bid, html).group(0)
        assert "trash-only" in tag, bid
    assert html.count('id="bulkbar"') == 1  # no second floating bar


def test_css_shows_trash_actions_only_in_the_trash_context():
    css = _read("style.css")
    assert re.search(r"\.bulk \.trash-only\s*\{\s*display:\s*none", css)
    assert re.search(r"\.bulk\.trash-ctx \.trash-only\s*\{\s*display:\s*inline-flex", css)
    assert re.search(r"\.bulk\.trash-ctx[^{]*\.lib-only[^{]*\{\s*display:\s*none", css)
    assert re.search(r'body\[data-stage="trash"\] #trash-toolbar\s*\{[^}]*display:\s*flex', css)
    assert re.search(r"#trash-toolbar\s*\{[^}]*display:\s*none", css)  # hidden outside the trash


def test_js_calls_every_trash_endpoint_and_reanalyses_with_the_trash_stage():
    js = _read("app.js")
    for path in ("/api/trash/restore", "/api/trash/return", "/api/trash/delete", "/api/trash/empty"):
        assert path in js, path
    assert re.search(r'post\("/api/infer",\s*\{[^}]*stage:\s*"trash"', js)
    assert 'classList.toggle("trash-ctx"' in js


def test_destructive_actions_confirm_and_restore_does_not():
    js = _read("app.js")
    body = lambda name: js.split(f"async function {name}", 1)[1].split("\n  }\n", 1)[0]  # noqa: E731
    for name in ("purgePhotos", "emptyTrash", "returnAllToOrigin"):
        assert "showConfirmDialog(" in body(name), name
    assert "variant: \"danger\"" in body("purgePhotos") and "variant: \"danger\"" in body("emptyTrash")
    assert "showConfirmDialog(" not in body("restoreSelected")
    assert "showConfirmDialog(" not in body("returnSelected")


def test_skipped_photos_are_explained_in_plain_words():
    js = _read("app.js")
    assert "the original folder is missing" in js
    assert "a file already sits at the origin" in js
    assert "the file is gone" in js


def test_cards_and_lightbox_offer_delete_permanently_in_the_trash():
    js = _read("app.js")
    assert "data-purge" in js and "Delete permanently" in js
    assert 'state.stage === "trash"' in js.split("function cardNode", 1)[1].split("const junkCardNode", 1)[0]
    assert 'state.stage === "trash"' in js.split('$("#lb-approve").onclick', 1)[1].split("\n", 1)[0]
