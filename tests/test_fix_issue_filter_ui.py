from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "photocure" / "ui" / "static"


def test_quality_filter_is_wired_through_the_grid_filters():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="f-issue"' in html
    assert 'p.set("issue", val("f-issue"))' in js
    assert "hasIssue(it, val(\"f-issue\"))" in js
    assert '"f-issue"].forEach' in js  # reset and change listeners
    assert "f.issues" in js
