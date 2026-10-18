"""One progress display: the bar. The speedometer is gone and the phase moved into the card."""
import re
from pathlib import Path

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_speedometer_is_gone():
    assert "gauge" not in _read("index.html").lower()
    assert "gauge" not in _read("app.js").lower()
    assert not re.search(r"\.gauge", _read("style.css"))


def test_progress_bar_is_the_accessible_progressbar():
    html = _read("index.html")
    tag = re.search(r'<div[^>]*class="progress-track"[^>]*>', html).group(0)
    for needle in ('role="progressbar"', 'aria-valuemin="0"', 'aria-valuemax="100"', "aria-label"):
        assert needle in tag, needle
    assert 'setAttribute("aria-valuenow"' in _read("app.js")


def test_phase_label_lives_in_the_progress_card():
    html, js = _read("index.html"), _read("app.js")
    card = html.split('id="progress-card"', 1)[1].split("</header>", 1)[0]
    assert 'id="progress-phase"' in card
    assert 'getElementById("progress-phase")' in js or '$("#progress-phase")' in js
    assert r"Phase\s+\d+" in js


def test_dismissal_sticks_until_a_new_task_starts():
    js = _read("app.js")
    handler = js.split('$("#btn-dismiss-progress").onclick', 1)[1].split("};", 1)[0]
    assert "dismissedRun" in handler
    body = js.split("function renderProgress", 1)[1].split("\n  }\n", 1)[0]
    assert "dismissedRun" in body and "p.run" in body
    assert "state.dismissedRun = null" in body
