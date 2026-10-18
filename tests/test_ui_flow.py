"""Which import step is the long curved button: Scan, then Analyze, then Group, then Scan again."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parent.parent / "photocure" / "ui" / "static"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _js() -> str:
    return (STATIC / "app.js").read_text(encoding="utf-8")


def _flow(total, pending, bursts, stored):
    src = re.search(r"function flowStates\([^)]*\)\s*\{.*?\n  \}", _js(), re.S)
    assert src, "flowStates not found in app.js"
    call = f"flowStates({total}, {pending}, {bursts}, {json.dumps(stored)})"
    out = subprocess.run(["node", "-e", f"{src.group(0)}\nconsole.log(JSON.stringify({call}))"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _states(r):
    return (r["scan"], r["infer"], r["cluster"])


def test_empty_library_starts_on_scan():
    assert _states(_flow(0, 0, 0, None)) == ("active", "locked", "locked")


def test_after_scan_new_photos_make_analyze_the_long_button():
    r = _flow(600, 40, 3, "560")
    assert r["next"] == "infer" and _states(r) == ("done", "active", "locked")


def test_after_analyze_group_is_the_long_button_and_is_its_own_step():
    r = _flow(600, 0, 3, "560")
    assert r["next"] == "cluster" and _states(r) == ("done", "done", "active")


def test_after_group_the_cycle_returns_to_scan():
    r = _flow(600, 0, 3, "600")
    assert r["next"] == "scan" and _states(r) == ("active", "done", "done")


def test_second_import_round_starts_on_scan_again():
    assert _flow(600, 0, 0, "600")["next"] == "scan"


def test_legacy_library_without_a_stored_total():
    assert _flow(600, 0, 5, None)["next"] == "scan"
    assert _flow(600, 0, 0, None)["next"] == "cluster"


def test_group_handlers_remember_the_grouped_total():
    js = _js()
    assert "pcur-grouped-total" in js and "function markGrouped(" in js
    for handler in ('$("#btn-cluster").onclick', '$("#btn-ingest-all").onclick'):
        body = js.split(handler, 1)[1].split("};", 1)[0]
        assert "markGrouped()" in body, handler


def test_analyze_does_not_start_grouping():
    body = _js().split('$("#btn-infer").onclick', 1)[1].split("};", 1)[0]
    assert "/api/cluster" not in body and "btn-cluster" not in body


def _hint(flow, sort_needed, total):
    src = re.search(r"function nextHint\([^)]*\)\s*\{.*?\n  \}", _js(), re.S)
    assert src, "nextHint not found in app.js"
    call = f"nextHint({json.dumps(flow)}, {json.dumps(sort_needed)}, {total})"
    return subprocess.run(["node", "-e", f"{src.group(0)}\nconsole.log({call})"],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_hint_tells_a_new_user_to_scan():
    assert "Scan" in _hint({"next": "scan"}, False, 0)


def test_hint_after_scan_says_analyze():
    assert "Analyze" in _hint({"next": "infer"}, False, 600)


def test_hint_names_sort_photos_when_photos_wait_to_be_sorted():
    assert "Sort Photos" in _hint({"next": "scan"}, True, 600)
    both = _hint({"next": "cluster"}, True, 600)
    assert "Sort Photos" in both and "Group" in both


def test_hint_after_analyze_says_group():
    h = _hint({"next": "cluster"}, False, 600)
    assert "Group" in h and "Sort Photos" not in h


def test_hint_when_everything_is_done_invites_the_next_batch():
    assert "Scan" in _hint({"next": "scan"}, False, 600)


def test_sort_photos_is_highlighted_when_relevant():
    js, html = _js(), (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="dock-hint"' in html.split('class="dock"', 1)[1].split("</section>", 1)[0]
    body = js.split("function renderCounts", 1)[1].split("\n  }\n", 1)[0]
    assert 'classList.toggle("needs-action", sortNeeded)' in body
    assert 'by_category' in body and "unrouted" in body
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert re.search(r"\.btn\.needs-action\s*\{", css) and re.search(r"\.btn\.needs-action:hover\s*\{", css)
