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
    return (r["scan"], r["infer"])


def test_empty_library_starts_on_scan():
    assert _states(_flow(0, 0, 0, None)) == ("active", "locked")


def test_after_scan_new_photos_make_analyze_the_long_button():
    r = _flow(600, 40, 3, "560")
    assert r["next"] == "infer" and _states(r) == ("done", "active")


def test_unfinished_grouping_keeps_analyze_active_because_analyze_also_groups():
    r = _flow(600, 0, 3, "560")
    assert r["next"] == "infer" and _states(r) == ("done", "active")


def test_after_analyze_the_cycle_returns_to_scan():
    r = _flow(600, 0, 3, "600")
    assert r["next"] == "scan" and _states(r) == ("active", "done")


def test_second_import_round_starts_on_scan_again():
    assert _flow(600, 0, 0, "600")["next"] == "scan"


def test_legacy_library_without_a_stored_total():
    assert _flow(600, 0, 5, None)["next"] == "scan"
    assert _flow(600, 0, 0, None)["next"] == "infer"


def test_there_is_no_separate_group_step():
    assert "cluster" not in _flow(600, 0, 3, "600")
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="step-cluster"' not in html and 'id="btn-cluster"' not in html


def test_analyze_infers_then_groups_and_remembers_the_grouped_total():
    js = _js()
    assert "function markGrouped(" in js and "pcur-grouped-total" in js
    body = js.split('$("#btn-infer").onclick', 1)[1].split("\n    };", 1)[0]
    assert body.index('"/api/infer"') < body.index('"/api/cluster"'), "infer first, then group"
    assert "markGrouped()" in body
    ingest = js.split('$("#btn-ingest-all").onclick', 1)[1].split("};", 1)[0]
    assert "markGrouped()" in ingest


def _hint(flow, total):
    src = re.search(r"function nextHint\([^)]*\)\s*\{.*?\n  \}", _js(), re.S)
    assert src, "nextHint not found in app.js"
    call = f"nextHint({json.dumps(flow)}, {total})"
    return subprocess.run(["node", "-e", f"{src.group(0)}\nconsole.log({call})"],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_hint_tells_a_new_user_to_scan():
    assert "Scan" in _hint({"next": "scan"}, 0)


def test_hint_after_scan_says_analyze_and_that_it_groups_and_sorts():
    h = _hint({"next": "infer"}, 600).lower()
    assert "analyze" in h and "group" in h and "sort" in h


def test_hint_when_everything_is_done_invites_the_next_batch():
    assert "Scan" in _hint({"next": "scan"}, 600)




def test_analyze_also_sorts_so_the_dock_has_no_sort_button():
    html, js = (STATIC / "index.html").read_text(encoding="utf-8"), _js()
    assert 'id="btn-sort-dock"' not in html and "needs-action" not in js
    body = js.split('$("#btn-infer").onclick', 1)[1].split("\n    };", 1)[0]
    assert "afterTask(true)" in body, "Analyze must end by sorting into categories"
    assert "sorted" in body.lower()
