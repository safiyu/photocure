"""Runs the real-browser scenarios (tests/browser/scenarios.mjs) against a seeded server.

Needs Node 22+ and Chrome or Chromium; skipped otherwise. Set PCUR_SKIP_BROWSER=1 to skip on purpose
and PCUR_CHROME to point at a browser.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BROWSER = ROOT / "tests" / "browser"
CHROME_CANDIDATES = [
    os.environ.get("PCUR_CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/opt/brave.com/brave/brave", "/opt/brave.com/brave/brave-browser", "/usr/bin/brave-browser",
]

pytestmark = pytest.mark.browser


def _skip_reason() -> str | None:
    if os.environ.get("PCUR_SKIP_BROWSER"):
        return "PCUR_SKIP_BROWSER is set"
    node = shutil.which("node")
    if not node:
        return "node is not installed"
    ver = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v")
    if not ver or int(ver.split(".")[0]) < 22:
        return f"node {ver or '?'} is too old (needs 22+ for a built-in WebSocket)"
    if not any(c and Path(c).exists() for c in CHROME_CANDIDATES):
        return "no Chrome or Chromium found"
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def outcome(tmp_path_factory):
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    work = tmp_path_factory.mktemp("browser")
    port = _free_port()
    log = (work / "server.log").open("w")
    server = subprocess.Popen([sys.executable, str(BROWSER / "serve.py"), "--dir", str(work), "--port", str(port),
                               "--n", "5000"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if server.poll() is not None:
                pytest.fail("the seeded server exited:\n" + (work / "server.log").read_text()[-1500:])
            if (work / "meta.json").exists():
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
                    break
                except OSError:
                    pass
            time.sleep(0.5)
        else:
            pytest.fail("the seeded server did not come up")
        results, shots = work / "results.json", work / "shots"
        run = subprocess.run(["node", str(BROWSER / "scenarios.mjs"), str(work / "meta.json"), str(results), str(shots)],
                             cwd=ROOT, capture_output=True, text=True, timeout=600)
        data = json.loads(results.read_text()) if results.exists() else []
        return {"results": data, "stdout": run.stdout, "stderr": run.stderr[-1500:], "shots": shots}
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()


def test_every_browser_scenario_passes(outcome):
    failed = [r for r in outcome["results"] if not r["ok"]]
    assert outcome["results"], "the scenarios produced no results:\n" + outcome["stderr"]
    assert not failed, "\n".join(f"{r['name']}: {r.get('error', '')}" for r in failed)


def test_the_expected_scenarios_all_ran(outcome):
    names = {r["name"] for r in outcome["results"]}
    for needle in (
        "one-time code", "sign-in box", "DOM stays within 100 cards", "recycled",
        "arrow keys move", "keys rate", "XMP sidecar", "edited elsewhere", "page boundary",
        "filters agree", "date scrubber", "resize", "WebP", "Reject all blinks", "Inspect faces",
        "category label", "lightbox focus restoration", "in-app prompt dialog",
        "persistent reset-filters", "version badge button", "touch swipe navigation",
        "date presets", "roving focus", "picker and help modals", "no script errors",
        "rapid tab switching", "reduced motion closes", "reopening a dialog that is closing",
        "import flow: the next step", "keyboard focus shows a visible ring", "clicking the version opens Settings", "empty states sit near the vertical middle", "step chevrons are centered", "a dismissed progress card stays dismissed", "category dropdown is alphabetical", "toolbar and sliders stay put",
    ):
        assert any(needle in n for n in names), f"no scenario mentions {needle!r}: {sorted(names)}"
