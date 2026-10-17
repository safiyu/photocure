"""C10: Import layout rework (hero gauge, blob cards, dock, sidebar) static assertions."""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATIC = ROOT / "photocure" / "ui" / "static"


def _css() -> str:
    return (STATIC / "style.css").read_text(encoding="utf-8")


def _html() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def _js() -> str:
    return (STATIC / "app.js").read_text(encoding="utf-8")


def _step(html: str, step_id: str) -> str:
    m = re.search(rf'<li class="step" id="{step_id}">(.*?)</li>', html, re.DOTALL)
    assert m, f"{step_id} <li> not found"
    return m.group(1)


# ------------------------------------------------------------------ triple chevrons

def test_step_flow_has_three_chevrons():
    html = _html()
    for step_id in ("step-scan", "step-infer"):
        assert _step(html, step_id).count('class="chev"') == 3, f"{step_id} needs 3 chevrons"
    rule = re.search(r"\.step-flow\s+\.chev:nth-child\(3\)\s*\{([^}]*)\}", _css())
    assert rule, ".step-flow .chev:nth-child(3) rule missing"
    assert ".7s" in rule.group(1), "third chevron must be delayed .7s to extend the stagger"
