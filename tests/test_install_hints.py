"""C1: install hints never print a bare pip and never undermine a working GPU."""
import sys
from pathlib import Path

import pytest

from photocure.inference import backends as bk


def test_in_venv_hint_uses_that_venvs_pip():
    cmd = bk.ml_install_cmd()
    assert cmd.startswith(str(Path(sys.prefix) / "bin" / "pip") + " install ")
    assert not cmd.startswith("pip ")


@pytest.mark.real_venv_detection
def test_detection_matches_interpreter_state():
    assert bk.in_venv() == (sys.prefix != sys.base_prefix)


def test_outside_a_venv_points_to_setup(monkeypatch):
    monkeypatch.setattr(bk, "in_venv", lambda: False)
    assert bk.ml_install_cmd() == "photocure setup --reinstall"
    hints = [p["install_hint"] for p in bk.list_accelerators(torch=None)]
    assert not any("pip install" in h for h in hints)


def _probe(acc, available):
    return {"id": acc, "available": available, "reason": "" if available else "no",
            "install_hint": f"hint-{acc}"}


def test_working_gpu_suppresses_hints_for_other_torch_builds(monkeypatch):
    monkeypatch.setattr(bk, "_probe_one",
                        lambda acc, torch, index=0: _probe(acc, acc in ("cuda", "cpu")))
    got = {p["id"]: p["install_hint"] for p in bk.list_accelerators(torch=object())}
    assert got["rocm"] == got["xpu"] == got["directml"] == ""


def test_hints_remain_when_no_gpu_works(monkeypatch):
    monkeypatch.setattr(bk, "_probe_one", lambda acc, torch, index=0: _probe(acc, acc == "cpu"))
    got = {p["id"]: p["install_hint"] for p in bk.list_accelerators(torch=object())}
    assert got["cuda"] == "hint-cuda"
