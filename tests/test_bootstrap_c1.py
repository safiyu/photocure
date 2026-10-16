"""C1 acceptance: first-launch bootstrap, upgrade mode, wrapper routing, /healthz while provisioning."""
import ast
import importlib.util
import json
import os
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BOOT = ROOT / "packaging" / "deb" / "bootstrap.py"
WRAPPER = ROOT / "packaging" / "deb" / "photocure-wrapper"


@pytest.fixture(scope="module")
def bs():
    spec = importlib.util.spec_from_file_location("pcur_bootstrap", BOOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bootstrap_uses_only_the_standard_library():
    tree = ast.parse(BOOT.read_text())
    mods = {(n.names[0].name if isinstance(n, ast.Import) else n.module).split(".")[0]
            for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))}
    assert mods <= set(sys.stdlib_module_names)


def test_healthz_is_503_while_provisioning_and_500_on_failure(bs):
    p = bs.Progress()
    server, port = bs.serve_progress("127.0.0.1", 0, p)
    try:
        p.step = "Installing"
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz")
        assert e.value.code == 503
        body = json.loads(e.value.read())
        assert body == {"status": "provisioning", "step": "Installing"}
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "location.reload()" in page and "r.status === 200" in page
        p.error = "boom"
        with pytest.raises(urllib.error.HTTPError) as e2:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz")
        assert e2.value.code == 500 and json.loads(e2.value.read())["error"] == "boom"
    finally:
        server.shutdown()
        server.server_close()


def test_upgrade_keeps_the_venv_and_only_upgrades_the_wheel(bs, tmp_path):
    venv = tmp_path / "venv"
    (venv / "lib").mkdir(parents=True)
    (venv / "lib" / "torch-weights.bin").write_text("keep me")
    (venv / ".photocure_version").write_text("1.0\n")
    calls = []
    bs.provision(venv, tmp_path / "photocure-2.0-py3-none-any.whl", "2.0", bs.Progress(),
                 upgrade=True, run=lambda cmd, check: calls.append(cmd))
    assert len(calls) == 1 and "venv" not in calls[0][1:3] and "--upgrade" in calls[0]
    assert (venv / "lib" / "torch-weights.bin").read_text() == "keep me"
    assert (venv / ".photocure_version").read_text().strip() == "2.0"


def test_first_run_builds_then_installs_then_marks(bs, tmp_path):
    venv = tmp_path / "data" / "venv"
    order = []

    def run(cmd, check):
        order.append("venv" if cmd[1:3] == ["-m", "venv"] else "pip")
        if order[-1] == "venv":
            venv.mkdir(parents=True)
        assert not (venv / ".photocure_version").exists()  # marker only after pip

    bs.provision(venv, tmp_path / "w.whl", "1.0", bs.Progress(), run=run)
    assert order == ["venv", "pip"] and (venv / ".photocure_version").read_text().strip() == "1.0"


def test_failed_install_leaves_no_marker_and_next_run_starts_clean(bs, tmp_path):
    venv = tmp_path / "venv"

    def broken(cmd, check):
        if cmd[1:3] == ["-m", "venv"]:
            venv.mkdir(parents=True)
        else:
            raise subprocess.CalledProcessError(1, cmd)

    with pytest.raises(subprocess.CalledProcessError):
        bs.provision(venv, tmp_path / "w.whl", "1.0", bs.Progress(), run=broken)
    assert not (venv / ".photocure_version").exists()
    (venv / "half-installed").write_text("x")
    seen = []
    bs.provision(venv, tmp_path / "w.whl", "1.0", bs.Progress(),
                 run=lambda cmd, check: (seen.append(cmd), venv.mkdir(exist_ok=True)))
    assert not (venv / "half-installed").exists()


def test_port_and_command_helpers(bs, tmp_path):
    assert bs.requested_port(["start", "--port", "9000"]) == 9000
    assert bs.requested_port(["start", "-p", "9001"]) == 9001
    assert bs.requested_port(["start", "--port=9002"]) == 9002
    assert bs.with_port(["start", "--port", "1", "--no-open"], 7) == ["start", "--no-open", "--port", "7"]
    cmd = bs.exec_args(tmp_path, ["start", "--open"], 8081)
    assert cmd[1:] == ["start", "--port", "8081", "--no-open"]
    assert bs.exec_args(tmp_path, ["stats"], None)[1:] == ["stats"]


def test_version_and_stop_need_no_environment(bs, tmp_path, monkeypatch, capsys):
    app = tmp_path / "opt"
    app.mkdir()
    (app / "VERSION").write_text("1.2.3\n")
    monkeypatch.setattr(bs, "APP_DIR", app)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    assert bs.main(["--version"]) == 0
    assert "1.2.3" in capsys.readouterr().out
    assert bs.main(["stop", "--port", "8765"]) == 0
    assert "No active Photocure server found on port 8765." in capsys.readouterr().out


# ------------------------------------------------------------------ wrapper
def _wrapper(tmp_path):
    opt = tmp_path / "opt"
    opt.mkdir(exist_ok=True)
    fake_boot = opt / "bootstrap.py"
    fake_boot.write_text("import sys; print('BOOTSTRAP', *sys.argv[1:])\n")
    script = tmp_path / "photocure"
    script.write_text(WRAPPER.read_text().replace("/opt/photocure", str(opt)))
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script, opt


def _run(script, tmp_path, *args, xdg=None):
    env = dict(os.environ, XDG_DATA_HOME=str(xdg or tmp_path / "data"), HOME=str(tmp_path))
    return subprocess.run(["sh", str(script), *args], capture_output=True, text=True, env=env)


def _venv(tmp_path, version):
    v = tmp_path / "data" / "photocure" / "venv"
    (v / "bin").mkdir(parents=True)
    cli = v / "bin" / "photocure"
    cli.write_text("#!/bin/sh\necho CLI \"$@\"\n")
    cli.chmod(0o755)
    (v / ".photocure_version").write_text(version + "\n")


def test_wrapper_first_run_goes_to_bootstrap(tmp_path):
    script, _ = _wrapper(tmp_path)
    r = _run(script, tmp_path, "start")
    assert r.stdout.strip() == "BOOTSTRAP start"


def test_wrapper_unfinished_install_goes_to_bootstrap(tmp_path):
    script, opt = _wrapper(tmp_path)
    (opt / "VERSION").write_text("1.0\n")
    _venv(tmp_path, "1.0")
    (tmp_path / "data" / "photocure" / "venv" / ".photocure_version").unlink()
    assert _run(script, tmp_path, "start").stdout.startswith("BOOTSTRAP")


def test_wrapper_upgrade_when_versions_differ(tmp_path):
    script, opt = _wrapper(tmp_path)
    (opt / "VERSION").write_text("2.0\n")
    _venv(tmp_path, "1.0")
    assert _run(script, tmp_path, "start").stdout.strip() == "BOOTSTRAP --upgrade start"


def test_wrapper_runs_the_venv_when_current(tmp_path):
    script, opt = _wrapper(tmp_path)
    (opt / "VERSION").write_text("1.0\n")
    _venv(tmp_path, "1.0")
    assert _run(script, tmp_path, "stats").stdout.strip() == "CLI stats"


def test_wrapper_refuses_a_system_path(tmp_path):
    script, _ = _wrapper(tmp_path)
    r = _run(script, tmp_path, "start", xdg="/usr/share")
    assert r.returncode == 1 and "system path" in r.stderr
