"""photocure setup: hardware detection, install plan, model check, serving."""
from __future__ import annotations

import json
import subprocess
import types

import pytest
from typer.testing import CliRunner

from photocure import cli as cli_mod
from photocure import setup_cmd as sc


def which_of(*names):
    return lambda n: f"/usr/bin/{n}" if n in names else None


# ------------------------------------------------------------------ detection
@pytest.mark.parametrize("tools,system,machine,expect", [
    (("nvidia-smi",), "Linux", "x86_64", "cuda"),
    (("nvidia-smi", "rocm-smi"), "Linux", "x86_64", "cuda"),
    (("rocm-smi",), "Linux", "x86_64", "rocm"),
    (("rocminfo",), "Windows", "AMD64", "cpu"),          # ROCm wheels are Linux only
    (("sycl-ls",), "Linux", "x86_64", "xpu"),
    ((), "Darwin", "arm64", "mps"),
    ((), "Darwin", "x86_64", "cpu"),
    ((), "Linux", "x86_64", "cpu"),
])
def test_detect_kind(tools, system, machine, expect):
    assert sc.detect_kind(which_of(*tools), system, machine,
                          dev_root="/nonexistent", sys_root="/nonexistent") == expect


# ---------------------------------------------------------------------- plan
FRESH = {"torch": False, "open_clip": None, "insightface": False}
READY = {"torch": True, "version": "2.5", "cuda_build": True, "hip_build": False, "has_xpu": False,
         "open_clip": "3.0", "insightface": True}


def titles(steps):
    return [s.title for s in steps]


def test_fresh_machine_installs_everything_in_order():
    steps = sc.plan_steps("cuda", FRESH, python="py", system="Linux")
    assert len(steps) == 3
    assert steps[0].cmd[:4] == ["py", "-m", "pip", "install"] and "torch" in steps[0].cmd
    assert steps[0].cmd[-2:] == ["--index-url", sc.TORCH_INDEX["cuda"]]
    assert "open-clip-torch>=2.20" in steps[1].cmd           # after torch, so torch is kept
    assert "onnxruntime-gpu" in steps[2].cmd and steps[2].optional


@pytest.mark.parametrize("kind,index", [("rocm", "rocm"), ("xpu", "xpu"), ("cpu", "cpu")])
def test_index_per_hardware_on_linux(kind, index):
    cmd = sc.plan_steps(kind, FRESH, python="py", system="Linux")[0].cmd
    assert cmd[-1] == sc.TORCH_INDEX[index]


def test_apple_uses_default_wheel_without_index():
    for kind in ("mps", "cpu"):
        cmd = sc.plan_steps(kind, FRESH, python="py", system="Darwin")[0].cmd
        assert "--index-url" not in cmd


def test_nothing_to_do_when_everything_present():
    assert sc.plan_steps("cuda", READY, python="py") == []


def test_cpu_torch_on_nvidia_machine_is_replaced():
    cpu_build = dict(READY, cuda_build=False)
    steps = sc.plan_steps("cuda", cpu_build, python="py", system="Linux")
    assert len(steps) == 1 and "--force-reinstall" in steps[0].cmd
    assert "wrong build" in steps[0].note


def test_cuda_torch_without_a_working_driver_is_not_reinstalled():
    # CUDA build but no usable GPU: reinstalling would not help.
    assert sc.plan_steps("cuda", dict(READY, cuda_ok=False), python="py") == []


def test_force_reinstalls_and_no_faces_skips_faces():
    steps = sc.plan_steps("mps", dict(READY, insightface=False), faces=False, force=True,
                          python="py", system="Darwin")
    assert titles(steps) == [steps[0].title] and "--force-reinstall" in steps[0].cmd


def test_total_download_includes_weights_only_when_missing():
    steps = sc.plan_steps("mps", FRESH, python="py", system="Darwin")
    base = sum(s.size_mb for s in steps)
    assert sc.total_download_mb(steps, weights_cached=True) == base
    assert sc.total_download_mb(steps, weights_cached=False) == base + sc.CLIP_WEIGHTS_MB


# --------------------------------------------------------- child process probes
def fake_run(stdout="", returncode=0, stderr=""):
    return lambda *a, **k: types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_probe_state_reads_last_json_line_and_survives_garbage():
    st = sc.probe_state(fake_run('warning text\n{"torch": true, "version": "2.1"}\n'))
    assert st["torch"] is True and st["version"] == "2.1"
    assert sc.probe_state(fake_run("nothing useful"))["torch"] is False

    def boom(*a, **k):
        raise OSError("no interpreter")
    assert sc.probe_state(boom)["torch"] is False


def test_verify_model_success_failure_and_stub_env_is_dropped(monkeypatch):
    seen = {}

    def runner(cmd, **kw):
        seen["env"] = kw["env"]
        return types.SimpleNamespace(stdout=json.dumps({"ok": True, "device": "mps"}), stderr="", returncode=0)

    monkeypatch.setenv("PCUR_BACKEND", "stub")
    assert sc.verify_model(runner)["ok"] is True
    assert "PCUR_BACKEND" not in seen["env"]
    bad = sc.verify_model(fake_run("", 1, "line1\nOSError: no network"))
    assert bad["ok"] is False and "no network" in bad["error"]


# ------------------------------------------------------------------- serving
def test_pick_port_skips_a_busy_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert sc.pick_port("127.0.0.1", busy) != busy


def test_browser_opens_once_the_server_answers():
    calls = {"n": 0, "opened": None}

    def fetch(url, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("not up yet")

    t = sc.open_browser_when_ready("0.0.0.0", 8123, timeout=10, opener=lambda u: calls.update(opened=u),
                                   fetch=fetch)
    t.join(10)
    assert calls["opened"] == "http://127.0.0.1:8123/" and calls["n"] == 3


# ------------------------------------------------------------------------ CLI
@pytest.fixture()
def cli_env(monkeypatch, tmp_path):
    ran, served = [], {}
    monkeypatch.setenv("PCUR_CACHE_DIR", str(tmp_path / "cache"))  # never touch the repo cache
    # setup edits os.environ itself, so register these for restore after each test.
    monkeypatch.setenv("PCUR_BACKEND", "stub")
    monkeypatch.setenv("PCUR_DEVICE_DEVICE", "cpu")
    monkeypatch.setattr(sc, "detect_kind", lambda *a, **k: "mps")
    monkeypatch.setattr(sc, "probe_state", lambda *a, **k: dict(FRESH))
    monkeypatch.setattr(sc, "run_step", lambda step, runner=None: ran.append(step.title) or True)
    monkeypatch.setattr(sc, "verify_model", lambda *a, **k: {"ok": True, "device": "mps", "model": "ViT-B-16", "seconds": 1.0})
    monkeypatch.setattr(sc, "prefetch_faces", lambda *a, **k: True)
    monkeypatch.setattr(sc, "open_browser_when_ready", lambda *a, **k: served.setdefault("opened", a))
    monkeypatch.setattr(cli_mod, "serve", lambda **kw: served.setdefault("serve", kw))
    return types.SimpleNamespace(ran=ran, served=served, db=tmp_path / "x.db")


def test_setup_installs_verifies_and_opens_the_ui(cli_env):
    r = CliRunner().invoke(cli_mod.app, ["setup", "--yes", "--db", str(cli_env.db), "--port", "8099"])
    assert r.exit_code == 0, r.output
    assert len(cli_env.ran) == 3
    assert "Ready." in r.output and "mps" in r.output
    assert cli_env.served["serve"]["host"] == "127.0.0.1"
    assert cli_env.served["opened"][0] == "127.0.0.1"


def test_setup_dry_run_installs_nothing(cli_env):
    r = CliRunner().invoke(cli_mod.app, ["setup", "--dry-run", "--db", str(cli_env.db)])
    assert r.exit_code == 0 and cli_env.ran == [] and "serve" not in cli_env.served
    assert "Dry run" in r.output


def test_setup_no_serve_and_no_open(cli_env):
    r = CliRunner().invoke(cli_mod.app, ["setup", "-y", "--no-serve", "--db", str(cli_env.db)])
    assert r.exit_code == 0 and "serve" not in cli_env.served and "opened" not in cli_env.served
    r = CliRunner().invoke(cli_mod.app, ["setup", "-y", "--no-open", "--db", str(cli_env.db)])
    assert r.exit_code == 0 and "serve" in cli_env.served and "opened" not in cli_env.served


def test_setup_stops_when_a_required_step_fails(cli_env, monkeypatch):
    monkeypatch.setattr(sc, "run_step", lambda step, runner=None: False)
    r = CliRunner().invoke(cli_mod.app, ["setup", "-y", "--db", str(cli_env.db)])
    assert r.exit_code == 1 and "serve" not in cli_env.served and "Failed" in r.output


def test_setup_continues_when_faces_fail(cli_env, monkeypatch):
    monkeypatch.setattr(sc, "run_step", lambda step, runner=None: not step.optional)
    r = CliRunner().invoke(cli_mod.app, ["setup", "-y", "--db", str(cli_env.db)])
    assert r.exit_code == 0 and "OpenCV" in r.output and "serve" in cli_env.served


def test_setup_stops_when_the_model_does_not_load(cli_env, monkeypatch):
    monkeypatch.setattr(sc, "verify_model", lambda *a, **k: {"ok": False, "error": "no network"})
    r = CliRunner().invoke(cli_mod.app, ["setup", "-y", "--db", str(cli_env.db)])
    assert r.exit_code == 1 and "no network" in r.output and "serve" not in cli_env.served


def test_setup_asks_before_downloading(cli_env):
    r = CliRunner().invoke(cli_mod.app, ["setup", "--db", str(cli_env.db)], input="n\n")
    assert r.exit_code == 1 and cli_env.ran == []


def test_cpu_only_ignores_the_gpu(cli_env, monkeypatch):
    monkeypatch.setattr(sc, "detect_kind", lambda *a, **k: pytest.fail("should not detect"))
    r = CliRunner().invoke(cli_mod.app, ["setup", "--dry-run", "--cpu-only", "--db", str(cli_env.db)])
    assert r.exit_code == 0 and "CPU only" in r.output


# ------------------------------------------------------- containers and Docker
def make_dev(tmp_path, nodes=(), render=None):
    dev, sysd = tmp_path / "dev", tmp_path / "sys"
    dev.mkdir(exist_ok=True); sysd.mkdir(exist_ok=True)
    for n in nodes:
        (dev / n).parent.mkdir(parents=True, exist_ok=True)
        (dev / n).write_text("")
    for node, vendor in (render or {}).items():
        (dev / "dri").mkdir(exist_ok=True)
        (dev / "dri" / node).write_text("")
        d = sysd / "class" / "drm" / node / "device"
        d.mkdir(parents=True, exist_ok=True)
        (d / "vendor").write_text(vendor + "\n")
    return str(dev), str(sysd)


def kind_in(tmp_path, nodes=(), render=None, env=None, system="Linux"):
    dev, sysd = make_dev(tmp_path, nodes, render)
    return sc.detect_kind(which_of(), system, "x86_64", dev_root=dev, sys_root=sysd, env=env or {})


def test_container_detects_nvidia_from_device_nodes(tmp_path):
    assert kind_in(tmp_path, nodes=["nvidiactl", "nvidia0"]) == "cuda"


def test_container_detects_amd_from_kfd(tmp_path):
    assert kind_in(tmp_path, nodes=["kfd"], render={"renderD128": "0x1002"}) == "rocm"


def test_container_detects_intel_from_render_vendor(tmp_path):
    assert kind_in(tmp_path, render={"renderD128": "0x8086"}) == "xpu"


def test_amd_render_node_without_kfd_falls_back_to_cpu(tmp_path):
    # An AMD GPU with no compute device cannot run ROCm, so use the CPU build.
    assert kind_in(tmp_path, render={"renderD128": "0x1002"}) == "cpu"


def test_nothing_visible_means_cpu(tmp_path):
    assert kind_in(tmp_path) == "cpu"


def test_accel_env_overrides_detection(tmp_path):
    for sub in ("a", "b", "c"):
        (tmp_path / sub).mkdir()
    assert kind_in(tmp_path / "a", nodes=["nvidiactl"], env={"PCUR_ACCEL": "cpu"}) == "cpu"
    assert kind_in(tmp_path / "b", env={"PCUR_ACCEL": "xpu"}) == "xpu"
    assert kind_in(tmp_path / "c", env={"PCUR_ACCEL": "nonsense"}) == "cpu"


def test_device_nodes_only_count_on_linux(tmp_path):
    assert kind_in(tmp_path, nodes=["nvidiactl"], system="Darwin") == "cpu"


def test_user_install_flag_adds_user_and_uses_the_volume(monkeypatch):
    steps = sc.plan_steps("cpu", FRESH, python="py", system="Linux", user=True)
    assert all("--user" in s.cmd for s in steps)
    assert "--user" not in sc.plan_steps("cpu", FRESH, python="py", system="Linux", user=False)[0].cmd
    monkeypatch.setenv("PYTHONUSERBASE", "/data/pylibs")
    monkeypatch.setattr(sc.sys, "prefix", "/usr", raising=False)
    monkeypatch.setattr(sc.sys, "base_prefix", "/usr", raising=False)
    assert sc.user_install_wanted() is True
    monkeypatch.setattr(sc.sys, "prefix", "/venv", raising=False)
    assert sc.user_install_wanted() is False        # never --user inside a virtualenv


def test_torch_index_override_for_older_drivers(monkeypatch):
    monkeypatch.setenv("PCUR_TORCH_INDEX", "https://download.pytorch.org/whl/cu118")
    cmd = sc.plan_steps("cuda", FRESH, python="py", system="Linux")[0].cmd
    assert cmd[-1].endswith("cu118")


def test_faces_step_tidies_up_the_duplicate_opencv():
    step = sc.plan_steps("cpu", FRESH, python="py", system="Linux")[-1]
    assert step.optional and any("uninstall" in c for c in step.after)
    assert any("opencv-python-headless" in c for c in step.after)
    ran = []
    ok = sc.run_step(step, runner=lambda cmd, **k: ran.append(cmd) or types.SimpleNamespace(returncode=0))
    assert ok and len(ran) == 1 + len(step.after)
    ran.clear()
    bad = sc.run_step(step, runner=lambda cmd, **k: ran.append(cmd) or types.SimpleNamespace(returncode=1))
    assert not bad and len(ran) == 1                  # tidy-up only after a good install


def test_verify_script_runs_for_real_with_the_stub_backend():
    """Executes the model check code itself, so API misuse cannot hide behind mocks."""
    import os
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-c", sc._VERIFY], capture_output=True, text=True,
                       env={**os.environ, "PCUR_BACKEND": "stub"}, timeout=120)
    assert r.returncode == 0, r.stderr[-400:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["backend"] == "stub" and out["ok"] is False and "device" in out


def test_probe_script_runs_for_real():
    st = sc.probe_state()
    assert "torch" in st and "open_clip" in st


def test_setup_skips_faces_after_an_earlier_failure(cli_env, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "setup_faces_failed").write_text("failed\n")
    monkeypatch.setenv("PCUR_CACHE_DIR", str(cache))
    r = CliRunner().invoke(cli_mod.app, ["setup", "--dry-run", "--db", str(cli_env.db)])
    assert r.exit_code == 0 and "InsightFace" not in r.output.replace("Skipping", "")
    assert "failed to install earlier" in r.output


def test_setup_remembers_a_failed_face_install(cli_env, monkeypatch, tmp_path):
    cache = tmp_path / "cache2"
    monkeypatch.setenv("PCUR_CACHE_DIR", str(cache))
    monkeypatch.setattr(sc, "run_step", lambda step, runner=None: not step.optional)
    r = CliRunner().invoke(cli_mod.app, ["setup", "-y", "--db", str(cli_env.db)])
    assert r.exit_code == 0 and (cache / "setup_faces_failed").exists()


def test_restart_skips_the_slow_model_check(cli_env, monkeypatch):
    monkeypatch.setattr(sc, "probe_state", lambda *a, **k: dict(READY))
    monkeypatch.setattr(cli_mod, "_clip_weights_cached", lambda cfg: True)
    monkeypatch.setattr(sc, "verify_model", lambda *a, **k: pytest.fail("should not verify"))
    r = CliRunner().invoke(cli_mod.app, ["setup", "-y", "--db", str(cli_env.db)])
    assert r.exit_code == 0 and "installed and cached" in r.output and "serve" in cli_env.served
    forced = []
    monkeypatch.setattr(sc, "verify_model", lambda *a, **k: forced.append(1) or {"ok": True, "device": "cpu", "model": "m", "seconds": 1})
    CliRunner().invoke(cli_mod.app, ["setup", "-y", "--verify", "--db", str(cli_env.db)])
    assert forced


def test_entrypoint_script_is_valid_shell():
    import pathlib
    import subprocess

    script = pathlib.Path(__file__).resolve().parent.parent / "docker" / "entrypoint.sh"
    assert subprocess.run(["sh", "-n", str(script)]).returncode == 0
    text = script.read_text()
    assert "PYTHONUSERBASE" in text and "photocure setup --yes" in text
    assert "PCUR_AUTO_SETUP" in text


# ----------------------------------------------- leftover stub settings in the shell
def test_setup_ignores_a_stub_setting_so_the_server_uses_the_model(cli_env, monkeypatch):
    import os
    seen = {}
    monkeypatch.setenv("PCUR_BACKEND", "stub")
    monkeypatch.setattr(cli_mod, "serve", lambda **kw: seen.update(backend=os.environ.get("PCUR_BACKEND")))
    r = CliRunner().invoke(cli_mod.app, ["setup", "-y", "--db", str(cli_env.db)])
    assert r.exit_code == 0 and seen["backend"] is None
    assert "turns the AI model off" in r.output and "unset PCUR_BACKEND" in r.output


def test_setup_drops_a_cpu_device_setting_on_a_gpu_machine(cli_env, monkeypatch):
    import os
    seen = {}
    monkeypatch.setenv("PCUR_DEVICE_DEVICE", "cpu")
    monkeypatch.setattr(cli_mod, "serve", lambda **kw: seen.update(dev=os.environ.get("PCUR_DEVICE_DEVICE")))
    r = CliRunner().invoke(cli_mod.app, ["setup", "-y", "--db", str(cli_env.db)])
    assert r.exit_code == 0 and seen["dev"] is None and "forces the CPU" in r.output


def test_cpu_only_keeps_the_cpu_setting(cli_env, monkeypatch):
    import os
    seen = {}
    monkeypatch.setenv("PCUR_DEVICE_DEVICE", "cpu")
    monkeypatch.setattr(cli_mod, "serve", lambda **kw: seen.update(dev=os.environ.get("PCUR_DEVICE_DEVICE")))
    r = CliRunner().invoke(cli_mod.app, ["setup", "-y", "--cpu-only", "--db", str(cli_env.db)])
    assert r.exit_code == 0 and seen["dev"] == "cpu" and "forces the CPU" not in r.output


def test_no_warning_when_nothing_is_set(cli_env, monkeypatch):
    monkeypatch.delenv("PCUR_BACKEND", raising=False)
    monkeypatch.delenv("PCUR_DEVICE_DEVICE", raising=False)
    r = CliRunner().invoke(cli_mod.app, ["setup", "-y", "--db", str(cli_env.db)])
    assert r.exit_code == 0 and "Ignoring" not in r.output
