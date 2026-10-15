"""Device choices, worker counts, the settings file, and the devices/doctor commands."""
from __future__ import annotations

import json
import types

import pytest
from typer.testing import CliRunner

from photocure import cli as cli_mod
from photocure.cli import app
from photocure.config import (
    DEVICE_CHOICES,
    MAX_WORKERS,
    CuratorConfig,
    DeviceSettings,
    auto_decode_workers,
    auto_scan_workers,
    auto_torch_threads,
    normalize_device,
    resolve_workers,
)
from photocure.services import (
    SETTINGS_KEYS,
    apply_settings,
    effective_settings,
    load_settings,
    save_settings,
    settings_path,
)

from .conftest import make_image

runner = CliRunner()


def flat(out: str) -> str:
    for ch in "│┃┏┓┗┛┡┩╇━─╭╮╰╯┳┻╋┠┨":
        out = out.replace(ch, " ")
    return " ".join(out.split())


# ------------------------------------------------------------------ config
def test_device_choices_cover_every_hardware_family():
    assert DEVICE_CHOICES == ("auto", "cuda", "rocm", "xpu", "mps", "directml",
                              "integrated", "cpu")
    assert normalize_device(" ROCm ") == "rocm"
    with pytest.raises(ValueError, match="integrated"):
        normalize_device("tpu")


@pytest.mark.parametrize("name", DEVICE_CHOICES)
def test_every_device_is_accepted_from_the_environment(monkeypatch, name):
    monkeypatch.setenv("PCUR_DEVICE_DEVICE", name.upper())
    assert DeviceSettings().device == name


def test_unknown_device_falls_back_to_auto(monkeypatch, caplog):
    monkeypatch.setenv("PCUR_DEVICE_DEVICE", "tpu")
    with caplog.at_level("WARNING"):
        assert DeviceSettings().device == "auto"
    assert "Unknown device" in caplog.text


def test_new_device_env_vars(monkeypatch):
    monkeypatch.setenv("PCUR_DEVICE_DEVICE_INDEX", "1")
    monkeypatch.setenv("PCUR_DEVICE_DECODE_WORKERS", "3")
    monkeypatch.setenv("PCUR_DEVICE_TORCH_THREADS", "5")
    d = DeviceSettings()
    assert (d.device_index, d.decode_workers, d.torch_threads) == (1, 3, 5)


def test_defaults_mean_auto():
    d = DeviceSettings.model_fields
    assert d["device_index"].default == 0
    assert d["decode_workers"].default == 0
    assert d["torch_threads"].default == 0


def test_out_of_range_workers_are_rejected(monkeypatch):
    monkeypatch.setenv("PCUR_DEVICE_DECODE_WORKERS", str(MAX_WORKERS + 1))
    with pytest.raises(ValueError):
        DeviceSettings()


def test_scan_workers_zero_resolves_to_auto_when_config_is_built(monkeypatch):
    monkeypatch.setenv("PCUR_SCAN_WORKERS", "0")
    assert CuratorConfig().scan_workers == auto_scan_workers()
    monkeypatch.setenv("PCUR_SCAN_WORKERS", "5")
    assert CuratorConfig().scan_workers == 5


@pytest.mark.parametrize("cpus,scan,decode", [(1, 2, 1), (4, 8, 4), (8, 16, 8),
                                              (16, 32, 8), (64, 32, 8)])
def test_auto_rules(cpus, scan, decode):
    assert auto_scan_workers(cpus) == scan
    assert auto_decode_workers(cpus) == decode


def test_auto_torch_threads_depends_on_device():
    assert auto_torch_threads(True, 4) == 4
    assert auto_torch_threads(True, 32) == 8
    assert auto_torch_threads(False, 32) == 2


def test_resolve_workers_auto_and_explicit():
    cfg = CuratorConfig()
    cfg.scan_workers = 0
    cfg.device.decode_workers = 0
    cfg.device.torch_threads = 0
    assert resolve_workers(cfg, "cpu", cpu_count=6) == {
        "scan_workers": 12, "decode_workers": 6, "torch_threads": 6}
    assert resolve_workers(cfg, "mps", cpu_count=6)["torch_threads"] == 2
    cfg.scan_workers, cfg.device.decode_workers, cfg.device.torch_threads = 3, 2, 7
    assert resolve_workers(cfg, "cpu", cpu_count=6) == {
        "scan_workers": 3, "decode_workers": 2, "torch_threads": 7}


def test_resolve_workers_uses_configured_device_when_none_given():
    cfg = CuratorConfig()
    cfg.device.device = "cpu"
    cfg.device.torch_threads = 0
    assert resolve_workers(cfg, cpu_count=4)["torch_threads"] == 4


# ---------------------------------------------------------------- settings
def test_settings_path_sits_next_to_the_database(tmp_path):
    assert settings_path(tmp_path / "photos.db") == tmp_path / "photos.settings.json"


def test_load_missing_or_corrupt_settings_is_empty(tmp_path):
    db = tmp_path / "a.db"
    assert load_settings(db) == {}
    settings_path(db).write_text("{not json", encoding="utf-8")
    assert load_settings(db) == {}
    settings_path(db).write_text("[1, 2]", encoding="utf-8")
    assert load_settings(db) == {}


def test_save_validates_and_merges(tmp_path):
    db = tmp_path / "a.db"
    assert save_settings(db, {"device": "MPS", "decode_workers": 4}) == {
        "device": "mps", "decode_workers": 4}
    assert save_settings(db, {"torch_threads": 2}) == {
        "device": "mps", "decode_workers": 4, "torch_threads": 2}
    assert save_settings(db, {"device": None}) == {"decode_workers": 4, "torch_threads": 2}
    assert json.loads(settings_path(db).read_text()) == {"decode_workers": 4,
                                                         "torch_threads": 2}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("bad", [
    {"device": "tpu"}, {"device": 3}, {"decode_workers": -1}, {"scan_workers": 129},
    {"torch_threads": "many"}, {"device_index": 1.5}, {"decode_workers": True},
    {"nonsense": 1},
])
def test_save_rejects_bad_values_and_stores_nothing(tmp_path, bad):
    db = tmp_path / "a.db"
    with pytest.raises(ValueError):
        save_settings(db, bad)
    assert not settings_path(db).exists()


def test_save_accepts_numeric_strings_and_zero(tmp_path):
    assert save_settings(tmp_path / "a.db", {"scan_workers": "0", "device_index": 2}) == {
        "scan_workers": 0, "device_index": 2}


def test_bad_stored_values_are_ignored_not_fatal(tmp_path):
    db = tmp_path / "a.db"
    settings_path(db).write_text(json.dumps({"device": "tpu", "decode_workers": 3,
                                             "scan_workers": 9999, "x": 1}))
    assert load_settings(db) == {"decode_workers": 3}


def test_apply_settings_returns_a_copy():
    cfg = CuratorConfig()
    cfg.device.device = "cpu"
    out = apply_settings(cfg, {"device": "rocm", "device_index": 1, "decode_workers": 3,
                               "torch_threads": 4, "scan_workers": 6})
    assert (out.device.device, out.device.device_index, out.device.decode_workers,
            out.device.torch_threads, out.scan_workers) == ("rocm", 1, 3, 4, 6)
    assert cfg.device.device == "cpu" and cfg.scan_workers != 6
    assert apply_settings(cfg, {"scan_workers": 0}).scan_workers == auto_scan_workers()
    with pytest.raises(ValueError):
        apply_settings(cfg, {"device": "tpu"})


def test_effective_settings_reports_sources_and_resolved_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_DEVICE_TORCH_THREADS", "3")
    db = tmp_path / "a.db"
    save_settings(db, {"device": "cpu", "decode_workers": 5})
    eff = effective_settings(db)
    assert eff["values"]["device"] == "cpu"
    assert eff["values"]["torch_threads"] == 3  # environment still applies
    assert eff["sources"]["device"] == "settings"
    assert eff["sources"]["torch_threads"] == "environment or default"
    assert eff["resolved"]["decode_workers"] == 5
    assert set(eff["values"]) == set(SETTINGS_KEYS)
    assert eff["choices"] == list(DEVICE_CHOICES)


def test_stored_zero_scan_workers_shows_as_auto(tmp_path):
    db = tmp_path / "a.db"
    save_settings(db, {"scan_workers": 0})
    assert effective_settings(db)["values"]["scan_workers"] == 0


def test_precedence_flag_over_settings_over_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PCUR_DEVICE_DEVICE", "cpu")
    monkeypatch.setenv("PCUR_DEVICE_DECODE_WORKERS", "2")
    db = tmp_path / "a.db"
    save_settings(db, {"decode_workers": 6})
    cfg = cli_mod._config_no_init(db)
    assert cfg.device.device == "cpu"          # environment, nothing stored
    assert cfg.device.decode_workers == 6      # settings file beats the environment
    cfg.device.decode_workers = 1              # a CLI flag is applied after, and wins
    assert cfg.device.decode_workers == 1


# --------------------------------------------------------------------- cli
@pytest.fixture()
def scanned(tmp_path):
    src = tmp_path / "photos"
    make_image(src / "a.jpg", rng_seed=1)
    db = tmp_path / "c.db"
    assert runner.invoke(app, ["scan", str(src), "--db", str(db)]).exit_code == 0
    return db


def capture_infer(monkeypatch):
    from photocure.inference import pipeline as pl
    seen: dict = {}
    real_init = pl.InferencePipeline.__init__

    def init(self, cfg=None, backend=None):
        seen["cfg"] = cfg
        real_init(self, cfg, backend=backend)

    def fake_run(self, db_path=None, only_paths=None, show_progress=True,
                 reanalyze_stub=None):
        return types.SimpleNamespace(processed=0, faces_total=0, quality_computed=0,
                                     semantic_computed=0, failures=0, duration_s=0.0)

    monkeypatch.setattr(pl.InferencePipeline, "__init__", init)
    monkeypatch.setattr(pl.InferencePipeline, "run", fake_run)
    return seen


def test_infer_flags_reach_the_config(monkeypatch, scanned):
    seen = capture_infer(monkeypatch)
    r = runner.invoke(app, ["infer", "--db", str(scanned), "--device", "cpu",
                            "--device-index", "1", "--decode-workers", "3",
                            "--threads", "5"])
    assert r.exit_code == 0, r.stdout
    d = seen["cfg"].device
    assert (d.device, d.device_index, d.decode_workers, d.torch_threads) == ("cpu", 1, 3, 5)
    assert "decode_workers=3 threads=5" in flat(r.stdout)


def test_infer_zero_workers_means_auto(monkeypatch, scanned):
    seen = capture_infer(monkeypatch)
    r = runner.invoke(app, ["infer", "--db", str(scanned), "--device", "cpu",
                            "--decode-workers", "0", "--threads", "0"])
    assert r.exit_code == 0
    d = seen["cfg"].device
    assert d.decode_workers == auto_decode_workers()
    assert d.torch_threads == auto_torch_threads(True)


def test_infer_flag_beats_settings_file(monkeypatch, scanned):
    save_settings(scanned, {"decode_workers": 6, "torch_threads": 7})
    seen = capture_infer(monkeypatch)
    runner.invoke(app, ["infer", "--db", str(scanned), "--device", "cpu"])
    assert seen["cfg"].device.decode_workers == 6   # settings file used
    runner.invoke(app, ["infer", "--db", str(scanned), "--device", "cpu",
                        "--decode-workers", "2"])
    assert seen["cfg"].device.decode_workers == 2   # flag wins


def test_infer_device_comes_from_settings_when_no_flag(monkeypatch, scanned):
    save_settings(scanned, {"device": "cpu", "device_index": 1})
    seen = capture_infer(monkeypatch)
    runner.invoke(app, ["infer", "--db", str(scanned)])
    assert seen["cfg"].device.device == "cpu" and seen["cfg"].device.device_index == 1


def test_infer_prints_the_fallback_reason(monkeypatch, scanned):
    capture_infer(monkeypatch)
    info = {"device": "cpu", "semantic_backend": "clip", "face_backend": "cv2",
            "semantic_is_stub": False, "face_is_fallback": True, "hint": "",
            "fallback_reason": "cuda: torch build has no CUDA support"}
    monkeypatch.setattr(cli_mod, "describe_backends", lambda cfg: dict(info))
    r = runner.invoke(app, ["infer", "--db", str(scanned), "--device", "cuda"])
    assert "torch build has no CUDA support" in flat(r.stdout)


@pytest.mark.parametrize("flag", ["--device-index", "--decode-workers", "--threads"])
def test_infer_rejects_out_of_range_numbers(scanned, flag):
    assert runner.invoke(app, ["infer", "--db", str(scanned), flag, "-1"]).exit_code != 0
    assert runner.invoke(app, ["infer", "--db", str(scanned), flag,
                               str(MAX_WORKERS + 1)]).exit_code != 0


def capture_scan(monkeypatch):
    from photocure.scanner import file_scanner as fs
    seen: dict = {}
    real = fs.scan_library

    def spy(target, config=None, **kw):
        seen["workers"] = config.scan_workers
        return real(target, config=config, **kw)

    monkeypatch.setattr(fs, "scan_library", spy)
    return seen


def test_scan_workers_zero_and_omitted_mean_auto(monkeypatch, tmp_path):
    src = tmp_path / "p"
    make_image(src / "a.jpg", rng_seed=1)
    seen = capture_scan(monkeypatch)
    db = tmp_path / "s.db"
    assert runner.invoke(app, ["scan", str(src), "--db", str(db), "-w", "0"]).exit_code == 0
    assert seen["workers"] == auto_scan_workers()
    assert runner.invoke(app, ["scan", str(src), "--db", str(db)]).exit_code == 0
    assert seen["workers"] == auto_scan_workers()
    assert runner.invoke(app, ["scan", str(src), "--db", str(db), "-w", "3"]).exit_code == 0
    assert seen["workers"] == 3


def test_scan_workers_from_settings_then_flag(monkeypatch, tmp_path):
    src = tmp_path / "p"
    make_image(src / "a.jpg", rng_seed=1)
    seen = capture_scan(monkeypatch)
    db = tmp_path / "s.db"
    save_settings(db, {"scan_workers": 5})
    runner.invoke(app, ["scan", str(src), "--db", str(db)])
    assert seen["workers"] == 5
    runner.invoke(app, ["scan", str(src), "--db", str(db), "-w", "2"])
    assert seen["workers"] == 2


def test_scan_rejects_too_many_workers(tmp_path):
    r = runner.invoke(app, ["scan", str(tmp_path), "--db", str(tmp_path / "s.db"),
                            "-w", str(MAX_WORKERS + 1)])
    assert r.exit_code != 0


# ------------------------------------------------- devices and doctor commands
ACCEL = [
    {"id": "cuda", "label": "NVIDIA GPU (CUDA)", "name": "RTX 4060", "kind": "discrete",
     "memory_gb": 8.0, "available": True, "reason": "", "install_hint": ""},
    {"id": "rocm", "label": "AMD GPU (ROCm)", "name": "", "kind": "discrete",
     "memory_gb": None, "available": False, "reason": "torch build has no ROCm support",
     "install_hint": "pip install torch --index-url https://download.pytorch.org/whl/rocm6.2"},
    {"id": "mps", "label": "Apple GPU (Metal)", "name": "", "kind": "integrated",
     "memory_gb": None, "available": False, "reason": "not an Apple Silicon Mac",
     "install_hint": ""},
    {"id": "cpu", "label": "CPU", "name": "x86 (8 threads)", "kind": "cpu",
     "memory_gb": None, "available": True, "reason": "", "install_hint": ""},
]
INFO = {"device": "cuda", "semantic_backend": "clip", "face_backend": "insightface",
        "semantic_is_stub": False, "face_is_fallback": False, "hint": "",
        "selected": "cuda", "requested": "auto", "fallback_reason": ""}


def patch_hw(monkeypatch, accels=ACCEL, info=INFO, hardware=()):
    from photocure.inference import backends as bk
    monkeypatch.setattr(bk, "list_accelerators", lambda torch=None, device_index=0: accels,
                        raising=False)
    monkeypatch.setattr(cli_mod, "describe_backends", lambda cfg: dict(info))
    monkeypatch.setattr(cli_mod, "_detect_hardware", lambda: list(hardware))


def test_devices_lists_every_accelerator(monkeypatch, tmp_path):
    patch_hw(monkeypatch)
    db = tmp_path / "none.db"
    r = runner.invoke(app, ["devices", "--db", str(db)])
    out = flat(r.stdout)
    assert r.exit_code == 0
    for text in ("NVIDIA GPU (CUDA)", "RTX 4060", "8.0 GB", "AMD GPU (ROCm)",
                 "torch build has no ROCm support", "whl/rocm6.2", "Apple GPU (Metal)",
                 "x86", "ready"):
        assert text in out, text
    assert "Requested device: auto, using: cuda" in out
    assert "Scan threads" in out and "Decode threads" in out and "Torch CPU threads" in out
    assert not db.exists()


def test_devices_marks_the_selected_one(monkeypatch, tmp_path):
    patch_hw(monkeypatch)
    r = runner.invoke(app, ["devices", "--db", str(tmp_path / "n.db")])
    line = next(ln for ln in r.stdout.splitlines() if "NVIDIA GPU" in ln)
    assert "*" in line
    cpu_line = next(ln for ln in r.stdout.splitlines() if "x86" in ln)
    assert "*" not in cpu_line


def test_devices_shows_settings_file_values(monkeypatch, tmp_path):
    patch_hw(monkeypatch)
    db = tmp_path / "n.db"
    save_settings(db, {"decode_workers": 3, "device_index": 1})
    out = flat(runner.invoke(app, ["devices", "--db", str(db)]).stdout)
    assert "device index: 1" in out
    assert "Decode threads 3 3" in out


def test_devices_offers_install_steps_for_present_but_unusable_hardware(monkeypatch, tmp_path):
    patch_hw(monkeypatch, hardware=["rocm"])
    out = flat(runner.invoke(app, ["devices", "--db", str(tmp_path / "n.db")]).stdout)
    assert "Enable your ROCM GPU" in out
    assert "pip install torch --index-url https://download.pytorch.org/whl/rocm6.2" in out
    assert 'pip install "photocure[ml]"' in out


def test_devices_falls_back_to_cpu_when_no_list_function(monkeypatch, tmp_path):
    from photocure.inference import backends as bk
    monkeypatch.delattr(bk, "list_accelerators", raising=False)
    monkeypatch.setattr(cli_mod, "describe_backends", lambda cfg: {"device": "cpu"})
    monkeypatch.setattr(cli_mod, "_detect_hardware", lambda: [])
    r = runner.invoke(app, ["devices", "--db", str(tmp_path / "n.db")])
    assert r.exit_code == 0 and "cpu" in flat(r.stdout)


def test_doctor_includes_accelerator_table_and_workers(monkeypatch, tmp_path):
    patch_hw(monkeypatch)
    monkeypatch.setattr(cli_mod, "_pkg_version", lambda *n: "1.0")
    monkeypatch.setattr(cli_mod, "_clip_weights_cached", lambda cfg: True)
    out = flat(runner.invoke(app, ["doctor", "--db", str(tmp_path / "n.db")]).stdout)
    assert "Accelerators" in out and "RTX 4060" in out
    assert "Workers and threads" in out


@pytest.mark.parametrize("kind,expected", [
    ("cuda", "pip install torch --index-url https://download.pytorch.org/whl/cu124"),
    ("rocm", "pip install torch --index-url https://download.pytorch.org/whl/rocm6.2"),
    ("xpu", "pip install torch --index-url https://download.pytorch.org/whl/xpu"),
])
def test_doctor_prints_matching_install_command_when_hardware_is_unusable(
        monkeypatch, tmp_path, kind, expected):
    cpu_only = [a for a in ACCEL if a["id"] == "cpu"]
    info = dict(INFO, device="cpu", selected="cpu")
    patch_hw(monkeypatch, accels=cpu_only, info=info, hardware=[kind])
    monkeypatch.setattr(cli_mod, "_pkg_version", lambda *n: "1.0")
    monkeypatch.setattr(cli_mod, "_clip_weights_cached", lambda cfg: True)
    out = flat(runner.invoke(app, ["doctor", "--db", str(tmp_path / "n.db")]).stdout)
    assert expected in out
    assert 'pip install "photocure[ml]"' in out
    if kind == "cuda":
        assert "may need changing" in out


def test_doctor_apple_silicon_uses_the_default_wheel(monkeypatch, tmp_path):
    cpu_only = [a for a in ACCEL if a["id"] == "cpu"]
    info = dict(INFO, device="cpu", selected="cpu")
    patch_hw(monkeypatch, accels=cpu_only, info=info, hardware=["mps"])
    monkeypatch.setattr(cli_mod, "_pkg_version", lambda *n: None)
    monkeypatch.setattr(cli_mod, "_clip_weights_cached", lambda cfg: None)
    out = flat(runner.invoke(app, ["doctor", "--db", str(tmp_path / "n.db")]).stdout)
    assert "pip install torch (the Apple GPU works with the default wheel)" in out


def test_doctor_has_no_gpu_advice_when_gpu_is_in_use(monkeypatch, tmp_path):
    patch_hw(monkeypatch, hardware=["cuda"])
    monkeypatch.setattr(cli_mod, "_pkg_version", lambda *n: "1.0")
    monkeypatch.setattr(cli_mod, "_clip_weights_cached", lambda cfg: True)
    out = flat(runner.invoke(app, ["doctor", "--db", str(tmp_path / "n.db")]).stdout)
    assert "not usable by torch" not in out and "Everything is ready" in out


def test_detect_hardware_from_tools_and_platform(monkeypatch):
    tools = {"nvidia-smi": "/usr/bin/nvidia-smi", "rocminfo": "/opt/rocm/bin/rocminfo",
             "xpu-smi": "/usr/bin/xpu-smi"}
    monkeypatch.setattr(cli_mod.shutil, "which", lambda n: tools.get(n))
    monkeypatch.setattr(cli_mod.platform, "system", lambda: "Linux")
    assert cli_mod._detect_hardware() == ["cuda", "rocm", "xpu"]
    monkeypatch.setattr(cli_mod.shutil, "which", lambda n: None)
    monkeypatch.setattr(cli_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli_mod.platform, "machine", lambda: "arm64")
    assert cli_mod._detect_hardware() == ["mps"]
    monkeypatch.setattr(cli_mod.platform, "machine", lambda: "x86_64")
    assert cli_mod._detect_hardware() == []


# ------------------------------------------------------------------ packaging
def test_docker_files_are_consistent():
    """One auto-detecting image: overrides only pass devices through and hint the hardware."""
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    docker = (root / "Dockerfile").read_text(encoding="utf-8")
    # Optional baked build still exists, but the default image installs at first start.
    assert "ARG TORCH_INDEX" in docker and '--index-url "$TORCH_INDEX"' in docker
    assert 'ENTRYPOINT ["pcur-entrypoint"]' in docker and 'CMD ["auto"]' in docker
    assert "docker/entrypoint.sh" in docker and "PYTHONUSERBASE=/data/pylibs" in docker
    entry = (root / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    assert "pylibs" in entry and "PYTHONUSERBASE" in entry and "photocure setup --yes" in entry

    base = yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf-8"))["services"]["photocure"]
    assert base["build"] == "." and "args" not in (base["build"] if isinstance(base["build"], dict) else {})
    assert base["environment"]["PCUR_DEVICE_DEVICE"] == "auto"   # never pinned to cpu

    expect = {"docker-compose.gpu.yml": "cuda", "docker-compose.rocm.yml": "rocm",
              "docker-compose.intel.yml": "xpu"}
    for name, accel in expect.items():
        text = (root / name).read_text(encoding="utf-8")
        svc = yaml.safe_load(text)["services"]["photocure"]
        assert svc["environment"]["PCUR_ACCEL"] == accel, name
        assert "EXTRAS" not in str(svc.get("build", {})), name   # no per-hardware image
        assert "\u2014" not in text
    nv = yaml.safe_load((root / "docker-compose.gpu.yml").read_text())["services"]["photocure"]
    assert nv["deploy"]["resources"]["reservations"]["devices"][0]["driver"] == "nvidia"
    rocm = yaml.safe_load((root / "docker-compose.rocm.yml").read_text())["services"]["photocure"]
    assert {"/dev/kfd", "/dev/dri"} <= set(rocm["devices"]) and rocm["group_add"]
    intel = yaml.safe_load((root / "docker-compose.intel.yml").read_text())["services"]["photocure"]
    assert intel["devices"] == ["/dev/dri"] and intel["group_add"]
