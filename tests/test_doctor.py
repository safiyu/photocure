"""doctor, infer warnings and flags, install extras and config defaults."""
from __future__ import annotations

import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from photocure import cli as cli_mod
from photocure.cli import app
from photocure.config import CuratorConfig, DeviceSettings, TaxonomySettings

from .conftest import make_image

runner = CliRunner()
ROOT = Path(__file__).resolve().parent.parent

STUB = {"device": "cpu", "semantic_backend": "stub", "face_backend": "cv2",
        "semantic_is_stub": True, "face_is_fallback": True, "hint": "x"}
REAL = {"device": "mps", "semantic_backend": "clip", "face_backend": "insightface",
        "semantic_is_stub": False, "face_is_fallback": False, "hint": ""}


def flat(out: str) -> str:
    for ch in "│┃┏┓┗┛┡┩╇━─╭╮╰╯┳┻╋┠┨":
        out = out.replace(ch, " ")
    return " ".join(out.split())


def patch_doctor(monkeypatch, info, versions, weights):
    cpu = {"id": "cpu", "name": "CPU", "kind": "cpu", "memory": "shared", "available": True,
           "reason": "", "hint": "", "selected": info.get("device") == "cpu"}
    monkeypatch.setattr(cli_mod, "_accelerators", lambda cfg, i: [cpu])
    monkeypatch.setattr(cli_mod, "_detect_hardware", lambda: [])
    monkeypatch.setattr(cli_mod, "describe_backends", lambda cfg: dict(info))
    monkeypatch.setattr(cli_mod, "_pkg_version",
                        lambda *names: versions.get(names[0]))
    monkeypatch.setattr(cli_mod, "_clip_weights_cached", lambda cfg: weights)


def test_doctor_without_model_says_placeholder_and_how_to_fix(monkeypatch, tmp_path):
    patch_doctor(monkeypatch, STUB, {}, None)
    db = tmp_path / "none.db"
    r = runner.invoke(app, ["doctor", "--db", str(db)])
    out = flat(r.stdout)
    assert r.exit_code == 0
    assert "placeholder" in out
    assert 'pip install "photocure[ml]"' in out
    assert "missing" in out
    assert not db.exists()  # doctor never creates the database


def test_doctor_with_model_is_ready(monkeypatch, tmp_path):
    versions = {"torch": "2.4", "open_clip_torch": "2.26", "insightface": "0.7",
                "onnxruntime": "1.18"}
    patch_doctor(monkeypatch, REAL, versions, True)
    r = runner.invoke(app, ["doctor", "--db", str(tmp_path / "none.db")])
    out = flat(r.stdout)
    assert r.exit_code == 0
    assert "mps" in out and "2.4" in out
    assert "Everything is ready" in out


def test_doctor_flags_missing_weights_and_old_stub_rows(monkeypatch, tmp_path):
    from photocure.db.models import SemanticScore
    from photocure.db.repository import PhotoRepository
    from photocure.db.session import get_session

    from .world import build_world

    w = build_world(tmp_path, n=2)
    with get_session(w.db) as s:
        repo = PhotoRepository(s)
        repo.upsert_semantic(SemanticScore(image_id=w.ids[0], backend="stub"))
        repo.upsert_semantic(SemanticScore(image_id=w.ids[1], backend="clip"))
    versions = {"torch": "2.4", "open_clip_torch": "2.26", "insightface": "0.7"}
    patch_doctor(monkeypatch, REAL, versions, False)
    r = runner.invoke(app, ["doctor", "--db", str(w.db)])
    out = flat(r.stdout)
    assert r.exit_code == 0
    assert "download on first use" in out
    assert "infer --reanalyze" in out
    assert "analyzed without a model" in out


def test_doctor_exit_code_zero_when_backend_detection_breaks(monkeypatch, tmp_path):
    patch_doctor(monkeypatch, {"device": "cpu", "semantic_backend": "stub",
                               "face_backend": "stub", "semantic_is_stub": True,
                               "face_is_fallback": True, "hint": "boom"}, {}, None)
    assert runner.invoke(app, ["doctor", "--db", str(tmp_path / "n.db")]).exit_code == 0


def test_clip_weights_cached_unknown_without_open_clip(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "open_clip":
            raise ImportError("no open_clip")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert cli_mod._clip_weights_cached(CuratorConfig()) is None


def test_clip_weights_cached_reads_hf_cache(monkeypatch, tmp_path):
    open_clip = pytest.importorskip("open_clip")
    monkeypatch.setattr(open_clip, "get_pretrained_cfg",
                        lambda m, p: {"hf_hub": "timm/some-model/"})
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert cli_mod._clip_weights_cached(CuratorConfig()) is False
    snap = tmp_path / "models--timm--some-model" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "weights.bin").write_bytes(b"x")
    assert cli_mod._clip_weights_cached(CuratorConfig()) is True


# ---------------------------------------------------------------- infer flags
@pytest.fixture()
def scanned(tmp_path):
    src = tmp_path / "photos"
    make_image(src / "a.jpg", rng_seed=1)
    make_image(src / "b.jpg", rng_seed=2)
    db = tmp_path / "c.db"
    assert runner.invoke(app, ["scan", str(src), "--db", str(db)]).exit_code == 0
    return db


def capture_run(monkeypatch):
    from photocure.inference import pipeline as pl
    seen: dict = {}

    def fake_run(self, db_path=None, only_paths=None, show_progress=True,
                 reanalyze_stub=None):
        seen["reanalyze_stub"] = reanalyze_stub
        return types.SimpleNamespace(processed=0, faces_total=0, quality_computed=0,
                                     semantic_computed=0, failures=0, duration_s=0.0)

    monkeypatch.setattr(pl.InferencePipeline, "run", fake_run)
    return seen


def test_infer_warns_when_placeholder_classifier_is_active(scanned, monkeypatch):
    from photocure.inference import backends as B
    monkeypatch.setattr(B, "_have_open_clip", lambda: False)   # model not installed
    r = runner.invoke(app, ["infer", "--db", str(scanned), "--device", "cpu"])
    out = flat(r.stdout)
    assert r.exit_code == 0
    assert "No AI model is installed, so photos cannot be recognised" in out
    assert 'pip install "photocure[ml]"' in out
    assert "placeholder classifier" in out  # repeated under the result table


def test_infer_says_unset_when_the_model_is_installed_but_switched_off(scanned, monkeypatch):
    from photocure.inference import backends as B
    monkeypatch.setattr(B, "_have_open_clip", lambda: True)
    r = runner.invoke(app, ["infer", "--db", str(scanned), "--device", "cpu"])
    out = flat(r.stdout)
    assert r.exit_code == 0
    assert "installed but switched off" in out and "unset PCUR_BACKEND" in out
    assert "No AI model is installed" not in out


def test_infer_rejects_unknown_device(scanned):
    r = runner.invoke(app, ["infer", "--db", str(scanned), "--device", "tpu"])
    assert r.exit_code == 2
    assert "auto, cuda, rocm, xpu, mps, directml, integrated, cpu" in flat(r.stdout)


@pytest.mark.parametrize("device", ["auto", "cuda", "rocm", "xpu", "mps", "directml",
                                    "integrated", "cpu"])
def test_infer_accepts_every_device(scanned, device):
    r = runner.invoke(app, ["infer", "--db", str(scanned), "--device", device,
                            "--limit", "1"])
    assert r.exit_code == 0, r.stdout


def test_infer_reanalyze_defaults_off_with_placeholder(monkeypatch, scanned):
    seen = capture_run(monkeypatch)
    assert runner.invoke(app, ["infer", "--db", str(scanned)]).exit_code == 0
    assert seen["reanalyze_stub"] is False


def test_infer_reanalyze_defaults_on_when_model_installed(monkeypatch, scanned):
    seen = capture_run(monkeypatch)
    monkeypatch.setattr(cli_mod, "describe_backends", lambda cfg: dict(REAL))
    assert runner.invoke(app, ["infer", "--db", str(scanned)]).exit_code == 0
    assert seen["reanalyze_stub"] is True


def test_infer_reanalyze_flags_override(monkeypatch, scanned):
    seen = capture_run(monkeypatch)
    runner.invoke(app, ["infer", "--db", str(scanned), "--reanalyze"])
    assert seen["reanalyze_stub"] is True
    monkeypatch.setattr(cli_mod, "describe_backends", lambda cfg: dict(REAL))
    runner.invoke(app, ["infer", "--db", str(scanned), "--no-reanalyze"])
    assert seen["reanalyze_stub"] is False


def test_infer_queues_stub_rows_when_pipeline_lacks_the_parameter(monkeypatch, scanned):
    from photocure.inference import pipeline as pl
    called = {}

    def old_run(self, db_path=None, only_paths=None, show_progress=True):
        return types.SimpleNamespace(processed=0, faces_total=0, quality_computed=0,
                                     semantic_computed=0, failures=0, duration_s=0.0)

    monkeypatch.setattr(pl.InferencePipeline, "run", old_run)
    monkeypatch.setattr(pl, "mark_stub_rows_for_reanalysis",
                        lambda session: called.setdefault("n", 3), raising=False)
    r = runner.invoke(app, ["infer", "--db", str(scanned), "--reanalyze"])
    assert r.exit_code == 0 and called["n"] == 3
    assert "Queued 3 photos" in flat(r.stdout)


# ------------------------------------------------------------ extras and config
def test_device_default_is_auto_and_taxonomy_keys_exist():
    assert DeviceSettings.model_fields["device"].default == "auto"
    t = TaxonomySettings()
    assert t.min_class_floor == 0.25
    assert t.feedback_weight == 0.6
    assert t.feedback_min_examples == 2


def test_device_env_override(monkeypatch):
    monkeypatch.setenv("PCUR_DEVICE_DEVICE", "mps")
    assert CuratorConfig().device.device == "mps"


def test_pyproject_extras():
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    names = lambda key: {r.split(">")[0].split("[")[0] for r in extras[key]}  # noqa: E731
    assert {"torch", "torchvision", "open-clip-torch"} <= names("ml")
    assert not names("ml") & {"onnxruntime-gpu", "insightface"}
    assert {"insightface", "onnxruntime"} == names("faces")
    assert names("ml") <= names("gpu") and "onnxruntime-gpu" in names("gpu")


def test_no_em_dashes_in_docs_and_packaging():
    for name in ("README.md", "pyproject.toml", "requirements.txt", "Dockerfile"):
        assert "\u2014" not in (ROOT / name).read_text(encoding="utf-8"), name


def test_doctor_says_unset_when_the_model_is_installed_but_switched_off(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from photocure import cli as cli_mod
    from photocure.inference import backends as B

    B.reset_backend_cache()
    monkeypatch.setenv("PCUR_BACKEND", "stub")
    monkeypatch.setattr(B, "_have_open_clip", lambda: True)
    r = CliRunner().invoke(cli_mod.app, ["doctor", "--db", str(tmp_path / "d.db")])
    B.reset_backend_cache()
    assert r.exit_code == 0
    out = flat(r.output)
    assert "unset PCUR_BACKEND" in out
    assert "Install the AI model" not in out


def test_stub_panel_text_depends_on_the_cause():
    from photocure import cli as cli_mod

    forced = cli_mod._stub_warning({"forced_by_env": True}).renderable
    missing = cli_mod._stub_warning({"forced_by_env": False}).renderable
    assert "unset PCUR_BACKEND" in forced and "pip install" not in forced
    assert "pip install" in missing and "unset" not in missing
