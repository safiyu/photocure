"""Accelerator detection, fallback, providers and worker counts (fake torch only)."""
from __future__ import annotations

import sys
import types

import pytest

from photocure.config import CuratorConfig
from photocure.inference import backends as B
from photocure.inference.backends import (
    BackendState,
    describe_backend,
    detect_backend,
    empty_cuda_cache,
    list_accelerators,
    reset_backend_cache,
    resolve_decode_workers,
    resolve_torch_threads,
    select_onnx_providers,
)

ACCEL_IDS = ["cuda", "rocm", "xpu", "mps", "directml", "cpu"]


def fake_torch(cuda=False, hip=None, xpu=None, mps=False, mps_attr=True,
               capability=(8, 9), rec=None, cuda_build=False):
    """A minimal torch stand-in. ``xpu=None`` means the build has no xpu module."""
    rec = rec if rec is not None else {}
    mod = types.ModuleType("torch")
    has_cuda_build = (cuda or cuda_build) and not hip
    mod.version = types.SimpleNamespace(hip=hip, cuda="12.4" if has_cuda_build else None)
    mod.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_capability=lambda: capability,
        set_per_process_memory_fraction=lambda f: rec.__setitem__("fraction", f),
        empty_cache=lambda: rec.__setitem__("cache", rec.get("cache", []) + ["cuda"]),
        get_device_name=lambda i=0: "Test GPU",
        get_device_properties=lambda i=0: types.SimpleNamespace(total_memory=8e9),
        set_device=lambda i: rec.__setitem__("cuda_index", i),
    )
    if xpu is not None:
        mod.xpu = types.SimpleNamespace(
            is_available=lambda: xpu,
            get_device_name=lambda i=0: "Intel(R) Iris(R) Xe Graphics",
            get_device_properties=lambda i=0: types.SimpleNamespace(total_memory=4e9),
            set_device=lambda i: rec.__setitem__("xpu_index", i),
            empty_cache=lambda: rec.__setitem__("cache", rec.get("cache", []) + ["xpu"]),
        )
    if mps_attr:
        mod.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps))
    mod.mps = types.SimpleNamespace(
        empty_cache=lambda: rec.__setitem__("cache", rec.get("cache", []) + ["mps"]))
    mod.set_num_threads = lambda n: rec.__setitem__("threads", n)
    mod.set_num_interop_threads = lambda n: rec.__setitem__("interop", n)
    return mod, rec


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(B, "_directml_module", lambda: None)
    reset_backend_cache()
    yield
    reset_backend_cache()


def _detect(monkeypatch, torch_mod, device="auto", **ds):
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    reset_backend_cache()
    c = CuratorConfig()
    c.device.device = device
    for k, v in ds.items():
        setattr(c.device, k, v)
    return detect_backend(c)


# ------------------------------------------------------------ resolution
def test_auto_prefers_cuda_over_everything(monkeypatch):
    t, _ = fake_torch(cuda=True, xpu=True, mps=True)
    bs = _detect(monkeypatch, t)
    assert bs.accelerator == "cuda" and bs.device == "cuda" and bs.fallback_reason == ""


def test_auto_falls_to_xpu_then_mps_then_cpu(monkeypatch):
    t, _ = fake_torch(xpu=True, mps=True)
    assert _detect(monkeypatch, t).accelerator == "xpu"
    t, _ = fake_torch(xpu=False, mps=True)
    assert _detect(monkeypatch, t).accelerator == "mps"
    t, _ = fake_torch()
    bs = _detect(monkeypatch, t)
    assert bs.accelerator == "cpu" and bs.device == "cpu"


@pytest.mark.parametrize("device,kw,acc,dev", [
    ("cuda", dict(cuda=True), "cuda", "cuda"),
    ("xpu", dict(xpu=True), "xpu", "xpu"),
    ("mps", dict(mps=True), "mps", "mps"),
    ("cpu", dict(cuda=True, mps=True), "cpu", "cpu"),
])
def test_requested_device_resolves(monkeypatch, device, kw, acc, dev):
    t, _ = fake_torch(**kw)
    bs = _detect(monkeypatch, t, device)
    assert (bs.accelerator, bs.device) == (acc, dev)
    assert bs.fallback_reason == ""


def test_rocm_detected_via_hip_and_uses_cuda_api(monkeypatch):
    t, rec = fake_torch(cuda=True, hip="6.2")
    bs = _detect(monkeypatch, t, "rocm")
    assert bs.accelerator == "rocm" and bs.device == "cuda"
    assert bs.fp16 is True and rec["fraction"] == bs.memory_fraction
    # auto picks it too, and plain cuda is reported unavailable with a reason
    auto = _detect(monkeypatch, t)
    assert auto.accelerator == "rocm"
    probes = {a["id"]: a for a in list_accelerators(t)}
    assert probes["rocm"]["available"] and not probes["cuda"]["available"]
    assert "ROCm" in probes["cuda"]["reason"]


def test_alias_integrated(monkeypatch):
    t, _ = fake_torch(xpu=True, mps=True)
    assert _detect(monkeypatch, t, "integrated").accelerator == "xpu"
    t, _ = fake_torch(xpu=None, mps=True)
    assert _detect(monkeypatch, t, "integrated").accelerator == "mps"
    t, _ = fake_torch(cuda=True)
    # a discrete NVIDIA GPU is not an integrated one, so the request falls to cpu
    assert _detect(monkeypatch, t, "integrated").accelerator == "cpu"


def test_alias_integrated_none_available_falls_to_cpu_with_reason(monkeypatch):
    t, _ = fake_torch()
    bs = _detect(monkeypatch, t, "integrated")
    assert bs.accelerator == "cpu"
    assert "integrated is not available" in bs.fallback_reason
    assert "xpu:" in bs.fallback_reason and "mps:" in bs.fallback_reason


def test_fallback_reason_text(monkeypatch):
    t, _ = fake_torch(cuda=False, mps=True)
    bs = _detect(monkeypatch, t, "cuda")
    assert bs.accelerator == "mps"
    assert bs.requested == "cuda"
    assert "cuda is not available" in bs.fallback_reason
    assert "using mps" in bs.fallback_reason
    t, _ = fake_torch()
    bs = _detect(monkeypatch, t, "xpu")
    assert bs.accelerator == "cpu"
    assert "xpu is not available" in bs.fallback_reason
    assert "no Intel GPU (XPU) support" in bs.fallback_reason


def test_directml_best_effort(monkeypatch):
    t, _ = fake_torch()
    dml = types.SimpleNamespace(device_count=lambda: 1, device=lambda: "privateuseone:0",
                                device_name=lambda i=0: "Test DML")
    monkeypatch.setattr(B, "_directml_module", lambda: dml)
    bs = _detect(monkeypatch, t, "directml")
    assert bs.accelerator == "directml" and bs.device == "privateuseone:0"
    assert bs.fp16 is False


def test_no_torch_means_cpu_and_reasons(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import raises ImportError
    reset_backend_cache()
    c = CuratorConfig()
    c.device.device = "cuda"
    bs = detect_backend(c)
    assert bs.accelerator == "cpu" and bs.torch is None
    assert "PyTorch is not installed" in bs.fallback_reason


# ------------------------------------------------------------------ fp16
def test_fp16_only_on_cuda_rocm_xpu(monkeypatch):
    for kw, device, expected in [
        (dict(cuda=True), "cuda", True),
        (dict(cuda=True, capability=(6, 1)), "cuda", False),
        (dict(cuda=True, hip="6.2"), "rocm", True),
        (dict(xpu=True), "xpu", True),
        (dict(mps=True), "mps", False),
    ]:
        t, _ = fake_torch(**kw)
        assert _detect(monkeypatch, t, device).fp16 is expected, (device, kw)


def test_memory_fraction_only_cuda_and_rocm(monkeypatch):
    t, rec = fake_torch(xpu=True)
    _detect(monkeypatch, t, "xpu")
    assert "fraction" not in rec
    t, rec = fake_torch(cuda=True)
    bs = _detect(monkeypatch, t, "cuda")
    assert rec["fraction"] == bs.memory_fraction


# ------------------------------------------------------------ device index
def test_device_index_applied(monkeypatch):
    t, rec = fake_torch(cuda=True)
    _detect(monkeypatch, t, "cuda", device_index=1)
    assert rec["cuda_index"] == 1
    t, rec = fake_torch(xpu=True)
    _detect(monkeypatch, t, "xpu", device_index=1)
    assert rec["xpu_index"] == 1
    t, rec = fake_torch(cuda=True)
    _detect(monkeypatch, t, "cuda", device_index=0)
    assert "cuda_index" not in rec


# ------------------------------------------------------------ empty cache
@pytest.mark.parametrize("device,expect", [("cuda", "cuda"), ("xpu", "xpu"), ("mps", "mps")])
def test_empty_cache_per_device(device, expect):
    t, rec = fake_torch(cuda=True, xpu=True, mps=True)
    bs = BackendState(device=device, fp16=False, torch=t, cuda_available=True,
                      memory_fraction=0.75, face_backend="stub", semantic_backend="stub")
    empty_cuda_cache(bs)
    assert rec["cache"] == [expect]
    cpu = BackendState(device="cpu", fp16=False, torch=t, cuda_available=True,
                       memory_fraction=0.75, face_backend="stub", semantic_backend="stub")
    empty_cuda_cache(cpu)
    assert rec["cache"] == [expect]


# -------------------------------------------------------- list_accelerators
def test_list_accelerators_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    rows = list_accelerators()
    assert [r["id"] for r in rows] == ACCEL_IDS
    for r in rows:
        assert set(r) == {"id", "label", "name", "kind", "memory_gb", "available",
                          "reason", "install_hint"}
    by = {r["id"]: r for r in rows}
    assert by["cpu"]["available"] and by["cpu"]["kind"] == "cpu"
    assert by["cuda"]["reason"] == "PyTorch is not installed"
    assert "photocure[ml]" in by["cuda"]["install_hint"]
    assert by["directml"]["reason"] == "package not installed"


def test_list_accelerators_with_torch_reasons_and_details():
    t, _ = fake_torch(cuda=False, cuda_build=True, xpu=None, mps=False)
    by = {r["id"]: r for r in list_accelerators(t)}
    assert by["cuda"]["reason"] == "no NVIDIA GPU or driver detected"
    assert by["rocm"]["reason"] == "torch build has no ROCm support"
    assert by["xpu"]["reason"] == "torch build has no Intel GPU (XPU) support"
    assert "macOS" in by["mps"]["reason"]
    t, _ = fake_torch(cuda=True, xpu=True, mps=True)
    by = {r["id"]: r for r in list_accelerators(t)}
    assert by["cuda"]["available"] and by["cuda"]["memory_gb"] == 8.0
    assert by["cuda"]["kind"] == "discrete" and by["cuda"]["name"] == "Test GPU"
    assert by["xpu"]["kind"] == "integrated"      # Iris Xe
    assert by["mps"]["kind"] == "integrated" and by["mps"]["available"]


def test_cpu_only_torch_build_reason():
    t, _ = fake_torch(cuda=False)
    by = {r["id"]: r for r in list_accelerators(t)}
    assert by["cuda"]["reason"] == "torch build has no CUDA support"


def test_describe_backend_keys(monkeypatch):
    t, _ = fake_torch(mps=True)
    bs = _detect(monkeypatch, t, "cuda")
    monkeypatch.setattr(B, "_have_open_clip", lambda: False)   # model not installed
    d = describe_backend(bs)
    for key in ("device", "semantic_backend", "face_backend", "semantic_is_stub", "forced_by_env",
                "face_is_fallback", "hint", "accelerators", "selected", "requested",
                "fallback_reason"):
        assert key in d
    assert d["selected"] == "mps" and d["requested"] == "cuda"
    assert "cuda is not available" in d["fallback_reason"]
    assert [a["id"] for a in d["accelerators"]] == ACCEL_IDS
    assert d["semantic_is_stub"] is True
    assert d["forced_by_env"] is False
    assert 'pip install "photocure[ml]"' in d["hint"] and "photocure setup" in d["hint"]
    bs.semantic_backend = "clip"
    assert describe_backend(bs)["hint"] == ""


def test_model_installed_but_switched_off_by_the_environment(monkeypatch):
    """The fix is unset PCUR_BACKEND, not installing something that is already there."""
    t, _ = fake_torch(mps=True)
    bs = _detect(monkeypatch, t, "auto")
    monkeypatch.setattr(B, "_have_open_clip", lambda: True)
    monkeypatch.setenv("PCUR_BACKEND", "stub")
    d = describe_backend(bs)
    assert d["semantic_is_stub"] and d["forced_by_env"] is True
    assert "unset PCUR_BACKEND" in d["hint"] and "pip install" not in d["hint"]
    monkeypatch.delenv("PCUR_BACKEND")
    assert describe_backend(bs)["forced_by_env"] is False


# -------------------------------------------------------------- providers
def test_provider_selection():
    avail = ["CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider",
             "OpenVINOExecutionProvider", "CoreMLExecutionProvider",
             "DmlExecutionProvider", "CPUExecutionProvider"]
    assert select_onnx_providers("cuda", avail) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert select_onnx_providers("rocm", avail) == [
        "ROCMExecutionProvider", "MIGraphXExecutionProvider", "CPUExecutionProvider"]
    assert select_onnx_providers("xpu", avail) == ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    assert select_onnx_providers("mps", avail) == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert select_onnx_providers("directml", avail) == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert select_onnx_providers("cpu", avail) == ["CPUExecutionProvider"]
    # only providers that exist are used
    assert select_onnx_providers("cuda", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert select_onnx_providers("rocm", ["MIGraphXExecutionProvider", "CPUExecutionProvider"]) == [
        "MIGraphXExecutionProvider", "CPUExecutionProvider"]


def test_providers_without_onnxruntime(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert select_onnx_providers("cuda") == ["CPUExecutionProvider"]


# ---------------------------------------------------------------- workers
def _cfg_with(**ds):
    c = CuratorConfig()
    for k, v in ds.items():
        setattr(c.device, k, v)
    return c


def test_decode_workers_auto_and_explicit(monkeypatch):
    monkeypatch.setattr(B, "_cpu_count", lambda: 4)
    assert resolve_decode_workers(_cfg_with(decode_workers=0)) == 4
    monkeypatch.setattr(B, "_cpu_count", lambda: 32)
    assert resolve_decode_workers(_cfg_with(decode_workers=0)) == 8
    assert resolve_decode_workers(_cfg_with(decode_workers=3)) == 3


def test_torch_threads_auto_and_explicit(monkeypatch):
    monkeypatch.setattr(B, "_cpu_count", lambda: 12)
    assert resolve_torch_threads(_cfg_with(torch_threads=0), "cpu") == 8
    assert resolve_torch_threads(_cfg_with(torch_threads=0), "cuda") == 2
    assert resolve_torch_threads(_cfg_with(torch_threads=5), "cpu") == 5
    monkeypatch.setattr(B, "_cpu_count", lambda: 2)
    assert resolve_torch_threads(_cfg_with(torch_threads=0), "cpu") == 2


def test_detect_applies_thread_counts_once(monkeypatch):
    t, rec = fake_torch()
    bs = _detect(monkeypatch, t, "cpu", torch_threads=3)
    assert rec["threads"] == 3 and rec["interop"] == 2
    assert bs.torch_threads == 3 and bs.decode_workers >= 1
    rec.clear()
    B.apply_thread_settings(t, 5)  # second call in the same process is a no-op
    assert rec == {}


def test_thread_env_only_before_libs_load(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    assert B.configure_threads_env(_cfg_with(torch_threads=3)) is True
    import os
    assert os.environ["OMP_NUM_THREADS"] == "3"
    monkeypatch.delenv("OMP_NUM_THREADS")
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    assert B.configure_threads_env(_cfg_with()) is False
    assert "OMP_NUM_THREADS" not in os.environ
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert B.configure_threads_env(_cfg_with()) is False  # user value wins
    assert os.environ["OMP_NUM_THREADS"] == "7"


def test_stub_warning_logged_once(monkeypatch, caplog):
    t, _ = fake_torch()
    monkeypatch.delenv("PCUR_BACKEND", raising=False)
    monkeypatch.setattr(B, "_have_open_clip", lambda: False)
    monkeypatch.setitem(sys.modules, "torch", t)
    with caplog.at_level("WARNING", logger="photocure.backends"):
        reset_backend_cache()
        detect_backend(CuratorConfig())
        reset_backend_cache()
        B._stub_warned = True  # a second detect in the same process stays quiet
        detect_backend(CuratorConfig())
    msgs = [r.getMessage() for r in caplog.records if "No AI model" in r.getMessage()]
    assert len(msgs) == 1 and 'photocure[ml]' in msgs[0]


def test_stub_warning_not_logged_when_user_forces_stub(monkeypatch, caplog):
    t, _ = fake_torch()
    monkeypatch.setenv("PCUR_BACKEND", "stub")
    monkeypatch.setitem(sys.modules, "torch", t)
    with caplog.at_level("WARNING", logger="photocure.backends"):
        reset_backend_cache()
        detect_backend(CuratorConfig())
    assert not [r for r in caplog.records if "No AI model" in r.getMessage()]


def test_list_accelerators_reports_every_gpu_for_the_picker():
    import types

    props = lambda i: types.SimpleNamespace(total_memory=(8 + 8 * i) * 10**9)  # noqa: E731
    cuda = types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 2,
        get_device_name=lambda i: f"NVIDIA Test {i}", get_device_properties=props)
    fake = types.SimpleNamespace(
        cuda=cuda, version=types.SimpleNamespace(cuda="12.4", hip=None),
        backends=types.SimpleNamespace(mps=None), xpu=None)
    entry = next(a for a in list_accelerators(fake) if a["id"] == "cuda")
    assert entry["available"] and entry["count"] == 2
    assert [d["name"] for d in entry["devices"]] == ["NVIDIA Test 0", "NVIDIA Test 1"]
    assert entry["devices"][1]["memory_gb"] == 16.0
