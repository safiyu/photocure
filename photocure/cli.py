"""Typer CLI for Photocure.

Commands:
    scan         Scan a directory into the state store (resumable, --prune-missing, --watch).
    infer        Run quality + face + semantic inference over pending images.
    cluster      Cluster bursts and pick best shots.
    plan         Build a persisted dry-run routing plan (+ optional HTML report).
    apply        Execute a persisted plan (move / copy / hardlink / symlink / trash).
    undo         Roll back a committed transaction.
    verify-txn   Re-hash every committed target of a transaction.
    repair       Fix images.path from the ledger.
    trash-purge  Delete trash staging older than the retention window.
    similar      Find images similar to one image.
    search       Find photos by describing them in words.
    highlights   Export the best photos of a year as a page, optionally with copies.
    keywords     Write (or remove) category, highlight and event keywords in XMP sidecars.
    events       Group images into events.
    export-onnx  Export the models to ONNX.
    transactions List ledger transactions.
    stats        Show state-store statistics.
    serve        Launch the web review dashboard.
"""
from __future__ import annotations

import inspect
import platform
import shlex
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import (
    DEFAULT_DB,
    DEVICE_CHOICES,
    MAX_WORKERS,
    CuratorConfig,
    auto_scan_workers,
    resolve_workers,
)
from sqlmodel import col, select

from .db.models import ImageRecord
from .db.session import get_session, init_db
from .db.repository import PhotoRepository
from .inference.backends import detect_backend, ml_install_cmd, pip_install_cmd, reset_backend_cache
from .services import (
    PlanDriftError,
    Planner,
    apply_settings,
    describe_backends,
    load_settings,
)

app = typer.Typer(
    name="photocure",
    help="Intelligent image curator: offline, GPU-accelerated, transactionally reversible photo curation.",
    no_args_is_help=True,
    add_completion=False,
)
cli = app  # alias for the console-script entry point (photocure.cli:cli)
console = Console()


def _config(db: Path, refresh: bool = True) -> CuratorConfig:
    """Return a config bound to ``db`` (initializing the schema on first use)."""
    reset_backend_cache() if refresh else None
    cfg = _config_no_init(db)
    init_db(cfg.db_path)
    return cfg


def _config_no_init(db: Path) -> CuratorConfig:
    """Config for ``db`` with the settings file applied over the environment.

    Precedence: explicit CLI flag (applied by the caller) > settings file > environment.
    """
    cfg = CuratorConfig.load(db)
    return apply_settings(cfg, load_settings(cfg.db_path))


def _lazy(module: str, attr: str):
    """Import ``module.attr`` or exit with a clear message if it is not available."""
    import importlib
    try:
        mod = importlib.import_module(module, package=__package__)
        return getattr(mod, attr)
    except (ImportError, AttributeError) as e:
        console.print(f"[red]This feature is not available:[/red] {module}.{attr} ({e})")
        raise typer.Exit(1) from e


def _lazy_optional(module: str, attr: str):
    """Import ``module.attr`` or return None when it is not available."""
    import importlib
    try:
        return getattr(importlib.import_module(module, package=__package__), attr, None)
    except Exception:
        return None


def _call_flex(fn, *args, **kwargs):
    """Call ``fn`` passing only the keyword arguments its signature accepts."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def _mode(mode: str):
    from .rules.organizer import OrganizeMode
    return OrganizeMode.IN_PLACE if mode == "in_place" else OrganizeMode.CATEGORY_FIRST


def _print_diff(rep) -> None:
    t = Table(title="Dry-run diff")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Actions", str(rep.executed))
    t.add_row("Name collisions (renamed)", str(rep.collisions))
    t.add_row("Cross-device moves", str(rep.cross_device))
    t.add_row("Missing sources", str(rep.missing_sources))
    t.add_row("Bytes to read", _human(rep.bytes_total))
    t.add_row("Extra space needed", _human(rep.bytes_needed))
    for where, free in rep.free_bytes.items():
        t.add_row(f"Free on {where}", _human(free))
    t.add_row("Enough space", "yes" if rep.space_ok else "[red]NO[/red]")
    console.print(t)


def _human(n: int) -> str:
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024 or unit == "TiB":
            return f"{v:.1f} {unit}" if unit != "B" else f"{int(v)} B"
        v /= 1024
    return f"{n} B"


_DEVICES = DEVICE_CHOICES
TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu124",
    "rocm": "https://download.pytorch.org/whl/rocm6.2",
    "xpu": "https://download.pytorch.org/whl/xpu",
}


def _stub_warning(info: dict) -> Panel:
    if info.get("forced_by_env"):
        return Panel(
            "[bold]The AI model is installed but switched off.[/bold]\n"
            "The environment setting PCUR_BACKEND is forcing the placeholder, so most photos "
            "will end up as Uncertain. Remove it and run again:\n"
            "  [cyan]unset PCUR_BACKEND[/cyan]",
            title="AI model switched off", border_style="yellow")
    return Panel(
        "[bold]No AI model is installed, so photos cannot be recognised.[/bold]\n"
        "Most photos will end up as Uncertain until you install it:\n"
        f"  [cyan]{escape(ml_install_cmd())}[/cyan]\n"
        "Then run [cyan]photocure infer --reanalyze[/cyan] to redo photos that were "
        "analyzed without it.",
        title="Placeholder classifier in use", border_style="yellow")


def _pkg_version(*names: str) -> str | None:
    from importlib import metadata
    for n in names:
        try:
            return metadata.version(n)
        except metadata.PackageNotFoundError:
            continue
    return None


def _first(obj, *keys, default=None):
    """First present key of a dict or attribute of an object."""
    for k in keys:
        v = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
        if v is not None:
            return v
    return default


def _accelerators(cfg: CuratorConfig, info: dict) -> list[dict]:
    """Every accelerator the app knows about, normalized for display."""
    raw = None
    fn = _lazy_optional(".inference.backends", "list_accelerators")
    if fn is not None:
        try:
            raw = _call_flex(fn, device_index=cfg.device.device_index)
        except Exception:
            raw = None
    if raw is None:
        raw = info.get("accelerators")
    chosen = info.get("selected")
    chosen_id = str(_first(chosen, "id", default=chosen)) if chosen is not None else None
    out = []
    for a in raw or []:
        aid = str(_first(a, "id", "name", default="?"))
        kind = str(_first(a, "kind", default="") or "")
        gb = _first(a, "memory_gb")
        memory = (f"{float(gb):.1f} GB" if isinstance(gb, (int, float)) and gb
                  else ("shared" if kind in ("integrated", "cpu") else "unknown"))
        label = str(_first(a, "label", default=aid) or aid)
        dev_name = str(_first(a, "name", default="") or "")
        out.append({
            "id": aid,
            "name": f"{label}, {dev_name}" if dev_name and dev_name != label else label,
            "kind": kind,
            "memory": memory,
            "available": bool(_first(a, "available", default=True)),
            "reason": str(_first(a, "reason", default="") or ""),
            "hint": str(_first(a, "install_hint", "hint", default="") or ""),
            "selected": bool(_first(a, "selected", default=False)) or aid == chosen_id,
        })
    if not out:
        out.append({"id": "cpu", "name": platform.processor() or platform.machine() or "CPU",
                    "kind": "cpu", "memory": "shared", "available": True, "reason": "",
                    "hint": "", "selected": info.get("device") == "cpu"})
    return out


def _detect_hardware() -> list[str]:
    """Likely accelerator kinds on this machine, even when torch is CPU only or missing."""
    found = []
    if shutil.which("nvidia-smi"):
        found.append("cuda")
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        found.append("rocm")
    if shutil.which("sycl-ls") or shutil.which("xpu-smi"):
        found.append("xpu")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        found.append("mps")
    return found


def _install_steps(kind: str) -> list[str]:
    if kind == "mps":
        steps = [pip_install_cmd("torch") + "   (the Apple GPU works with the default wheel)"]
    else:
        steps = [pip_install_cmd(f"torch --index-url {TORCH_INDEX[kind]}")]
        if kind == "cuda":
            steps[0] += "   (cu124 may need changing to match your NVIDIA driver)"
    return steps + [ml_install_cmd()]


def _usable_kinds(accels: list[dict], info: dict) -> set[str]:
    kinds = {a["id"] for a in accels if a["available"] and a["id"] != "cpu"}
    dev = str(info.get("device", ""))
    if dev and dev != "cpu":
        kinds.add(dev)
    return kinds


def _print_accelerators(cfg: CuratorConfig, info: dict) -> list[dict]:
    accels = _accelerators(cfg, info)
    t = Table(title="Accelerators")
    for col in ("", "ID", "Name", "Type", "Memory", "Status"):
        t.add_column(col, style="cyan" if col == "ID" else None)
    for a in accels:
        status = "[green]ready[/green]" if a["available"] else \
            "[yellow]" + escape(a["reason"] or "not available") + "[/yellow]"
        t.add_row("[bold green]*[/bold green]" if a["selected"] else "", a["id"],
                  escape(a["name"]), a["kind"], a["memory"], status)
    console.print(t)
    hints = [a for a in accels if not a["available"] and a["hint"]]
    if hints:
        console.print("[bold]To enable a device[/bold]")
        for a in hints:
            console.print(f"  {a['id']}: {escape(a['hint'])}", soft_wrap=True)
    if info.get("fallback_reason"):
        console.print(f"[yellow]{escape(str(info['fallback_reason']))}[/yellow]")
    return accels


def _print_workers(cfg: CuratorConfig, info: dict) -> None:
    res = resolve_workers(cfg, str(info.get("device") or ""))
    t = Table(title="Workers and threads")
    t.add_column("Setting", style="cyan")
    t.add_column("In use", justify="right")
    t.add_column("Set to", justify="right")
    # The config resolves "auto" scan threads when it is built, so compare with that value.
    scan_raw = 0 if cfg.scan_workers == auto_scan_workers() else cfg.scan_workers
    raw = {"scan_workers": scan_raw, "decode_workers": cfg.device.decode_workers,
           "torch_threads": cfg.device.torch_threads}
    for k, label in (("scan_workers", "Scan threads"), ("decode_workers", "Decode threads"),
                     ("torch_threads", "Torch CPU threads")):
        t.add_row(label, str(res[k]), "auto" if raw[k] <= 0 else str(raw[k]))
    console.print(t)


def _clip_weights_cached(cfg: CuratorConfig) -> bool | None:
    """True/False when the CLIP weights are in the local cache, None if unknown."""
    import os
    try:
        import open_clip
    except Exception:
        return None
    try:
        pc = open_clip.get_pretrained_cfg(cfg.taxonomy.clip_model_name,
                                          cfg.taxonomy.clip_pretrained) or {}
    except Exception:
        return None
    repo = str(pc.get("hf_hub", "")).strip("/")
    if not repo:
        return None
    home = os.environ.get("HF_HOME")
    hub = Path(os.environ.get("HF_HUB_CACHE")
               or (Path(home) / "hub" if home else
                   Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                   / "huggingface" / "hub"))
    snap = hub / ("models--" + repo.replace("/", "--")) / "snapshots"
    try:
        return snap.is_dir() and any(snap.iterdir())
    except OSError:
        return False


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------
@app.command()
def scan(
    target: Path = typer.Argument(..., help="Directory to scan (recursive)."),
    db: Path = typer.Option(DEFAULT_DB, "--db", help="SQLite state store path."),
    workers: int | None = typer.Option(None, "--workers", "-w", min=0, max=MAX_WORKERS,
                                       help="Scan threads. 0 or omitted means auto."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    prune_missing: bool = typer.Option(False, "--prune-missing",
                                       help="Mark rows whose files vanished as missing."),
    watch: bool = typer.Option(False, "--watch",
                               help="Keep running and ingest new files as they appear."),
):
    """Scan a directory: EXIF, two-tier hashes, resumable upsert."""
    cfg = _config(db)
    if workers is not None:
        cfg.scan_workers = workers
    cfg.scan_workers = resolve_workers(cfg)["scan_workers"]  # 0 means auto
    if watch:
        watch_fn = _lazy(".scanner.watcher", "watch")
        console.print(f"[cyan]Watching {target} (Ctrl-C to stop)[/cyan]")
        try:
            _call_flex(watch_fn, target, db_path=cfg.db_path, cfg=cfg, config=cfg,
                       prune_missing=prune_missing, show_progress=not quiet)
        except KeyboardInterrupt:
            console.print("[yellow]Stopped watching.[/yellow]")
        return

    from .scanner import file_scanner as fs
    scan_library = getattr(fs, "scan_library", None)
    with console.status("[cyan]Scanning...[/cyan]"):
        if scan_library is not None:
            result = scan_library(target, config=cfg, db_path=cfg.db_path,
                                  show_progress=not quiet, prune_missing=prune_missing)
        else:
            scanner = fs.FileScanner(cfg)
            result = _call_flex(scanner.scan_directory, target, db_path=cfg.db_path,
                                show_progress=not quiet, prune_missing=prune_missing)
    t = Table(title="Scan result")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Files seen", str(result.scanned))
    t.add_row("New indexed", str(result.new))
    t.add_row("Skipped (known)", str(result.skipped_known))
    t.add_row("Duplicates", str(result.duplicates))
    t.add_row("Errors", str(result.errors))
    pruned = getattr(result, "missing", getattr(result, "pruned", getattr(result, "marked_missing", None)))
    if prune_missing and pruned is not None:
        t.add_row("Marked missing", str(pruned))
    t.add_row("Duration", f"{result.duration_s:.2f}s")
    console.print(t)
    if result.errors_list:
        console.print("[red]First errors:[/red]\n  " +
                      "\n  ".join(result.errors_list[:10]))


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------
@app.command()
def infer(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    batch_size: int | None = typer.Option(None, "--batch-size", "-b",
                                          help="Override face/CLIP batch size."),
    device: str | None = typer.Option(
        None, "--device", "-d",
        help="auto|cuda|rocm|xpu|mps|directml|integrated|cpu. Default: settings file, then "
             "PCUR_DEVICE_DEVICE."),
    device_index: int | None = typer.Option(
        None, "--device-index", min=0, max=MAX_WORKERS,
        help="Which GPU to use when there is more than one."),
    decode_workers: int | None = typer.Option(
        None, "--decode-workers", min=0, max=MAX_WORKERS,
        help="Threads that decode photos ahead of the model. 0 or omitted means auto."),
    threads: int | None = typer.Option(
        None, "--threads", min=0, max=MAX_WORKERS,
        help="CPU threads for the model. 0 or omitted means auto."),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Max images."),
    paths: str | None = typer.Option(None, "--paths", help="Comma-separated paths (debug)."),
    retry_failed: bool = typer.Option(False, "--retry-failed",
                                      help="Reset failed rows to pending first."),
    reanalyze: bool | None = typer.Option(
        None, "--reanalyze/--no-reanalyze",
        help="Redo photos that were analyzed without an AI model. "
             "Default: on whenever the AI model is installed."),
    faces: bool = typer.Option(True, "--faces/--no-faces",
                               help="Find faces and blinks. --no-faces skips face analysis entirely."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
):
    """Analyze photos: quality, faces and what each photo shows."""
    from .inference.pipeline import InferencePipeline
    if device is not None:
        device = device.strip().lower()
    if device is not None and device not in _DEVICES:
        console.print(f"[red]Unknown device '{device}'. Choose one of: "
                      f"{', '.join(_DEVICES)}.[/red]")
        raise typer.Exit(2)
    cfg = _config(db)
    cfg.face.enabled = faces
    if device:
        cfg.device.device = device
    if device_index is not None:
        cfg.device.device_index = device_index
    if decode_workers is not None:
        cfg.device.decode_workers = decode_workers
    if threads is not None:
        cfg.device.torch_threads = threads
    if batch_size:
        cfg.device.face_batch_size = batch_size
        cfg.device.clip_batch_size = batch_size
    if retry_failed:
        with get_session(cfg.db_path) as s:
            n = PhotoRepository(s).reset_failed_inference()
        console.print(f"[dim]Reset {n} failed rows to pending.[/dim]")
    bs = detect_backend(cfg)
    info = describe_backends(cfg)
    res = resolve_workers(cfg, bs.device)
    cfg.device.decode_workers = res["decode_workers"]  # 0 means auto, now concrete
    cfg.device.torch_threads = res["torch_threads"]
    console.print(f"[dim]device={bs.device} fp16={bs.fp16} "
                 f"face={bs.face_backend} clip={bs.semantic_backend} "
                 f"decode_workers={res['decode_workers']} threads={res['torch_threads']}[/dim]")
    if info.get("fallback_reason"):
        console.print(f"[yellow]{escape(str(info['fallback_reason']))}[/yellow]")
    stub = bool(info.get("semantic_is_stub"))
    if stub:
        console.print(_stub_warning(info))
    do_reanalyze = (not stub) if reanalyze is None else reanalyze
    only = None
    if paths:
        only = [p.strip() for p in paths.split(",") if p.strip()]
    elif limit:
        with get_session(cfg.db_path) as s:
            repo = PhotoRepository(s)
            pend = repo.images_pending_inference(limit)
            only = [r.path for r in pend]
    pipeline = InferencePipeline(cfg, backend=bs)
    if do_reanalyze and "reanalyze_stub" not in inspect.signature(pipeline.run).parameters:
        mark = _lazy_optional(".inference.pipeline", "mark_stub_rows_for_reanalysis")
        if mark is not None:
            with get_session(cfg.db_path) as s:
                n = mark(s)
            if n:
                console.print(f"[dim]Queued {n} photos analyzed without a model.[/dim]")
    result = _call_flex(pipeline.run, db_path=cfg.db_path, only_paths=only,
                        show_progress=not quiet, reanalyze_stub=do_reanalyze)
    t = Table(title="Inference result")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Processed", str(result.processed))
    t.add_row("Faces detected", str(result.faces_total))
    t.add_row("Quality computed", str(result.quality_computed))
    t.add_row("Semantic computed", str(result.semantic_computed))
    t.add_row("Failures", str(result.failures))
    t.add_row("Duration", f"{result.duration_s:.2f}s")
    console.print(t)
    if stub and info.get("forced_by_env"):
        console.print("[yellow]These results came from the placeholder classifier because "
                      "PCUR_BACKEND is set. Run [bold]unset PCUR_BACKEND[/bold] and run infer "
                      "again.[/yellow]")
    elif stub:
        console.print("[yellow]These results came from the placeholder classifier. "
                      f"Install the AI model ({escape(ml_install_cmd())}) and run "
                      "infer again.[/yellow]")


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------
@app.command()
def cluster(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    window_seconds: int | None = typer.Option(None, "--window-seconds"),
    similarity_threshold: float | None = typer.Option(None, "--similarity-threshold", "-s"),
):
    """Cluster burst shots and select the best shot in each group."""
    cfg = _config(db)
    planner = Planner(cfg, db_path=cfg.db_path)
    n = planner.cluster(window_seconds, similarity_threshold)
    console.print(f"[green]Clustered {n} burst groups[/green] "
                  f"(window={cfg.burst.window_seconds}s, "
                  f"sim>={cfg.burst.similarity_threshold}). "
                  "`plan` and `apply` reuse these bursts.")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
@app.command()
def plan(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    output_dir: Path = typer.Option(Path("Organized"), "--output-dir", "-o",
                                    help="Organize root (Option 1)."),
    mode: str = typer.Option("category_first", "--mode",
                             help="category_first (Option 1) | in_place (Option 2)."),
    operation: str = typer.Option("move", "--operation",
                                  help="move|copy|hardlink|symlink."),
    report: Path | None = typer.Option(None, "--report", "-r",
                                          help="Write an HTML audit report here."),
    plan_id: str | None = typer.Option(None, "--plan-id"),
    exclude_duplicates: bool = typer.Option(False, "--exclude-duplicates"),
    recluster: bool = typer.Option(False, "--recluster",
                                   help="Recompute bursts instead of reusing them."),
    window_seconds: int | None = typer.Option(None, "--window-seconds"),
    similarity_threshold: float | None = typer.Option(None, "--similarity-threshold", "-s"),
):
    """Build and persist a dry-run routing plan and print a summary."""
    cfg = _config(db)
    planner = Planner(cfg, db_path=cfg.db_path)
    p = planner.build_plan(organize_root=output_dir, operation=operation,
                           mode=_mode(mode), plan_id=plan_id,
                           include_duplicates=not exclude_duplicates,
                           recluster=recluster or None,
                           window_seconds=window_seconds,
                           similarity_threshold=similarity_threshold)
    p.dry_run = True
    planner.print_summary(p)
    _print_diff(planner.dry_run(p))
    if report:
        rp = planner.render_html_report(p, report, extra={
            "Mode": mode, "Operation": operation,
            "Organize root": str(output_dir), "DB": str(cfg.db_path),
        })
        console.print(f"[green]Report written:[/green] {rp}")
    console.print(f"\n[dim]Plan ID: {p.plan_id} (saved in the state store)[/dim]")
    console.print("[dim]Review the report, then run:[/dim] "
                  f"photocure apply --db {cfg.db_path} --plan-id {p.plan_id}")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
@app.command()
def apply(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    plan_id: str | None = typer.Option(None, "--plan-id",
                                          help="Apply this persisted plan exactly as reviewed."),
    operation: str | None = typer.Option(None, "--operation",
                                         help="Override: move|copy|hardlink|symlink."),
    output_dir: Path = typer.Option(Path("Organized"), "--output-dir", "-o",
                                    help="Only used when no --plan-id is given."),
    mode: str = typer.Option("category_first", "--mode",
                             help="Only used when no --plan-id is given."),
    exclude_duplicates: bool = typer.Option(False, "--exclude-duplicates",
                                            help="Only used when no --plan-id is given."),
    window_seconds: int | None = typer.Option(None, "--window-seconds"),
    similarity_threshold: float | None = typer.Option(None, "--similarity-threshold", "-s"),
    only_category: list[str] | None = typer.Option(
        None, "--only-category", help="Apply only these categories (repeatable)."),
    trash: bool = typer.Option(False, "--trash",
                               help="Stage approved junk in <root>/.trash/<txn>/. "
                                    "Unapproved junk stays in the junk review folder."),
    approve_deletes: bool = typer.Option(False, "--approve-deletes",
                                         help="With --trash, approve every junk photo in the plan first."),
    force: bool = typer.Option(False, "--force",
                               help="Apply even if the plan is stale or already applied."),
    yes: bool = typer.Option(False, "--yes", "-y",
                             help="Skip the confirmation prompt."),
):
    """Execute a persisted plan (records a transaction for later undo)."""
    cfg = _config(db)
    planner = Planner(cfg, db_path=cfg.db_path)
    if plan_id:
        try:
            p = planner.load_plan(plan_id)
        except KeyError:
            console.print(f"[red]Unknown plan id:[/red] {plan_id}")
            raise typer.Exit(1) from None
        if exclude_duplicates:
            console.print("[yellow]--exclude-duplicates is ignored with --plan-id; "
                          "the plan was fixed when it was built.[/yellow]")
    else:
        p = planner.build_plan(organize_root=output_dir, operation=operation or "move",
                               mode=_mode(mode),
                               include_duplicates=not exclude_duplicates,
                               window_seconds=window_seconds,
                               similarity_threshold=similarity_threshold)
    if p.total == 0:
        console.print("[yellow]Nothing to apply (no inferred images).[/yellow]")
        return
    op = operation or p.operation
    console.print(f"About to [bold]{op}[/bold] [bold]{p.total}[/bold] files "
                  f"into [bold]{p.organize_root}[/bold] (plan {p.plan_id})"
                  + (f", only {', '.join(only_category)}" if only_category else "")
                  + (", junk staged in trash" if trash else "") + ".")
    diff = planner.dry_run(p)
    if not diff.space_ok:
        console.print("[red]Not enough free space on the target volume for the "
                      "copies this plan needs.[/red]")
        if not force:
            raise typer.Exit(1)
    if not yes and not typer.confirm("Proceed?"):
        console.print("[yellow]Aborted.[/yellow]")
        return
    if trash and approve_deletes:
        junk_ids = [a.image_id for a in p.actions
                    if a.image_id and getattr(a.category, "is_junk", False)]
        with get_session(cfg.db_path) as s:
            for rec in s.exec(select(ImageRecord).where(col(ImageRecord.id).in_(junk_ids))).all():
                rec.delete_approved = True
    try:
        rep = planner.execute_plan(p, only_category=only_category, trash=trash,
                                   force=force, operation=operation)
    except PlanDriftError as e:
        console.print(f"[red]Refusing to apply:[/red] {e}")
        raise typer.Exit(1) from None
    t = Table(title=f"Transaction {rep.transaction_id}")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Executed", str(rep.executed))
    t.add_row("Skipped (filtered)", str(rep.skipped))
    t.add_row("Collisions (renamed)", str(rep.collisions))
    t.add_row("Failures", str(rep.failures))
    if trash:
        t.add_row("Junk kept (not approved)", str(rep.unapproved_junk))
    t.add_row("Duration", f"{rep.duration_s:.2f}s")
    console.print(t)
    if trash and rep.unapproved_junk and not approve_deletes:
        console.print("[yellow]Junk that is not approved stays in the review folder. "
                      "Approve it in the app, or pass --approve-deletes.[/yellow]")
    if rep.errors:
        console.print("[red]Errors:[/red]\n  " + "\n  ".join(rep.errors[:20]))
    console.print(f"[green]Done.[/green] To undo: "
                  f"photocure undo --db {cfg.db_path} "
                  f"--transaction-id {rep.transaction_id}")


# ---------------------------------------------------------------------------
# undo / verify / repair / trash
# ---------------------------------------------------------------------------
@app.command()
def undo(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    transaction_id: str | None = typer.Option(None, "--transaction-id", "-t",
                                                 help="Defaults to the latest not yet undone."),
    no_verify: bool = typer.Option(False, "--no-verify",
                                   help="Skip the pre-flight SHA-256 check."),
):
    """Roll back a committed transaction (reverse-chronological)."""
    cfg = _config(db)
    planner = Planner(cfg, db_path=cfg.db_path)
    rep = planner.undo(transaction_id, verify=not no_verify)
    if not rep.transaction_id:
        console.print("[yellow]No transactions found.[/yellow]")
        return
    t = Table(title=f"Rollback {rep.transaction_id}")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Reversed", str(rep.reversed))
    t.add_row("Skipped", str(rep.skipped))
    t.add_row("Failures", str(rep.failures))
    t.add_row("Duration", f"{rep.duration_s:.2f}s")
    console.print(t)
    if rep.errors:
        console.print("[red]Errors:[/red]\n  " + "\n  ".join(rep.errors[:20]))


@app.command("verify-txn")
def verify_txn(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    transaction_id: str | None = typer.Option(None, "--transaction-id", "-t",
                                                 help="Defaults to the latest."),
):
    """Re-hash every committed target and report drift."""
    cfg = _config(db)
    res = Planner(cfg, db_path=cfg.db_path).verify_transaction(transaction_id)
    if not res["transaction_id"]:
        console.print("[yellow]No transactions found.[/yellow]")
        return
    t = Table(title=f"Verify {res['transaction_id']}")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Checked", str(res["checked"]))
    t.add_row("OK", str(res["ok"]))
    t.add_row("Drifted", str(len(res["drift"])))
    t.add_row("Missing", str(len(res["missing"])))
    console.print(t)
    for d in res["drift"][:20]:
        console.print(f"[red]drift[/red] {d['path']}")
    for m in res["missing"][:20]:
        console.print(f"[red]missing[/red] {m}")
    if res["drift"] or res["missing"]:
        raise typer.Exit(2)


@app.command()
def repair(db: Path = typer.Option(DEFAULT_DB, "--db")):
    """Fix images.path from the action ledger (older stores)."""
    cfg = _config(db)
    n = Planner(cfg, db_path=cfg.db_path).repair_paths()
    console.print(f"[green]Repaired {n} image paths.[/green]")


@app.command("trash-purge")
def trash_purge(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    output_dir: Path = typer.Option(Path("Organized"), "--output-dir", "-o"),
    older_than_days: float | None = typer.Option(
        None, "--older-than-days", help="Default PCUR_TRASH_RETENTION_DAYS (30)."),
    yes: bool = typer.Option(False, "--yes", "-y"),
):
    """Permanently delete trash staging older than the retention window."""
    cfg = _config(db)
    days = cfg.trash_retention_days if older_than_days is None else older_than_days
    if not yes and not typer.confirm(
            f"Permanently delete trash under {output_dir} older than {days} days?"):
        console.print("[yellow]Aborted.[/yellow]")
        return
    res = Planner(cfg, db_path=cfg.db_path).purge_trash(output_dir, days)
    console.print(f"[green]Purged {len(res['removed_transactions'])} transactions, "
                  f"{res['files']} files, {_human(res['bytes'])}.[/green]")


@app.command()
def transactions(db: Path = typer.Option(DEFAULT_DB, "--db")):
    """List ledger transactions, newest first."""
    cfg = _config(db)
    rows = Planner(cfg, db_path=cfg.db_path).list_transactions()
    t = Table(title="Transactions")
    t.add_column("Transaction", style="cyan")
    t.add_column("Status counts")
    for r in rows:
        t.add_row(r["transaction_id"],
                  ", ".join(f"{k}={v}" for k, v in sorted(r["counts"].items())))
    console.print(t)


# ---------------------------------------------------------------------------
# features implemented in other modules (thin wrappers)
# ---------------------------------------------------------------------------
@app.command()
def similar(
    path: Path = typer.Argument(..., help="Image path (as indexed)."),
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    k: int = typer.Option(10, "-k", "--top", help="Number of neighbours."),
):
    """Find images similar to one image (CLIP embedding search)."""
    find_similar = _lazy(".clustering.similarity", "find_similar")
    cfg = _config(db)
    with get_session(cfg.db_path) as s:
        repo = PhotoRepository(s)
        img = repo.get_by_path(str(path)) or repo.get_by_path(str(Path(path).resolve()))
        if img is None:
            console.print(f"[red]Not indexed:[/red] {path}")
            raise typer.Exit(1)
        try:
            hits = find_similar(s, img.id, k)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None
        t = Table(title=f"Similar to {img.filename}")
        t.add_column("Score", justify="right")
        t.add_column("Path")
        for image_id, score in hits:
            other = repo.get_by_id(image_id)
            t.add_row(f"{score:.3f}", other.path if other else str(image_id))
        console.print(t)


@app.command()
def search(
    query: str = typer.Argument(..., help="What to look for, e.g. 'dog on the beach'."),
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    limit: int = typer.Option(10, "-n", "--top", min=1, max=300, help="Number of results."),
    duplicates: bool = typer.Option(False, "--duplicates",
                                    help="Include duplicates and burst alternates."),
):
    """Find photos by describing them in words (CLIP text against stored embeddings)."""
    from . import search as search_mod
    cfg = _config(db)
    encoder = search_mod.default_encoder(cfg)
    try:
        try:
            vec = encoder.encode(query)
        except (ValueError, search_mod.SearchUnavailable) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None
        with get_session(cfg.db_path) as s:
            hits = search_mod.search_embeddings(s, vec, limit, include_duplicates=duplicates)
            repo = PhotoRepository(s)
            t = Table(title=f"Photos matching: {query}")
            t.add_column("Match", justify="right")
            t.add_column("Path", overflow="fold")
            for image_id, score in hits:
                other = repo.get_by_id(image_id)
                t.add_row(f"{score:.3f}", other.path if other else str(image_id))
            console.print(t)
    finally:
        encoder.close()


@app.command()
def highlights(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    year: int | None = typer.Option(None, "--year", min=1, max=9999,
                                    help="Default: the newest year with photos."),
    top: int = typer.Option(50, "--top", "-n", min=1, max=500, help="How many photos."),
    out: Path = typer.Option(Path("highlights"), "--out", help="Folder for the page and copies."),
    copy: bool = typer.Option(False, "--copy", help="Also copy the originals into <out>/photos."),
):
    """Export the best photos of a year as a self-contained page, optionally with copies."""
    from . import highlights as hl
    from .features import load_features
    cfg = _config(db)
    with get_session(cfg.db_path) as s:
        years = hl.years_available(s)
        chosen = year if year is not None else (years[0][0] if years else None)
        ranked = (hl.top_of_year(s, hl.HighlightWeights.from_features(load_features(cfg.db_path)),
                                 chosen, top) if chosen is not None else [])
        if not ranked:
            console.print(f"[red]No scored photos{f' from {chosen}' if chosen else ''}. "
                          "Analyze the library first.[/red]")
            raise typer.Exit(1)
        out.mkdir(parents=True, exist_ok=True)
        page = out / f"highlights-{chosen}.html"
        page.write_text(hl.export_html(s, ranked, f"Best of {chosen}"), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {page} ({len(ranked)} photos)")
        if copy:
            n = hl.copy_originals(s, ranked, out / "photos")
            console.print(f"[green]Copied[/green] {n} originals to {out / 'photos'}")


@app.command()
def keywords(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    remove: bool = typer.Option(False, "--remove", help="Remove Photocure keywords instead."),
):
    """Write (or remove) category, highlight and event keywords in XMP sidecars.

    Only photocure-* keywords are touched. Ratings and other programs' keywords are kept, and a
    sidecar edited outside Photocure is skipped. Writing creates a sidecar next to every photo.
    """
    from . import keywords as kw
    cfg = _config(db)
    try:
        res = kw.write_all(cfg.db_path, cfg, keywords=not remove)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    t = Table(title="Keywords removed" if remove else "Keywords written")
    t.add_column("Outcome")
    t.add_column("Photos", justify="right")
    labels = {"written": "Sidecars updated", "kept_foreign_rating": "Updated, existing rating kept",
              "edited_elsewhere": "Skipped, edited by another program", "skipped": "Skipped",
              "no_sidecar": "No sidecar", "nothing_to_write": "Nothing to write", "errors": "Errors"}
    for key, label in labels.items():
        if res.get(key):
            t.add_row(label, str(res[key]))
    console.print(t)


@app.command()
def events(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    gap_hours: float = typer.Option(2.0, "--gap-hours"),
    gps_km: float | None = typer.Option(None, "--gps-km"),
):
    """Group images into events by time gap (and optionally GPS distance)."""
    cluster_events = _lazy(".clustering.events", "cluster_events")
    cfg = _config(db)
    with get_session(cfg.db_path) as s:
        n = cluster_events(s, gap_hours=gap_hours, gps_km=gps_km)
    console.print(f"[green]Grouped images into {n} events.[/green]")


@app.command("export-onnx")
def export_onnx(
    output_dir: Path = typer.Option(..., "--output-dir", "-o"),
    db: Path = typer.Option(DEFAULT_DB, "--db"),
):
    """Export the CLIP image tower and face detector to ONNX."""
    export_models = _lazy(".inference.onnx_export", "export_models")
    cfg = _config(db)
    try:
        export_models(output_dir, cfg)
    except (RuntimeError, ImportError) as e:
        console.print(f"[red]ONNX export failed:[/red] {e}")
        raise typer.Exit(1) from None
    console.print(f"[green]Exported models to {output_dir}[/green]")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
@app.command()
def devices(db: Path = typer.Option(DEFAULT_DB, "--db")):
    """List every GPU and CPU the app can use, which one is chosen, and the worker counts."""
    reset_backend_cache()
    cfg = _config_no_init(db)
    info = describe_backends(cfg)
    _print_accelerators(cfg, info)
    req = info.get("requested") or cfg.device.device
    console.print(f"Requested device: [cyan]{escape(str(req))}[/cyan], "
                  f"using: [cyan]{escape(str(info.get('device')))}[/cyan], "
                  f"device index: {cfg.device.device_index}")
    _print_workers(cfg, info)
    accels = _accelerators(cfg, info)
    missing = [k for k in _detect_hardware() if k not in _usable_kinds(accels, info)]
    for kind in missing:
        console.print(Panel("\n".join(escape(x) for x in _install_steps(kind)),
                            title=f"Enable your {kind.upper()} GPU", border_style="yellow"))


@app.command()
def doctor(db: Path = typer.Option(DEFAULT_DB, "--db")):
    """Check that the AI model, GPU and face detector are ready, and say how to fix gaps."""
    reset_backend_cache()
    cfg = _config_no_init(db)
    info = describe_backends(cfg)
    ver = {
        "torch": _pkg_version("torch"),
        "open_clip": _pkg_version("open_clip_torch", "open-clip-torch"),
        "mediapipe": _pkg_version("mediapipe"),
    }
    weights = _clip_weights_cached(cfg)
    stub_rows = None
    if cfg.db_path.exists():
        init_db(cfg.db_path)
        with get_session(cfg.db_path) as s:
            stub_rows = PhotoRepository(s).count_stub_rows()

    t = Table(title="Photocure health check")
    t.add_column("Check", style="cyan")
    t.add_column("Status")
    t.add_row("Device", str(info.get("device")))
    sem = str(info.get("semantic_backend"))
    t.add_row("Photo recognition", sem if not info.get("semantic_is_stub")
              else "[yellow]placeholder (no AI model)[/yellow]")
    face = str(info.get("face_backend"))
    t.add_row("Face detection", face if not info.get("face_is_fallback")
              else f"[yellow]{face} (no face detector)[/yellow]")
    t.add_row("Blink detection", "ready" if info.get("blink_available")
              else "[yellow]not installed (eye state unknown)[/yellow]")
    for name, v in ver.items():
        t.add_row(name, v if v else "[yellow]missing[/yellow]")
    t.add_row("CLIP weights cached",
              "unknown" if weights is None else ("yes" if weights else "[yellow]no[/yellow]"))
    if stub_rows is not None:
        t.add_row("Photos analyzed without a model",
                  f"[yellow]{stub_rows}[/yellow]" if stub_rows else "0")
    console.print(t)
    accels = _print_accelerators(cfg, info)
    _print_workers(cfg, info)
    usable = _usable_kinds(accels, info)
    hardware = _detect_hardware()

    fixes: list[str] = []
    for kind in hardware:
        if kind not in usable:
            fixes.append(f"A {kind.upper()} GPU looks present but is not usable by torch. "
                         "Install a matching build:\n    " + "\n    ".join(_install_steps(kind)))
    if info.get("forced_by_env"):
        fixes.append("The AI model is installed but switched off by PCUR_BACKEND. "
                     "Run: unset PCUR_BACKEND")
    elif (not ver["torch"] or not ver["open_clip"] or info.get("semantic_is_stub")) \
            and not any(k not in usable for k in hardware):
        fixes.append(f"Install the AI model: {ml_install_cmd()}")
    if not ver["mediapipe"]:
        faces_cmd = pip_install_cmd('"photocure[faces]"')
        fixes.append(f"Blink detection (optional, eyes stay unmeasured without it): {faces_cmd}")
    if ver["open_clip"] and weights is False:
        fixes.append("The model weights download on first use (about 350 MB), "
                     "so connect to the internet once.")
    if stub_rows and not info.get("semantic_is_stub"):
        fixes.append("Redo photos that were analyzed without a model: "
                     "photocure infer --reanalyze")
    if ver["torch"] and info.get("device") == "cpu":
        fixes.append("No GPU was found, so analysis runs on the CPU. "
                     "That works but is slower.")
    if fixes:
        console.print(Panel("\n".join(f"- {escape(f)}" for f in fixes), title="How to fix",
                            border_style="yellow"))
    else:
        console.print("[green]Everything is ready.[/green]")


@app.command()
def learn(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    rebuild: bool = typer.Option(True, "--rebuild/--no-rebuild",
                                 help="Refresh each photo's category afterwards."),
):
    """Re-sort photos using the categories you chose by hand."""
    cfg = _config(db)
    planner = Planner(cfg)
    try:
        r = planner.learn_from_corrections(rebuild=rebuild)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    if not r.get("examples"):
        console.print("[yellow]No corrections found yet. Move a few photos to the right "
                      "category in the dashboard, then run this again.[/yellow]")
        return
    t = Table(title="Learned from your corrections")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Corrected photos used", str(r.get("examples", 0)))
    t.add_row("Categories learned", str(r.get("classes_used", 0)))
    t.add_row("Photos re-scored", str(r.get("updated", 0)))
    t.add_row("Uncertain before", str(r.get("uncertain_before", 0)))
    t.add_row("Uncertain after", str(r.get("uncertain_after", 0)))
    t.add_row("Moved out of Uncertain", str(r.get("moved_out_of_uncertain", 0)))
    console.print(t)


@app.command()
def stats(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
):
    """Show state-store statistics."""
    cfg = _config(db)
    with get_session(cfg.db_path) as s:
        st = PhotoRepository(s).stats()
    t = Table(title="State store")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    for k, v in st.items():
        t.add_row(k.replace("_", " ").title(), str(v))
    console.print(t)
    console.print(f"[dim]DB: {cfg.db_path}[/dim]")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
@app.command()
def serve(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    host: str = typer.Option("127.0.0.1", "--host", envvar="PCUR_HOST"),
    port: int = typer.Option(8080, "--port", "-p", envvar="PCUR_PORT"),
    reload: bool = typer.Option(False, "--reload"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the browser automatically."),
    token: str | None = typer.Option(None, "--token", envvar="PCUR_AUTH_TOKEN",
                                     help="Access token. Generated and stored when not given."),
):
    """Launch the web review dashboard."""
    import os
    import uvicorn

    from . import setup_cmd as sc
    from .auth import AuthManager, new_otc
    cfg = _config(db)
    os.environ["PCUR_DB_PATH"] = str(cfg.db_path.resolve())
    free = sc.pick_port(host, port)
    if free != port:
        console.print(f"[yellow]Port {port} is busy, using {free}.[/yellow]")
    os.environ["PCUR_HOST"], os.environ["PCUR_PORT"] = host, str(free)
    if isinstance(token, str) and token:  # a direct call that omits it leaves Typer's option object here
        os.environ["PCUR_AUTH_TOKEN"] = token
    auth = AuthManager(cfg.cache_dir)  # creates and stores the token when none exists
    otc = new_otc()
    os.environ["PCUR_OTC"] = otc  # read once by the app, never written to disk
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(f"[bold yellow]Initial authentication token:[/bold yellow] {auth.token}")
    if open_browser:
        sc.open_browser_when_ready(host, free, code=otc)
    else:
        console.print(f"Open [cyan]http://{shown}:{free}/#code={otc}[/cyan] (single use), "
                      "or paste the code into the sign-in box.")
    console.print(f"[cyan]Serving review dashboard on http://{host}:{free}/ "
                  f"(db={cfg.db_path})[/cyan]")
    uvicorn.run("photocure.ui.app:app", host=host, port=free,
                reload=reload, factory=False,
                log_level="info")


@app.command(name="start")
def start(
    db: Path = typer.Option(DEFAULT_DB, "--db"),
    host: str = typer.Option("127.0.0.1", "--host", envvar="PCUR_HOST"),
    port: int = typer.Option(8080, "--port", "-p", envvar="PCUR_PORT"),
    reload: bool = typer.Option(False, "--reload"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the browser automatically."),
    token: str | None = typer.Option(None, "--token", envvar="PCUR_AUTH_TOKEN"),
):
    """Start the web review dashboard and open the browser (alias for 'serve')."""
    serve(db=db, host=host, port=port, reload=reload, open_browser=open_browser, token=token)


@app.command("mcp")
def mcp_cmd(
    db: Path = typer.Option(DEFAULT_DB, "--db", help="Library database. Use an absolute path."),
):
    """Run an MCP server on stdio so an AI agent can search your photos and make clean copies."""
    try:
        from .mcp_server import run_stdio
    except ImportError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    try:
        run_stdio(db)
    except ImportError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None


@app.command()
def setup(
    db: Path = typer.Option(DEFAULT_DB, "--db", help="Database to use for the web UI."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port", "-p"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before downloading."),
    faces: bool = typer.Option(True, "--faces/--no-faces", help="Also install blink detection (MediaPipe)."),
    serve_ui: bool = typer.Option(True, "--serve/--no-serve", help="Start the web UI when done."),
    open_ui: bool = typer.Option(True, "--open/--no-open", help="Open the browser automatically."),
    verify: bool | None = typer.Option(None, "--verify/--no-verify",
                                       help="Load the model to check it. Default: only when something changed."),
    reinstall: bool = typer.Option(False, "--reinstall", help="Reinstall PyTorch even if present."),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Ignore any GPU and install the CPU build."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen and stop."),
):
    """Install the AI model for your hardware, check it works, and open the web UI."""
    import os

    from . import setup_cmd as sc

    kind = "cpu" if cpu_only else sc.detect_kind()
    console.print(f"[bold]Photocure setup[/bold]  hardware: [cyan]{sc.KIND_LABEL[kind]}[/cyan]")
    # Leftover test settings in the shell would otherwise keep the model off after a good install.
    forced = os.environ.get("PCUR_BACKEND", "").strip().lower()
    if forced in ("stub", "off"):
        console.print(f"[yellow]PCUR_BACKEND={forced} in your shell turns the AI model off. "
                      "Ignoring it for this run. To remove it for good, run: "
                      "[bold]unset PCUR_BACKEND[/bold][/yellow]")
        os.environ.pop("PCUR_BACKEND", None)
    if kind != "cpu" and os.environ.get("PCUR_DEVICE_DEVICE", "").strip().lower() == "cpu":
        console.print("[yellow]PCUR_DEVICE_DEVICE=cpu in your shell forces the CPU. Ignoring it for "
                      "this run so the GPU can be used. To remove it for good, run: "
                      "[bold]unset PCUR_DEVICE_DEVICE[/bold] (or pass --cpu-only to keep the CPU)[/yellow]")
        os.environ.pop("PCUR_DEVICE_DEVICE", None)

    state = sc.probe_state()
    cfg = _config_no_init(db)
    # A blink-detection package that failed to install is not retried on every restart.
    faces_marker = Path(cfg.cache_dir) / "setup_faces_failed"
    if faces and faces_marker.exists() and not reinstall:
        console.print("[dim]Skipping blink detection: it failed to install earlier. "
                      "Use --reinstall to try again.[/dim]")
        faces = False
    steps = sc.plan_steps(kind, state, faces=faces, force=reinstall)
    weights = _clip_weights_cached(cfg)
    already = [f"PyTorch {state['version']}" if state.get("torch") and not sc.torch_needs_reinstall(kind, state) else None,
               f"OpenCLIP {state['open_clip']}" if state.get("open_clip") else None,
               "MediaPipe" if state.get("mediapipe") else None]
    already = [a for a in already if a]
    if already:
        console.print("Already installed: " + ", ".join(already))

    if steps:
        t = Table(title="To do")
        t.add_column("Step"); t.add_column("Download", justify="right"); t.add_column("Notes")
        for st in steps:
            t.add_row(st.title, f"~{st.size_mb} MB", escape(st.note) + (" (optional)" if st.optional else ""))
        if weights is not True:
            t.add_row("Download the classifier weights on first load", f"~{sc.CLIP_WEIGHTS_MB} MB", "needs internet once")
        console.print(t)
        console.print(f"Total download: about {sc.total_download_mb(steps, weights is True) / 1000:.1f} GB")
    elif weights is not True:
        console.print(f"Libraries are ready. The classifier weights (~{sc.CLIP_WEIGHTS_MB} MB) "
                      "download on first load.")
    else:
        console.print("[green]Everything is installed already.[/green]")

    if dry_run:
        for st in steps:
            console.print(f"[dim]$ {escape(shlex.join(st.cmd))}[/dim]")
        console.print("[dim]Dry run: nothing was installed.[/dim]")
        raise typer.Exit(0)

    if steps and not yes and not typer.confirm("Continue?", default=True):
        raise typer.Exit(1)

    import importlib
    for st in steps:
        console.print(f"\n[bold cyan]{st.title}[/bold cyan]")
        if not sc.run_step(st):
            if st.optional:
                console.print(f"[yellow]Skipped: {escape(st.title)} did not install. "
                              "Faces are still found; eye state stays unknown.[/yellow]")
                try:
                    faces_marker.parent.mkdir(parents=True, exist_ok=True)
                    faces_marker.write_text("failed\n")
                except OSError:
                    pass
                continue
            console.print(f"[red]Failed: {escape(st.title)}.[/red] Run the command above by hand to see why.")
            raise typer.Exit(1)
    importlib.invalidate_caches()
    sc.activate_user_site()

    # Restarts of an already working install skip the slow model load.
    do_verify = verify if verify is not None else bool(steps) or weights is not True or reinstall
    if do_verify:
        console.print("\n[bold cyan]Checking the model[/bold cyan] (downloads the weights if needed)")
        res = sc.verify_model()
        if not res.get("ok"):
            console.print(f"[red]The classifier did not load.[/red] {escape(str(res.get('error', '')))}")
            console.print("Run [bold]photocure doctor[/bold] for details.")
            raise typer.Exit(1)
        console.print(f"[green]Ready.[/green] {res.get('model')} running on [bold]{res.get('device')}[/bold] "
                      f"(first load took {res.get('seconds')}s)")
    else:
        console.print("[green]Ready.[/green] The model is installed and cached.")

    if not serve_ui:
        console.print("\nDone. Start the web UI any time with: [bold]photocure serve[/bold]")
        raise typer.Exit(0)
    free = sc.pick_port(host, port)
    if free != port:
        console.print(f"[yellow]Port {port} is busy, using {free}.[/yellow]")
    serve(db=db, host=host, port=free, reload=False, open_browser=open_ui, token=None)


@app.command()
def stop(
    port: int = typer.Option(8080, "--port", "-p", envvar="PCUR_PORT"),
    force: bool = typer.Option(False, "--force", help="Send SIGKILL after 5 seconds."),
):
    """Stop a running Photocure server on the given port."""
    import os
    import signal
    import time

    from .lifecycle import pid_file_path
    pid_file = pid_file_path(port)

    if not pid_file.exists():
        console.print(f"[yellow]No active Photocure server found on port {port}.[/yellow]")
        raise typer.Exit(0)

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        console.print(f"[red]Invalid PID file: {pid_file}[/red]")
        raise typer.Exit(1)

    try:
        os.kill(pid, 0)
    except OSError:
        console.print(f"[yellow]Process {pid} not found.[/yellow]")
        pid_file.unlink(missing_ok=True)
        raise typer.Exit(0)

    from .lifecycle import is_photocure_server, process_cmdline
    if not is_photocure_server(process_cmdline(pid)):
        console.print(f"[red]PID {pid} is not a Photocure server (stale PID file?). Nothing was stopped.[/red]")
        pid_file.unlink(missing_ok=True)
        raise typer.Exit(1)

    console.print(f"[cyan]Terminating Photocure (PID {pid})...[/cyan]")
    os.kill(pid, signal.SIGTERM)

    for _ in range(10):  # 5 seconds with 0.5 Hz poll.
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
        except OSError:
            console.print("[green]Process stopped.[/green]")
            pid_file.unlink(missing_ok=True)
            raise typer.Exit(0)

    if force:
        console.print("[yellow]Process did not stop. Sending SIGKILL...[/yellow]")
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        pid_file.unlink(missing_ok=True)
        console.print("[green]Process killed.[/green]")
    else:
        console.print("[red]Process did not stop. Try --force.[/red]")
        raise typer.Exit(1)


@app.command()
def kill(
    port: int = typer.Option(8080, "--port", "-p", envvar="PCUR_PORT"),
):
    """Kill a running Photocure server on the given port (alias: stop --force)."""
    stop(port=port, force=True)


db_app = typer.Typer(help="Database maintenance.", no_args_is_help=True)
app.add_typer(db_app, name="db")


@db_app.command("restore")
def db_restore(
    snapshot: Path = typer.Argument(..., exists=True, dir_okay=False, help="A backups/*.bak.db file."),
    db: Path = typer.Option(DEFAULT_DB, "--db"),
):
    """Replace the database with a snapshot taken before an apply."""
    import contextlib
    import sqlite3

    from .db.session import close_engine
    from .lifecycle import running_servers

    live = running_servers()
    if live:
        console.print(f"[red]Photocure is running on port {live[0]}. Run 'photocure stop' first.[/red]")
        raise typer.Exit(1)
    with contextlib.closing(sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)) as con:
        if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            console.print("[red]That snapshot failed its integrity check. Nothing was changed.[/red]")
            raise typer.Exit(1)
    close_engine(db)
    if db.exists():
        keep = db.with_name(db.name + ".pre-restore")
        shutil.copy2(db, keep)
        console.print(f"Current database kept as {keep}")
    with contextlib.closing(sqlite3.connect(str(snapshot))) as src, \
            contextlib.closing(sqlite3.connect(str(db))) as dst:
        src.backup(dst)
    console.print(f"[green]Restored {db} from {snapshot}.[/green]")


@app.command()
def geocode(db: Path = typer.Option(DEFAULT_DB, "--db"),
            batch: int = typer.Option(500, "--batch", min=1, help="Positions looked up at a time.")):
    """Fill in place names (city, region, country) for photos that have GPS. Works offline."""
    from .scanner import geo
    if not geo.available():
        cmd = pip_install_cmd('"photocure[geo]"')
        console.print(f"[yellow]Place names need the geo extra:[/yellow] {escape(cmd)}")
        raise typer.Exit(1)
    cfg = _config(db)
    with get_session(cfg.db_path) as s:
        n = geo.geocode_pending(s, batch)
    console.print(f"Found a place for [bold]{n}[/bold] photo{'' if n == 1 else 's'}.")


bench_app = typer.Typer(help="Calibration and timing. Quote only the numbers these print.", no_args_is_help=True)
app.add_typer(bench_app, name="bench")


@bench_app.command("blinks")
def bench_blinks(
    labels: Path = typer.Option(Path("tests/fixtures/faces/labels.csv"), "--labels",
                                help="CSV with columns path,label (open or closed); paths are relative to it."),
):
    """Precision, recall and a threshold sweep for eye-closed scores against your own labels."""
    from . import bench
    if not labels.exists():
        console.print(f"[yellow]No labels at {labels}. Nothing to calibrate.[/yellow] "
                      "Pass --labels with a CSV of path,label rows.")
        raise typer.Exit(0)
    items = bench.read_labels(labels)
    if not items:
        console.print("[yellow]The labels file has no usable rows (label must be open or closed).[/yellow]")
        raise typer.Exit(1)
    cfg = _config_no_init(DEFAULT_DB)
    scored, no_face, unreadable = bench.score_labelled_photos(items, cfg)
    measured = [x for x in scored if x[1] is not None]
    console.print(f"{len(items)} labelled photos: {len(measured)} measured, "
                  f"{len(scored) - len(measured)} eyes not measured, {no_face} without a face, "
                  f"{unreadable} unreadable.")
    if not measured:
        console.print("[yellow]No eye scores to evaluate. Is MediaPipe installed? "
                      "See: photocure doctor[/yellow]")
        raise typer.Exit(1)
    rows = bench.sweep(scored)
    t = Table(title="Closed-eye threshold sweep (score >= threshold means closed)")
    for c in ("threshold", "precision", "recall", "F1", "TP", "FP", "FN", "TN"):
        t.add_column(c, justify="right")
    fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    for r in rows:
        mark = "  <- closed_hi" if abs(r.threshold - cfg.face.closed_hi) < 1e-9 else ""
        t.add_row(f"{r.threshold:.2f}{mark}", fmt(r.precision), fmt(r.recall), fmt(r.f1),
                  str(r.tp), str(r.fp), str(r.fn), str(r.tn))
    console.print(t)
    best = bench.best_threshold(rows)
    if best:
        console.print(f"Best F1 on these labels: threshold {best.threshold:.2f} "
                      f"(F1 {best.f1:.2f}). The setting in use is closed_hi={cfg.face.closed_hi}; "
                      "change it with PCUR_FACE_CLOSED_HI.")


@bench_app.command("pipeline")
def bench_pipeline_cmd(
    folder: Path = typer.Argument(..., exists=True, file_okay=False),
    limit: int = typer.Option(50, "--limit", "-n", min=1, help="Photos to time."),
    device: str | None = typer.Option(None, "--device", "-d", help="auto|cuda|rocm|xpu|mps|cpu"),
):
    """Time each stage (decode, faces, landmarks, CLIP, quality, database) on a folder of photos."""
    from . import bench
    cfg = _config_no_init(DEFAULT_DB)
    if device:
        if device.strip().lower() not in _DEVICES:
            console.print(f"[red]Unknown device '{device}'. Choose one of: {', '.join(_DEVICES)}.[/red]")
            raise typer.Exit(2)
        cfg.device.device = device.strip().lower()
    res = bench.bench_pipeline(folder, cfg, limit)
    console.print(f"{res.photos} photos on [bold]{res.device}[/bold] "
                  f"(faces: {res.face_backend}, classifier: {res.semantic_backend})")
    t = Table(title="Time per photo (milliseconds)")
    for c in ("stage", "n", "mean", "median", "p95"):
        t.add_column(c, justify="left" if c == "stage" else "right")
    for stage, n, mean, med, p95 in res.summary():
        t.add_row(stage, str(n), f"{mean:.1f}", f"{med:.1f}", f"{p95:.1f}")
    console.print(t)
    for n in res.notes[:10]:
        console.print(f"[yellow]{escape(n)}[/yellow]")
    console.print("Run again with --device cpu and a GPU device to compare. "
                  "Quote only numbers measured on your own photos.")


@app.command()
def version():
    """Show the version."""
    console.print(f"photocure {__version__}")


if __name__ == "__main__":
    app()
