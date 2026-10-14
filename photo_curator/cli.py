"""Typer CLI for the AI Photo Curator.

Commands:
    scan    Scan a directory into the state store (resumable).
    infer   Run quality + face + semantic inference over pending images.
    cluster Cluster bursts and pick best shots.
    plan    Build a dry-run routing plan (+ optional HTML audit report).
    apply   Execute a plan (move / copy / hardlink / symlink).
    undo    Roll back a committed transaction.
    stats   Show state-store statistics.
    serve   Launch the web review dashboard.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import CuratorConfig
from .db.session import get_session, init_db
from .db.repository import PhotoRepository
from .inference.backends import detect_backend, reset_backend_cache
from .services import Planner

app = typer.Typer(
    name="photo-curator",
    help="Offline, GPU-accelerated, transactionally reversible photo curation.",
    no_args_is_help=True,
    add_completion=False,
)
cli = app  # alias for the console-script entry point (photo_curator.cli:cli)
console = Console()


@contextmanager
def _config(db: Path, refresh: bool = True):
    """Yield a config bound to ``db`` (initializing the schema on first use)."""
    reset_backend_cache() if refresh else None
    cfg = CuratorConfig.load(db)
    init_db(cfg.db_path)
    return cfg


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------
@app.command()
def scan(
    target: Path = typer.Argument(..., help="Directory to scan (recursive)."),
    db: Path = typer.Option(Path("photo_curator.db"), "--db", help="SQLite state store path."),
    workers: int = typer.Option(8, "--workers", "-w", help="Worker threads."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
):
    """Scan a directory: EXIF, two-tier hashes, resumable upsert."""
    from .scanner.file_scanner import FileScanner
    cfg = _config(db)
    cfg.scan_workers = workers
    scanner = FileScanner(cfg)
    with console.status("[cyan]Scanning…[/cyan]"):
        result = scanner.scan_directory(target, db_path=cfg.db_path,
                                        show_progress=not quiet)
    t = Table(title="Scan result")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Files seen", str(result.scanned))
    t.add_row("New indexed", str(result.new))
    t.add_row("Skipped (known)", str(result.skipped_known))
    t.add_row("Duplicates", str(result.duplicates))
    t.add_row("Errors", str(result.errors))
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
    db: Path = typer.Option(Path("photo_curator.db"), "--db"),
    batch_size: int = typer.Option(32, "--batch-size", "-b"),
    device: str = typer.Option("auto", "--device", "-d", help="cuda|cpu|auto"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Max images."),
    paths: str | None = typer.Option(None, "--paths", help="Comma-separated paths (debug)."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
):
    """Run quality + face + semantic inference (sequential GPU phases)."""
    from .inference.pipeline import InferencePipeline
    cfg = _config(db)
    cfg.device.device = device
    if batch_size:
        cfg.device.face_batch_size = batch_size
        cfg.device.clip_batch_size = max(batch_size, 1)
    bs = detect_backend(cfg)
    console.print(f"[dim]device={bs.device} fp16={bs.fp16} "
                 f"face={bs.face_backend} clip={bs.semantic_backend}[/dim]")
    only = None
    if paths:
        only = [p.strip() for p in paths.split(",") if p.strip()]
    elif limit:
        with get_session(cfg.db_path) as s:
            repo = PhotoRepository(s)
            pend = repo.images_pending_inference(limit)
            only = [r.path for r in pend]
    result = InferencePipeline(cfg, backend=bs).run(
        db_path=cfg.db_path, only_paths=only, show_progress=not quiet)
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


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------
@app.command()
def cluster(
    db: Path = typer.Option(Path("photo_curator.db"), "--db"),
    window_seconds: int = typer.Option(30, "--window-seconds"),
    similarity_threshold: float = typer.Option(0.88, "--similarity-threshold", "-s"),
):
    """Cluster burst shots and select the best shot in each group."""
    from .clustering.burst_detector import BurstDetector
    from .clustering.best_shot_selector import BestShotSelector, _candidate_from_record
    from .db.models import BurstGroup
    cfg = _config(db)
    cfg.burst.window_seconds = window_seconds
    cfg.burst.similarity_threshold = similarity_threshold
    with get_session(cfg.db_path) as s:
        repo = PhotoRepository(s)
        records = repo.all_images()
        for r in records:
            r.burst_id = None
            r.is_best_shot = False
            r.is_alternate = False
        for b in repo.all_bursts():
            repo.s.delete(b)
        s.flush()
        detector = BurstDetector(cfg)
        results = detector.cluster_records(records)
        by_id = {r.id: r for r in records}
        sel = BestShotSelector(cfg)
        n = 0
        for res in results:
            members = [by_id[m] for m in res.image_ids if m in by_id]
            if len(members) < cfg.burst.min_group_size:
                continue
            best_id = sel.pick_best_shot(
                [_candidate_from_record(m) for m in members]).image_id
            earliest = min(members, key=lambda m: (m.taken_at is None, m.taken_at))
            from datetime import datetime
            _EPOCH = datetime(1970, 1, 1)
            stamp = (earliest.taken_at.strftime("%Y%m%d")
                     if earliest.taken_at else "Unknown")
            name = f"Burst_{stamp}_{n:04d}"
            bg = repo.upsert_burst(BurstGroup(
                name=name, image_count=len(members), best_shot_id=best_id,
                start_time=earliest.taken_at,
                end_time=max((m.taken_at for m in members), default=earliest.taken_at)))
            for m in members:
                m.burst_id = bg.id
                m.is_best_shot = (m.id == best_id)
                m.is_alternate = (m.id != best_id)
            n += 1
        console.print(f"[green]Clustered {n} burst groups[/green] "
                      f"(window={window_seconds}s, sim≥{similarity_threshold})")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
@app.command()
def plan(
    db: Path = typer.Option(Path("photo_curator.db"), "--db"),
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
):
    """Build a dry-run routing plan and print a summary."""
    from .rules.organizer import OrganizeMode
    cfg = _config(db)
    planner = Planner(cfg, db_path=cfg.db_path)
    m = OrganizeMode.IN_PLACE if mode == "in_place" else OrganizeMode.CATEGORY_FIRST
    p = planner.build_plan(organize_root=output_dir, operation=operation,
                           mode=m, plan_id=plan_id,
                           include_duplicates=not exclude_duplicates)
    p.dry_run = True
    planner.print_summary(p)
    if report:
        rp = planner.render_html_report(p, report, extra={
            "Mode": mode, "Operation": operation,
            "Organize root": str(output_dir), "DB": str(cfg.db_path),
        })
        console.print(f"[green]Report written:[/green] {rp}")
    console.print(f"\n[dim]Plan ID: {p.plan_id}[/dim]")
    console.print("[dim]Review the report, then run:[/dim] "
                  f"photo-curator apply --db {cfg.db_path} --plan-id {p.plan_id} "
                  f"--operation {operation}")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------
@app.command()
def apply(
    db: Path = typer.Option(Path("photo_curator.db"), "--db"),
    plan_id: str | None = typer.Option(None, "--plan-id",
                                          help="Rebuild this plan before applying."),
    operation: str = typer.Option("move", "--operation",
                                  help="move|copy|hardlink|symlink"),
    output_dir: Path = typer.Option(Path("Organized"), "--output-dir", "-o"),
    mode: str = typer.Option("category_first", "--mode"),
    yes: bool = typer.Option(False, "--yes", "-y",
                             help="Skip the confirmation prompt."),
):
    """Execute a plan (records a transaction for later undo)."""
    from .rules.organizer import OrganizeMode
    cfg = _config(db)
    planner = Planner(cfg, db_path=cfg.db_path)
    m = OrganizeMode.IN_PLACE if mode == "in_place" else OrganizeMode.CATEGORY_FIRST
    p = planner.build_plan(organize_root=output_dir, operation=operation, mode=m,
                           plan_id=plan_id)
    if p.total == 0:
        console.print("[yellow]Nothing to apply (no inferred images).[/yellow]")
        return
    console.print(f"About to [bold]{operation}[/bold] [bold]{p.total}[/bold] files "
                  f"into [bold]{output_dir}[/bold].")
    if not yes:
        ok = typer.confirm("Proceed?")
        if not ok:
            console.print("[yellow]Aborted.[/yellow]")
            return
    rep = planner.execute_plan(p, operation=operation)
    t = Table(title=f"Transaction {rep.transaction_id}")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    t.add_row("Executed", str(rep.executed))
    t.add_row("Collisions (renamed)", str(rep.collisions))
    t.add_row("Failures", str(rep.failures))
    t.add_row("Duration", f"{rep.duration_s:.2f}s")
    console.print(t)
    console.print(f"[green]Done.[/green] To undo: "
                  f"photo-curator undo --db {cfg.db_path} "
                  f"--transaction-id {rep.transaction_id}")


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------
@app.command()
def undo(
    db: Path = typer.Option(Path("photo_curator.db"), "--db"),
    transaction_id: str | None = typer.Option(None, "--transaction-id", "-t",
                                                 help="Defaults to the latest."),
    no_verify: bool = typer.Option(False, "--no-verify",
                                   help="Skip the pre-flight SHA-256 check."),
):
    """Roll back a committed transaction (reverse-chronological)."""
    cfg = _config(db)
    planner = Planner(cfg, db_path=cfg.db_path)
    tid = transaction_id
    if not tid:
        with get_session(cfg.db_path) as s:
            tid = PhotoRepository(s).latest_transaction()
    if not tid:
        console.print("[yellow]No transactions found.[/yellow]")
        return
    rep = planner.undo(tid)
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


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
@app.command()
def stats(
    db: Path = typer.Option(Path("photo_curator.db"), "--db"),
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
    db: Path = typer.Option(Path("photo_curator.db"), "--db"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Launch the web review dashboard."""
    import uvicorn
    cfg = _config(db)
    console.print(f"[cyan]Serving review dashboard on http://{host}:{port}/ "
                  f"(db={cfg.db_path})[/cyan]")
    uvicorn.run("photo_curator.ui.app:app", host=host, port=port,
                reload=reload, factory=False,
                log_level="info")


@app.command()
def version():
    """Show the version."""
    console.print(f"photo-curator {__version__}")


if __name__ == "__main__":
    app()
