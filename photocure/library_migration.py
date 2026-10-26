"""Move the whole library to another folder: copy and verify first, then rewrite the database."""
from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

from sqlmodel import select

from .config import CuratorConfig
from .db import models as M
from .db.session import get_session
from .engine.hash import file_sha256, file_size
from .engine.locks import LEDGER_WRITING
from .library import require_library_root


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _nearest_existing(p: Path) -> Path:
    cur = p
    while not cur.exists() and cur != cur.parent:
        cur = cur.parent
    return cur


def _target_problem(old: Path, new: Path) -> str | None:
    if new == old or old in new.parents:
        return "the new folder must not be inside the current library"
    if new in old.parents:
        return "the new folder must not contain the current library"
    if new.exists() and (not new.is_dir() or any(new.iterdir())):
        return "the new folder must be empty or not exist yet"
    return None


def preview_migration(dbp: Path, cfg: CuratorConfig, new_root: Path) -> dict:
    """What a move would copy, and whether the chosen folder can take it."""
    old = require_library_root(dbp, cfg).resolve()
    new = Path(new_root).resolve()
    files = _files(old) if old.is_dir() else []
    size = sum(f.stat().st_size for f in files)
    free = shutil.disk_usage(_nearest_existing(new)).free
    error = _target_problem(old, new)
    return {"ok": error is None and size <= free, "error": error or (None if size <= free else "not enough free space"),
            "files": len(files), "bytes": size, "free_bytes": free, "enough_space": size <= free}


def _copy_verified(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if file_size(dst) != file_size(src) or file_sha256(dst) != file_sha256(src):
        raise OSError(f"copy does not match the original: {src}")


def _swap(value: str | None, old: Path, new: Path) -> str | None:
    if not value:
        return value
    p = Path(value)
    if p == old or old in p.parents:
        return str(new / p.relative_to(old))
    return value


def migrate_library(dbp: Path, cfg: CuratorConfig, new_root: Path, *,
                    on_progress: Callable[[int, int, str], None] | None = None,
                    cancel_check: Callable[[], bool] | None = None) -> dict:
    """Copy every file to the same place under ``new_root``, verify it, then point the
    database and the library setting at the new folder in one step. The old folder is left alone."""
    old = require_library_root(dbp, cfg).resolve()
    new = Path(new_root).resolve()
    problem = _target_problem(old, new)
    if problem:
        return {"ok": False, "error": problem, "copied": 0, "bytes": 0, "failed": [],
                "old_root": str(old), "new_root": str(new)}
    with LEDGER_WRITING:  # nothing else moves a file between the first copy and the database switch
        return _copy_and_switch(dbp, old, new, on_progress, cancel_check)


def _copy_and_switch(dbp: Path, old: Path, new: Path, on_progress, cancel_check) -> dict:
    from .services import load_settings, save_settings

    files = _files(old)
    copied = nbytes = 0
    failed: list[str] = []
    for n, src in enumerate(files, 1):
        if cancel_check and cancel_check():
            failed.append("stopped before every file was copied")
            break
        try:
            _copy_verified(src, new / src.relative_to(old))
            copied += 1
            nbytes += src.stat().st_size
        except OSError as e:
            failed.append(f"{src}: {e}")
            break
        if on_progress:
            on_progress(n, len(files), src.name)
    result = {"ok": not failed, "copied": copied, "bytes": nbytes, "failed": failed,
              "old_root": str(old), "new_root": str(new)}
    if failed:
        return result

    previous = load_settings(dbp).get("library_root")
    with get_session(dbp) as s:
        for rec in s.exec(select(M.ImageRecord).where(M.ImageRecord.stage.in_(("library", "trash")))).all():
            rec.path = _swap(rec.path, old, new) or rec.path
            rec.target_path = _swap(rec.target_path, old, new)
        for e in s.exec(select(M.ActionLedger)).all():
            e.target_path = _swap(e.target_path, old, new) or e.target_path
            e.source_path = _swap(e.source_path, old, new) or e.source_path
        s.flush()
        save_settings(dbp, {"library_root": str(new)}, force=True)
        try:
            s.commit()
        except Exception:
            if previous:
                save_settings(dbp, {"library_root": previous}, force=True)
            raise
    return result
