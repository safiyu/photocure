"""The one library root: where finished photos live, how it is chosen and where things go inside it."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from .config import CuratorConfig

BLOCKED_DIRS = (
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/boot", "/dev",
    "/proc", "/sys", "/System", "/Library", "/Applications", "/private/etc",
    "/var/lib", "/var/log", "/var/db",
)


class LibraryRootMissing(ValueError):
    """No library folder is configured yet."""


class LibraryRootError(ValueError):
    """The chosen folder is not allowed; ``status`` is the HTTP code to answer with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def library_root(dbp: Path, cfg: CuratorConfig | None = None) -> Path | None:
    """The settings file wins, then PCUR_LIBRARY_ROOT, else None."""
    from .services import load_settings
    stored = load_settings(dbp).get("library_root")
    if stored:
        return Path(stored)
    env = (cfg or CuratorConfig.load(Path(dbp))).library_root
    return Path(env) if env else None


def library_root_source(dbp: Path, cfg: CuratorConfig | None = None) -> Literal["settings", "environment", "none"]:
    from .services import load_settings
    if load_settings(dbp).get("library_root"):
        return "settings"
    env = (cfg or CuratorConfig.load(Path(dbp))).library_root
    return "environment" if env else "none"


def require_library_root(dbp: Path, cfg: CuratorConfig | None = None) -> Path:
    root = library_root(dbp, cfg)
    if root is None:
        raise LibraryRootMissing("library folder not set")
    return root


def trash_dir(root: Path) -> Path:
    return Path(root) / ".trash"


def source_library_root(dbp: Path) -> Path | None:
    """Common directory of the indexed source photos, or None when unknown or too shallow."""
    from sqlmodel import func, select

    from .db import models as M
    from .db.session import get_session
    moved = select(M.ActionLedger.target_path).where(
        M.ActionLedger.status == "committed",
        M.ActionLedger.operation_type == "move")
    from sqlalchemy.exc import OperationalError
    try:
        with get_session(dbp) as s:
            lo, hi = s.exec(select(func.min(M.ImageRecord.path),
                                   func.max(M.ImageRecord.path)).where(
                M.ImageRecord.scan_status != "missing",
                M.ImageRecord.stage == "new",
                M.ImageRecord.path.not_in(moved))).one()
    except OperationalError:  # no database yet, so nothing is indexed
        return None
    if not lo or not hi:
        return None
    try:
        root = os.path.commonpath([lo, hi])
    except ValueError:
        return None
    if root in (lo, hi):
        root = os.path.dirname(root)
    rp = Path(root)
    return rp if len(rp.parts) >= 3 else None


def _within(path: Path, roots: list[Path]) -> bool:
    try:
        rp = path.resolve()
    except OSError:
        return False
    return any(rp == r.resolve() or r.resolve() in rp.parents for r in roots)


def validate_library_root(raw: str, dbp: Path, *, browse_roots: list[Path] | None = None,
                          inside_source_ok: bool = False) -> Path:
    """Reject relative, system, shallow, home, hidden and photocure-data folders."""
    if not raw or not raw.strip():
        raise LibraryRootError("library folder is required")
    p = Path(raw.strip()).expanduser()
    if not p.is_absolute():
        raise LibraryRootError("library folder must be an absolute path")
    p = p.resolve()
    if len(p.parts) < 3 or p == Path.home().resolve():
        raise LibraryRootError(f"library folder is too broad: {p}")
    for b in BLOCKED_DIRS:
        bp = Path(b)
        if p == bp or bp in p.parents:
            raise LibraryRootError(f"library folder is inside a system directory: {p}")
    try:
        hidden = p.relative_to(Path.home().resolve()).parts[:1]
    except ValueError:
        hidden = ()
    if hidden and hidden[0].startswith("."):
        raise LibraryRootError(f"library folder is inside a hidden home folder: {p}")
    if browse_roots is not None and not _within(p, browse_roots):
        raise LibraryRootError("library folder is outside the allowed locations", 403)
    from .auth import token_file_path
    tdir = token_file_path(CuratorConfig.load(Path(dbp)).cache_dir).parent.resolve()
    if p == tdir or (tdir.name == "photocure" and tdir in p.parents):
        raise LibraryRootError("library folder must not be the photocure data directory")
    src = None if inside_source_ok else source_library_root(dbp)
    if src is not None and (p == src or src in p.parents):
        raise LibraryRootError(f"library folder must not be inside your photo library ({src})")
    return p


def category_folder(cfg: CuratorConfig, category: str) -> str:
    """Folder name under the root for a category, honouring taxonomy folder overrides."""
    from .rules.taxonomy import get_taxonomy
    tax = get_taxonomy(cfg)
    cat = tax.category_of(category)
    return tax.folder_override_of(category) or (cat.folder if cat else category)


def library_target(cfg: CuratorConfig, root: Path, rec, category: str) -> Path:
    return Path(root) / category_folder(cfg, category) / rec.filename
