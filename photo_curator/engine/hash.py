"""Small filesystem hash/size helpers for the action & rollback engines."""
from __future__ import annotations

from pathlib import Path

from ..scanner.hasher import compute_sha256


def file_sha256(path: Path | str) -> str:
    return compute_sha256(Path(path))


def file_size(path: Path | str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0
