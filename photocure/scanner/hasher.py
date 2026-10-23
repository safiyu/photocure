"""Two-stage hierarchical hashing.

Tier 1 (fast probe): file size + ``xxh64`` of the first 64 KiB and last 64 KiB.
Tier 2 (definitive): full-file streaming ``sha256`` (memory-mapped for large
files so a 500 MB RAW does not balloon RAM).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import xxhash

DEFAULT_HEAD = 64 * 1024
DEFAULT_TAIL = 64 * 1024
_READ_CHUNK = 1024 * 1024


def compute_quick_hash(path: Path | str,
                       head_bytes: int = DEFAULT_HEAD,
                       tail_bytes: int = DEFAULT_TAIL) -> str:
    """Tier-1 fingerprint: ``<size>:<xxh64(head+tail)>``.

    Reading only the head and tail keeps cost ``O(1)`` regardless of file
    size, which is what makes this viable over high-latency CIFS/SMB mounts.
    """
    path = Path(path)
    size = path.stat().st_size
    h = xxhash.xxh64()
    with open(path, "rb", buffering=0) as f:
        if size <= head_bytes + tail_bytes:
            h.update(f.read())
        else:
            h.update(f.read(head_bytes))
            f.seek(-tail_bytes, os.SEEK_END)
            h.update(f.read(tail_bytes))
    return f"{size}:{h.hexdigest()}"


def compute_sha256(path: Path | str, chunk_size: int = _READ_CHUNK) -> str:
    """Tier-2 definitive hash via chunked reads (mmap risks SIGBUS if the file shrinks)."""
    digest = hashlib.sha256()
    with open(Path(path), "rb", buffering=chunk_size) as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_same_quick_hash(a: str, b: str) -> bool:
    """Compare two quick-hash strings (ignoring the embedded size prefix)."""
    return a.rsplit(":", 1)[-1] == b.rsplit(":", 1)[-1]


def quick_hash_parts(quick_hash: str) -> tuple[int, str]:
    """Split ``"size:xxh64hex"`` into ``(size, hex)``."""
    size_s, hexv = quick_hash.rsplit(":", 1)
    return int(size_s), hexv
