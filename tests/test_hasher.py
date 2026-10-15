"""Tests for the two-stage hierarchical hasher (xxh64 quick + SHA-256)."""
from __future__ import annotations

import hashlib
import xxhash

from photocure.scanner.hasher import (
    compute_quick_hash,
    compute_sha256,
    is_same_quick_hash,
    quick_hash_parts,
)


def _known_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_sha256_matches_stdlib(tmp_path):
    p = tmp_path / "a.bin"
    data = b"photocure" * 1000
    p.write_bytes(data)
    assert compute_sha256(p) == _known_sha256(data)


def test_sha256_empty_file(tmp_path):
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert compute_sha256(p) == hashlib.sha256(b"").hexdigest()


def test_sha256_chunked_is_size_independent(tmp_path):
    p = tmp_path / "b.bin"
    data = bytes(range(256)) * 4096  # 1 MiB
    p.write_bytes(data)
    assert compute_sha256(p, chunk_size=64 * 1024) == _known_sha256(data)
    assert compute_sha256(p, chunk_size=10 * 1024 * 1024) == _known_sha256(data)


def test_quick_hash_structure(tmp_path):
    p = tmp_path / "c.bin"
    data = b"x" * 300_000
    p.write_bytes(data)
    qh = compute_quick_hash(p, head_bytes=65536, tail_bytes=65536)
    size, hexv = quick_hash_parts(qh)
    assert size == 300_000
    # Independently compute the expected xxh64.
    expected = xxhash.xxh64()
    expected.update(data[:65536])
    expected.update(data[-65536:])
    assert hexv == expected.hexdigest()


def test_quick_hash_small_file_reads_whole_file(tmp_path):
    p = tmp_path / "small.bin"
    data = b"tiny"
    p.write_bytes(data)
    size, hexv = quick_hash_parts(compute_quick_hash(p))
    assert size == 4
    assert hexv == xxhash.xxh64(data).hexdigest()


def test_quick_hash_distinguishes_middle_content(tmp_path):
    """Two files with same size, same head and tail but different middles
    must still get the *same* quick hash (by design: the quick hash is only a
    cheap probe) but different SHA-256."""
    head = b"H" * 100_000
    tail = b"T" * 100_000
    p1 = tmp_path / "m1.bin"
    p2 = tmp_path / "m2.bin"
    p1.write_bytes(head + b"A" * 10_000 + tail)
    p2.write_bytes(head + b"B" * 10_000 + tail)
    q1, q2 = compute_quick_hash(p1), compute_quick_hash(p2)
    assert is_same_quick_hash(q1, q2)
    assert compute_sha256(p1) != compute_sha256(p2)


def test_quick_hash_distinguishes_tail(tmp_path):
    p1 = tmp_path / "t1.bin"
    p2 = tmp_path / "t2.bin"
    p1.write_bytes(b"S" * 200_000 + b"AAAA")
    p2.write_bytes(b"S" * 200_000 + b"BBBB")
    assert not is_same_quick_hash(compute_quick_hash(p1), compute_quick_hash(p2))
