"""Perceptual hashing (64-bit DCT pHash / dHash) and a near-duplicate index.

The index uses multi-index (banded) lookup: a 64-bit hash is split into
``threshold + 1`` disjoint bands, so any pair within ``threshold`` bits must
share at least one band exactly (pigeonhole). Candidates are then verified
with a popcount, which keeps lookups sub-linear without a BK-tree.
"""
from __future__ import annotations

import io
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("photocure.scanner.phash")

HASH_BITS = 64
DEFAULT_THRESHOLD = 6
_DCT_SIZE = 32
_RAW_EXTS = {".cr2", ".nef", ".arw", ".dng"}

_ORIENT_OPS = {2: "FLIP_LEFT_RIGHT", 3: "ROTATE_180", 4: "FLIP_TOP_BOTTOM",
               5: "TRANSPOSE", 6: "ROTATE_270", 7: "TRANSVERSE", 8: "ROTATE_90"}


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0, :] = np.sqrt(1.0 / n)
    return m


_DCT = _dct_matrix(_DCT_SIZE)


def phash_from_gray(gray: np.ndarray) -> int:
    """pHash of a 32x32 float grayscale array (top-left 8x8 DCT vs median)."""
    coeffs = _DCT @ gray.astype(np.float64) @ _DCT.T
    low = coeffs[:8, :8].flatten()
    med = np.median(low[1:])  # exclude the DC term from the median
    bits = low > med
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def dhash_from_gray(gray: np.ndarray) -> int:
    """dHash of a 9x8 grayscale array (row-wise gradient sign)."""
    diff = gray[:, 1:] > gray[:, :-1]
    out = 0
    for b in diff.flatten():
        out = (out << 1) | int(b)
    return out


def to_hex(h: int) -> str:
    return f"{h:016x}"


def from_hex(s: str) -> int:
    return int(s, 16)


def hamming(a: int | str, b: int | str) -> int:
    ai = from_hex(a) if isinstance(a, str) else a
    bi = from_hex(b) if isinstance(b, str) else b
    return (ai ^ bi).bit_count()


def _open_small(path: Path):
    """Open an image cheaply, using a RAW embedded thumbnail when needed."""
    from PIL import Image

    if path.suffix.lower() in _RAW_EXTS:
        try:
            import rawpy
            with rawpy.imread(str(path)) as rp:
                thumb = rp.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                return Image.open(io.BytesIO(thumb.data))
        except Exception:
            return None
    return Image.open(path)


def _gray_array(path: Path | str, size: tuple[int, int]) -> np.ndarray | None:
    from PIL import Image

    path = Path(path)
    try:
        im = _open_small(path)
        if im is None:
            return None
        with im:
            try:
                im.draft("L", (size[0] * 4, size[1] * 4))
            except Exception:
                pass
            orient = 1
            try:
                orient = int(im.getexif().get(274, 1) or 1)
            except Exception:
                pass
            g = im.convert("L").resize(size, Image.LANCZOS)
            op = _ORIENT_OPS.get(orient)
            if op:
                g = g.transpose(getattr(Image.Transpose, op))
                if g.size != size:
                    g = g.resize(size, Image.LANCZOS)
            return np.asarray(g, dtype=np.float64)
    except Exception as e:
        log.debug("phash decode failed for %s: %s", path, e)
        return None


def compute_phash(path: Path | str) -> str | None:
    """64-bit DCT pHash as 16 hex chars, or ``None`` if the file cannot decode."""
    arr = _gray_array(path, (_DCT_SIZE, _DCT_SIZE))
    return None if arr is None else to_hex(phash_from_gray(arr))


def compute_dhash(path: Path | str) -> str | None:
    """64-bit dHash as 16 hex chars, or ``None`` if the file cannot decode."""
    arr = _gray_array(path, (9, 8))
    # PIL size is (w, h) = (9, 8); the array is (rows=8, cols=9).
    return None if arr is None else to_hex(dhash_from_gray(arr))


@dataclass
class _Entry:
    key: int
    hash: int
    pixels: int
    size: int
    quick_hash: str
    sha256: str
    near_dup_of: int | None = None


def _bands(bits: int, n: int) -> list[tuple[int, int]]:
    """Split ``bits`` into ``n`` contiguous (shift, mask) bands."""
    base, extra = divmod(bits, n)
    out, shift = [], 0
    for i in range(n):
        width = base + (1 if i < extra else 0)
        out.append((shift, (1 << width) - 1))
        shift += width
    return out


class NearDupIndex:
    """Incremental near-duplicate grouping keyed by record id.

    ``add`` returns ``(dup_key, keeper_key)`` updates: the lower-resolution
    member of each near-duplicate group points at the highest-resolution one.
    """

    def __init__(self, threshold: int = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._bands = _bands(HASH_BITS, threshold + 1)
        self._buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._entries: dict[int, _Entry] = {}
        self._children: dict[int, set[int]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: int) -> bool:
        return key in self._entries

    def near_dup_of(self, key: int) -> int | None:
        return self._entries[key].near_dup_of

    @staticmethod
    def _rank(e: _Entry) -> tuple:
        # Higher resolution wins, then bigger file, then the older id.
        return (e.pixels, e.size, -e.key)

    @staticmethod
    def _same_file(a: _Entry, b: _Entry) -> bool:
        if a.sha256 and b.sha256:
            return a.sha256 == b.sha256
        return bool(a.quick_hash) and a.quick_hash == b.quick_hash

    def _root(self, key: int) -> int:
        seen = set()
        while True:
            nxt = self._entries[key].near_dup_of
            if nxt is None or nxt in seen or nxt not in self._entries:
                return key
            seen.add(key)
            key = nxt

    def _candidates(self, h: int) -> set[int]:
        out: set[int] = set()
        for i, (shift, mask) in enumerate(self._bands):
            out.update(self._buckets.get((i, (h >> shift) & mask), ()))
        return out

    def _store(self, e: _Entry) -> None:
        self._entries[e.key] = e
        for i, (shift, mask) in enumerate(self._bands):
            self._buckets[(i, (e.hash >> shift) & mask)].append(e.key)
        if e.near_dup_of is not None and e.near_dup_of in self._entries:
            self._children[e.near_dup_of].add(e.key)

    def seed(self, rows: Iterable[tuple]) -> None:
        """Bulk-load persisted rows ``(key, phash_hex, pixels, size, qh, sha, near_dup_of)``."""
        for key, ph, pixels, size, qh, sha, nd in rows:
            if not ph or key in self._entries:
                continue
            self._store(_Entry(key, from_hex(ph), int(pixels or 0), int(size or 0),
                               qh or "", sha or "", nd))

    def add(self, key: int, phash_hex: str, pixels: int, size: int,
            quick_hash: str = "", sha256: str = "") -> list[tuple[int, int]]:
        if key in self._entries:
            return []
        e = _Entry(key, from_hex(phash_hex), int(pixels), int(size),
                   quick_hash or "", sha256 or "")
        matches = [self._entries[k] for k in self._candidates(e.hash)
                   if (e.hash ^ self._entries[k].hash).bit_count() <= self.threshold
                   and not self._same_file(e, self._entries[k])]
        self._store(e)
        if not matches:
            return []
        group = {self._root(m.key) for m in matches} | {key}
        best = max((self._entries[k] for k in group), key=self._rank)
        updates: list[tuple[int, int]] = []
        for k in sorted(group):
            if k == best.key:
                continue
            kids = list(self._children.pop(k, ()))
            self._entries[k].near_dup_of = best.key
            self._children[best.key].add(k)
            updates.append((k, best.key))
            for c in kids:
                self._entries[c].near_dup_of = best.key
                self._children[best.key].add(c)
                updates.append((c, best.key))
        return updates
