"""Calibration and timing commands behind ``photocure bench``.

Nothing here is a published number. Quote only what these commands print on your own photos.
"""
from __future__ import annotations

import csv
import statistics
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .config import CuratorConfig

SWEEP = tuple(round(0.30 + 0.05 * i, 2) for i in range(14))  # 0.30 .. 0.95


@dataclass
class SweepRow:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return None if p is None or r is None or (p + r) == 0 else 2 * p * r / (p + r)


def read_labels(path: Path) -> list[tuple[Path, str]]:
    """Rows of ``path,label``. Paths are relative to the CSV; label is open or closed."""
    out: list[tuple[Path, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip().lower()
            name = (row.get("path") or "").strip()
            if name and label in ("open", "closed"):
                out.append((path.parent / name, label))
    return out


def sweep(scored: Sequence[tuple[str, float | None]], thresholds: Sequence[float] = SWEEP) -> list[SweepRow]:
    """Confusion counts per threshold, where score >= threshold predicts 'closed'.

    Photos whose eyes could not be measured (score None) are left out: unknown is neither
    a hit nor a miss.
    """
    measured = [(lab, s) for lab, s in scored if s is not None]
    rows = []
    for t in thresholds:
        tp = sum(1 for lab, s in measured if lab == "closed" and s >= t)
        fp = sum(1 for lab, s in measured if lab == "open" and s >= t)
        fn = sum(1 for lab, s in measured if lab == "closed" and s < t)
        tn = sum(1 for lab, s in measured if lab == "open" and s < t)
        rows.append(SweepRow(t, tp, fp, fn, tn))
    return rows


def best_threshold(rows: Sequence[SweepRow]) -> SweepRow | None:
    scored = [r for r in rows if r.f1 is not None]
    return max(scored, key=lambda r: (r.f1, r.threshold)) if scored else None


def _orientation(path: Path) -> int:
    from .scanner.exif_reader import extract_exif
    try:
        return extract_exif(path).orientation or 1
    except Exception:
        return 1


def score_labelled_photos(items: Sequence[tuple[Path, str]], cfg: CuratorConfig) -> tuple[list, int, int]:
    """Measure the largest face of each labelled photo. Returns (scored, no_face, unreadable)."""
    from .inference.backends import detect_backend
    from .inference.face_engine import FaceEngine
    from .quality.image_loader import load_image

    engine = FaceEngine(cfg, backend=detect_backend(cfg))
    scored: list[tuple[str, float | None]] = []
    no_face = unreadable = 0
    try:
        for path, label in items:
            try:
                img = load_image(path, max_side=None, orientation=_orientation(path)).bgr
            except Exception:
                unreadable += 1
                continue
            faces = engine.detect_faces_batch([img])[0]
            if not faces:
                no_face += 1
                continue
            face = max(faces, key=lambda f: f.w * f.h)
            engine.analyze_faces(img, [face])
            scored.append((label, face.eye_closed_score))
    finally:
        engine.release()
    return scored, no_face, unreadable


# ------------------------------------------------------------------ pipeline
STAGES = ("decode", "face detection", "crop + sharpness", "landmarks", "CLIP", "quality metrics",
          "SQLite commit")


@dataclass
class StageTimes:
    samples: dict[str, list[float]] = field(default_factory=lambda: {s: [] for s in STAGES})
    notes: list[str] = field(default_factory=list)
    photos: int = 0
    device: str = ""
    face_backend: str = ""
    semantic_backend: str = ""

    def add(self, stage: str, seconds: float) -> None:
        self.samples[stage].append(seconds * 1000.0)

    def summary(self) -> list[tuple[str, int, float, float, float]]:
        out = []
        for stage in STAGES:
            v = sorted(self.samples[stage])
            if not v:
                continue
            p95 = v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]
            out.append((stage, len(v), statistics.fmean(v), statistics.median(v), p95))
        return out


def bench_pipeline(folder: Path, cfg: CuratorConfig, limit: int = 50,
                   clock: Callable[[], float] = time.perf_counter) -> StageTimes:
    """Time each stage on up to ``limit`` photos from ``folder``."""
    from .db.models import FaceDetection, ImageRecord
    from .db.session import get_session, init_db
    from .inference.backends import detect_backend
    from .inference.face_engine import FaceEngine
    from .inference.semantic_engine import SemanticEngine
    from .quality.engine import QualityEngine
    from .quality.image_loader import load_image

    exts = {e.lower() for e in cfg.extensions}
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)[:max(1, limit)]
    bs = detect_backend(cfg)
    res = StageTimes(photos=len(files), device=bs.accelerator or bs.device,
                     face_backend=bs.face_backend, semantic_backend=bs.semantic_backend)
    if bs.semantic_backend == "stub":
        res.notes.append("CLIP ran as the placeholder classifier: its time says nothing about the real model.")
    if bs.face_backend == "stub":
        res.notes.append("No face detector is available: face stages were skipped.")
    blink_ok = False
    faces_engine, sem, quality = FaceEngine(cfg, backend=bs), SemanticEngine(cfg, backend=bs), QualityEngine(cfg.quality)
    if files and bs.face_backend != "stub":  # keep model loading out of the timings
        faces_engine.detect_faces_batch([load_image(files[0], max_side=cfg.device.face_max_side,
                                             orientation=_orientation(files[0])).bgr])
        blink_ok = faces_engine.blink_available
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "bench.db"
        init_db(db)
        for i, path in enumerate(files):
            orient = _orientation(path)
            try:
                t = clock()
                small = load_image(path, max_side=cfg.device.face_max_side, orientation=orient)
                res.add("decode", clock() - t)
            except Exception as e:
                res.notes.append(f"{path.name}: unreadable ({e.__class__.__name__})")
                continue
            faces = []
            if bs.face_backend != "stub":
                t = clock()
                faces = faces_engine.detect_faces_batch([small.bgr])[0]
                res.add("face detection", clock() - t)
                if faces:
                    full = load_image(path, max_side=None, orientation=orient).bgr
                    t = clock()
                    from .inference.face_engine import crop_face, shrink_to, standardized_sharpness
                    for f in faces:
                        crop = crop_face(full, (f.x1, f.y1, f.x2, f.y2), cfg.face.crop_margin)
                        if crop.size:
                            standardized_sharpness(crop, cfg.face.sharpness_width)
                            shrink_to(crop, cfg.face.landmark_max_side)
                    res.add("crop + sharpness", clock() - t)
                    t = clock()
                    faces_engine.analyze_faces(full, faces)
                    # analyze_faces repeats the crop work, so subtract it to isolate the landmarker.
                    crop_cost = res.samples["crop + sharpness"][-1] / 1000.0
                    res.add("landmarks", max(0.0, clock() - t - crop_cost))
            clip = load_image(path, max_side=cfg.device.clip_max_side, orientation=orient)
            t = clock()
            sem.classify_and_embed([clip.bgr])
            res.add("CLIP", clock() - t)
            t = clock()
            quality.score(clip.bgr)
            res.add("quality metrics", clock() - t)
            t = clock()
            with get_session(db) as s:
                rec = ImageRecord(path=f"/bench/{i}", filename=path.name)
                s.add(rec)
                s.flush()
                for f in faces:
                    s.add(FaceDetection(image_id=rec.id, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1,
                                        confidence=f.confidence, x1=f.x1, y1=f.y1, x2=f.x2, y2=f.y2,
                                        eye_closed_score=f.eye_closed_score))
            res.add("SQLite commit", clock() - t)
    faces_engine.release()
    sem.release()
    if not blink_ok and bs.face_backend != "stub":
        res.notes.append("MediaPipe is not installed, so the landmarks stage did not run.")
    return res
