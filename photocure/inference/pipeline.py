"""Sequential GPU inference pipeline (Phase A: faces, Phase B: CLIP).

Orchestrates the two model families without ever holding both in VRAM at once:
    1. Phase A - FaceEngine (InsightFace/SCRFD or fallback).
    2. Memory sweep - ``gc.collect()`` + ``torch.cuda.empty_cache()``.
    3. Phase B - SemanticEngine (OpenCLIP).

Quality metrics are computed on the same decoded frame and persisted alongside
the inference results. Progress is durable: every chunk is committed, Phase A
leaves rows in ``faces_done`` and only Phase B promotes them to ``inferred``, so
a crash loses at most one chunk and the next run resumes where it stopped.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sqlmodel import col, select as sselect

from ..config import CuratorConfig
from ..db.models import (
    FaceDetection,
    ImageRecord,
    QualityMetric,
    SemanticScore,
)
from ..db.repository import PhotoRepository, quantize_embedding
from ..db.session import get_session
from ..quality.engine import QualityEngine
from ..quality.image_loader import load_image
from .aesthetic import get_head
from .backends import (
    BackendState,
    detect_backend,
    empty_cuda_cache,
    is_oom,
    resolve_decode_workers,
)
from .face_engine import FaceEngine
from .semantic_engine import SemanticEngine

log = logging.getLogger("photocure.pipeline")

# Statuses that still need work. "faces_done" rows only need Phase B.
_PENDING_STATUSES = ("pending", "faces_done")
# SQLite's host-parameter limit is 999 on old builds.
_IN_BATCH = 500


@dataclass
class InferResult:
    processed: int = 0
    faces_total: int = 0
    quality_computed: int = 0
    semantic_computed: int = 0
    failures: int = 0
    duration_s: float = 0.0
    skipped_already_inferred: int = 0
    reanalyzed_stub: int = 0
    oom_retries: int = 0
    errors: list = field(default_factory=list)


@dataclass
class _LoadedFrame:
    image_id: int
    bgr: np.ndarray
    # Decoded side / original side, stored so pixel coords can be mapped back.
    scale: float = 1.0


def mark_stub_rows_for_reanalysis(session) -> int:
    """Queue inferred rows scored by the stub (or an unknown backend) again.

    Returns how many rows went back to ``pending``. Rows scored by a real
    backend are never touched. The caller owns the commit.
    """
    backend_col = getattr(SemanticScore, "backend", None)
    if backend_col is None:  # schema without the backend column
        return 0
    rows = session.exec(
        sselect(ImageRecord)
        .join(SemanticScore, SemanticScore.image_id == ImageRecord.id)
        .where(ImageRecord.infer_status == "inferred",
               col(backend_col).in_(("", "stub")))
    ).all()
    for rec in rows:
        rec.infer_status = "pending"
    session.flush()
    return len(rows)


def run_with_retry(fn: Callable[[Sequence], list], items: Sequence,
                   bs: BackendState | None = None,
                   on_oom: Callable[[], None] | None = None) -> list:
    """Apply ``fn`` to ``items``, bisecting the batch when it fails.

    CUDA out-of-memory empties the cache first. A failure on a single item
    yields ``None`` for that item instead of aborting the run. The returned
    list is aligned with ``items``.
    """
    items = list(items)
    if not items:
        return []
    try:
        out = list(fn(items))
        if len(out) != len(items):
            raise RuntimeError(f"batch returned {len(out)} results for {len(items)} items")
        return out
    except Exception as e:
        oom = is_oom(e)
        if oom:
            if on_oom:
                on_oom()
            if bs is not None:
                empty_cuda_cache(bs)
            gc.collect()
        if len(items) == 1:
            log.warning("inference failed for a single item (%s: %s)",
                        e.__class__.__name__, e)
            return [None]
        mid = len(items) // 2
        log.info("batch of %d failed (%s); retrying as %d + %d",
                 len(items), "oom" if oom else e.__class__.__name__,
                 mid, len(items) - mid)
        return (run_with_retry(fn, items[:mid], bs, on_oom)
                + run_with_retry(fn, items[mid:], bs, on_oom))


class InferencePipeline:
    """Run face + semantic inference over all pending images."""

    def __init__(self, config: CuratorConfig | None = None,
                 backend: BackendState | None = None):
        self.cfg = config or CuratorConfig()
        self.bs = backend or detect_backend(self.cfg)
        self.quality = QualityEngine(self.cfg.quality)

    # ------------------------------------------------------------ selection
    @staticmethod
    def _select_records(repo: PhotoRepository, only_paths: list[str] | None,
                        retry_failed: bool, limit: int) -> list[ImageRecord]:
        s = repo.s
        if only_paths is not None:
            paths = list(dict.fromkeys(only_paths))
            recs: list[ImageRecord] = []
            for i in range(0, len(paths), _IN_BATCH):
                part = paths[i:i + _IN_BATCH]
                recs.extend(s.exec(
                    sselect(ImageRecord).where(col(ImageRecord.path).in_(part))
                ).all())
            recs.sort(key=lambda r: r.id or 0)
            return recs[:limit] if limit else recs
        statuses = _PENDING_STATUSES + (("failed",) if retry_failed else ())
        stmt = (sselect(ImageRecord)
                .where(col(ImageRecord.infer_status).in_(statuses))
                .order_by(col(ImageRecord.id)))
        if limit:
            stmt = stmt.limit(limit)
        return list(s.exec(stmt).all())

    # ------------------------------------------------------------------ run
    def run(self, db_path: Path | str | None = None,
            only_paths: list[str] | None = None,
            on_progress: Callable[[int, int, str], None] | None = None,
            show_progress: bool = True,
            retry_failed: bool = False,
            limit: int = 0,
            reanalyze_stub: bool = True,
            ) -> InferResult:
        db = db_path or self.cfg.db_path
        result = InferResult()
        t0 = time.monotonic()

        with get_session(db) as session:
            # Keep ORM rows usable across the per-chunk commits below.
            session.expire_on_commit = False
            repo = PhotoRepository(session)
            if reanalyze_stub and self.bs.semantic_backend != "stub":
                # A real model is available now: redo what the stub analysed.
                result.reanalyzed_stub = mark_stub_rows_for_reanalysis(session)
                if result.reanalyzed_stub:
                    log.info("re-analysing %d photos first scored by the stub",
                             result.reanalyzed_stub)
                    session.commit()
            pending = self._select_records(repo, only_paths, retry_failed, limit)

            todo: list[ImageRecord] = []
            for rec in pending:
                if rec.infer_status == "inferred":
                    result.skipped_already_inferred += 1
                else:
                    todo.append(rec)

            if only_paths is None:
                # Pending selection never returns inferred rows; count them so
                # the resumption skip is observable.
                result.skipped_already_inferred = repo.count_inferred()

            if not todo:
                result.duration_s = time.monotonic() - t0
                return result

            total_steps = len(todo) * 2
            if on_progress:
                on_progress(0, total_steps,
                            f"Loading face detection models for {len(todo)} images…")

            dev = self.cfg.device
            chunk_size = max(1, int(dev.infer_chunk_size))
            chunks = [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]

            def _on_oom() -> None:
                result.oom_retries += 1

            def _decode_single(rec: ImageRecord, max_side: int):
                try:
                    img = load_image(rec.path, max_side=max_side,
                                     orientation=rec.orientation)
                    return rec, img, None
                except Exception as e:
                    return rec, None, e

            def _decode(chunk_recs: list[ImageRecord], max_side: int
                        ) -> list[_LoadedFrame]:
                """Decode a chunk in parallel; failures are marked and counted."""
                frames: list[_LoadedFrame] = []
                if not chunk_recs:
                    return frames
                workers = min(resolve_decode_workers(self.cfg), max(1, len(chunk_recs)))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    decoded = list(executor.map(
                        lambda r: _decode_single(r, max_side), chunk_recs))
                for rec, img, exc in decoded:
                    if img is not None:
                        frames.append(_LoadedFrame(rec.id, img.bgr, img.scale))
                    else:
                        log.warning("decode failed for %s: %s", rec.path, exc)
                        rec.infer_status = "failed"
                        result.failures += 1
                        result.errors.append(f"{rec.path}: decode failure ({exc})")
                return frames

            progress = None
            tasks: dict = {}
            if show_progress:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    transient=False,
                )
                tasks["faces"] = progress.add_task("Faces (Phase A)", total=len(todo))
                tasks["clip"] = progress.add_task("CLIP (Phase B)", total=len(todo))
                progress.start()

            try:
                self._phase_a(repo, chunks, result, _decode, _on_oom, on_progress,
                              progress, tasks, total_steps)
                if on_progress:
                    on_progress(len(todo), total_steps,
                                f"Loading CLIP semantic model for {len(todo)} images…")
                self._phase_b(repo, chunks, result, _decode, _on_oom, on_progress,
                              progress, tasks, total_steps, len(todo))
                if on_progress:
                    on_progress(total_steps, total_steps,
                                f"Inference complete: {result.processed} processed")
            finally:
                if progress is not None:
                    progress.stop()

        result.duration_s = time.monotonic() - t0
        return result

    # -------------------------------------------------------------- phase A
    def _phase_a(self, repo, chunks, result, decode, on_oom, on_progress,
                 progress, tasks, total_steps) -> None:
        """Decode -> faces + quality, committed per chunk, then free pixels."""
        dev = self.cfg.device
        face_max = dev.face_max_side
        face_batch = max(1, int(dev.face_batch_size))
        face_engine = FaceEngine(self.cfg, backend=self.bs)
        done = 0
        try:
            for chunk in chunks:
                # Rows already in "faces_done" only need Phase B.
                a_recs = [r for r in chunk if r.infer_status in ("pending", "failed")]
                done += len(chunk) - len(a_recs)
                rec_by_id = {r.id: r for r in a_recs}
                frames = decode(a_recs, face_max)
                decoded = {fr.image_id for fr in frames}
                for rec in a_recs:
                    if rec.id not in decoded:
                        done += 1
                        if on_progress:
                            on_progress(done, total_steps,
                                        f"Phase 1/2 (Faces & Quality): Skipped unreadable {Path(rec.path).name}")
                for start in range(0, len(frames), face_batch):
                    batch = frames[start:start + face_batch]
                    faces_per = run_with_retry(
                        lambda fs: face_engine.detect_faces_batch([f.bgr for f in fs]),
                        batch, self.bs, on_oom)
                    for fr, faces in zip(batch, faces_per):
                        rec = rec_by_id[fr.image_id]
                        done += 1
                        if progress is not None:
                            progress.advance(tasks["faces"])
                        if faces is None:
                            rec.infer_status = "failed"
                            result.failures += 1
                            result.errors.append(f"{rec.path}: face detection failed")
                            continue
                        result.faces_total += len(faces)
                        try:
                            q = self.quality.score(fr.bgr)
                            repo.upsert_quality(QualityMetric(
                                image_id=fr.image_id,
                                **{k: v for k, v in q.items() if k != "needs_review"},
                            ))
                            repo.replace_faces(
                                fr.image_id,
                                [self._face_row(fr.image_id, f, fr.scale) for f in faces],
                            )
                            # Phase B promotes this to "inferred".
                            rec.infer_status = "faces_done"
                        except Exception as e:
                            log.warning("persist failed for %s: %s", rec.path, e)
                            rec.infer_status = "failed"
                            result.failures += 1
                            result.errors.append(f"{rec.path}: {e}")
                        if on_progress:
                            on_progress(done, total_steps,
                                        f"Phase 1/2 (Faces & Quality): {Path(rec.path).name}")
                repo.s.commit()
                frames.clear()  # release this chunk's decoded pixels
        finally:
            face_engine.release()
            gc.collect()
            empty_cuda_cache(self.bs)

    # -------------------------------------------------------------- phase B
    def _phase_b(self, repo, chunks, result, decode, on_oom, on_progress,
                 progress, tasks, total_steps, n_todo) -> None:
        """Re-decode each chunk at CLIP size, embed, classify, score, commit."""
        clip_max = self.cfg.device.clip_max_side
        sem_engine = SemanticEngine(self.cfg, backend=self.bs)
        head = get_head()
        done = 0
        try:
            for chunk in chunks:
                rec_by_id = {r.id: r for r in chunk}
                b_recs = [r for r in chunk if r.infer_status == "faces_done"]
                done += len(chunk) - len(b_recs)  # failed in Phase A
                frames = decode(b_recs, clip_max)
                decoded = {fr.image_id for fr in frames}
                for rec in b_recs:
                    if rec.id not in decoded:
                        done += 1
                        if on_progress:
                            on_progress(n_todo + done, total_steps,
                                        f"Phase 2/2 (CLIP Semantics): Skipped unreadable {Path(rec.path).name}")
                sems = run_with_retry(
                    lambda fs: sem_engine.classify_and_embed([f.bgr for f in fs]),
                    frames, self.bs, on_oom)
                for fr, sem in zip(frames, sems):
                    rec = rec_by_id[fr.image_id]
                    done += 1
                    if progress is not None:
                        progress.advance(tasks["clip"])
                    if sem is None:
                        rec.infer_status = "failed"
                        result.failures += 1
                        result.errors.append(f"{rec.path}: semantic inference failed")
                        continue
                    try:
                        emb = sem.embedding or []
                        # Read after classify: the engine may have fallen back to stub.
                        backend_name = "clip" if self.bs.semantic_backend == "clip" else "stub"
                        repo.upsert_semantic(SemanticScore(
                            image_id=fr.image_id,
                            model_name=(self.cfg.taxonomy.clip_model_name
                                        if backend_name == "clip" else "stub"),
                            backend=backend_name,
                            top1_category=sem.top1_name,
                            top1_score=sem.top1_score,
                            top2_category=sem.top2_name,
                            top2_score=sem.top2_score,
                            score_margin=sem.margin,
                            distribution=_json_sem(sem.distribution),
                            embedding=quantize_embedding(emb),
                            aesthetic=head.score(emb) if len(emb) == head.w.shape[0] else None,
                        ))
                        rec.infer_status = "inferred"
                        result.quality_computed += 1
                        result.semantic_computed += 1
                        result.processed += 1
                        if on_progress:
                            on_progress(n_todo + done, total_steps,
                                        f"Phase 2/2 (CLIP Semantics): {Path(rec.path).name}")
                    except Exception as e:
                        log.warning("semantic persist failed for %s: %s", rec.path, e)
                        rec.infer_status = "failed"
                        result.failures += 1
                        result.errors.append(f"{rec.path}: {e}")
                repo.s.commit()
                frames.clear()  # release this chunk's decoded pixels
        finally:
            sem_engine.release()
            gc.collect()
            empty_cuda_cache(self.bs)

    @staticmethod
    def _face_row(image_id: int, f, scale: float = 1.0) -> FaceDetection:
        lm = f.landmarks or []
        emb = getattr(f, "embedding", None)
        return FaceDetection(
            image_id=image_id,
            bbox_x1=f.x1, bbox_y1=f.y1, bbox_x2=f.x2, bbox_y2=f.y2,
            confidence=f.confidence,
            landmarks=_json_sem([(round(x, 5), round(y, 5)) for x, y in lm]),
            scale=float(scale),
            embedding_json=quantize_embedding(emb) if emb else None,
            eye_aspect_ratio=f.eye_aspect_ratio,
            face_sharpness=f.face_sharpness,
        )


def _json_sem(obj) -> str:
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return "[]"


def run_inference(config: CuratorConfig | None = None,
                  db_path: Path | str | None = None,
                  only_paths: list[str] | None = None,
                  show_progress: bool = True,
                  retry_failed: bool = False,
                  limit: int = 0,
                  reanalyze_stub: bool = True) -> InferResult:
    cfg = config or CuratorConfig.load(db_path)
    return InferencePipeline(cfg).run(db_path=db_path, only_paths=only_paths,
                                      show_progress=show_progress,
                                      retry_failed=retry_failed, limit=limit,
                                      reanalyze_stub=reanalyze_stub)
