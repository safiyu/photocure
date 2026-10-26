"""Sequential GPU inference pipeline (Phase A: faces, Phase B: CLIP).

Orchestrates the two model families without ever holding both in VRAM at once:
    1. Phase A - FaceEngine (YuNet detection, then crop analysis on the full-resolution photo).
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
from sqlalchemy import and_, or_
from sqlmodel import col, select as sselect, update as sa_update

from ..clustering.face_identity import (
    ClusterCentroid,
    NewFace,
    OldFace,
    assign_cluster,
    remap_person_ids,
    update_centroid,
)
from ..config import CuratorConfig
from ..db.models import (
    FaceDetection,
    ImageRecord,
    PersonCluster,
    QualityMetric,
    SemanticScore,
)
from ..db.repository import PhotoRepository, dequantize_embedding, quantize_embedding
from ..db.session import get_session
from ..quality.engine import QualityEngine
from ..quality.image_loader import load_image
from .aesthetic import get_head, sync_aesthetic
from .backends import (
    BackendState,
    detect_backend,
    empty_cuda_cache,
    is_oom,
    resolve_decode_workers,
)
from ..clustering.blinks import classify_group
from .face_engine import FaceEngine
from .preference import apply_preferences
from .semantic_engine import SemanticEngine

log = logging.getLogger("photocure.pipeline")

# Statuses that still need work. "faces_done" rows only need Phase B. Inferred rows whose
# faces_schema_version is old also need Phase A again (and stay "inferred" meanwhile).
_PENDING_STATUSES = ("pending", "faces_done")
# Decoded 1600 px frames are ~6 MB each; cap how many Phase A holds at once (~400 MB).
_PHASE_A_MAX_FRAMES = 64
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
    missing: int = 0
    faces_redone: int = 0
    aesthetic_synced: int = 0
    preference_scored: int = 0
    oom_retries: int = 0
    processed_ids: list[int] = field(default_factory=list)
    errors: list = field(default_factory=list)


@dataclass
class _PeopleUndo:
    """What one image's person assignment changed, so a later failure can revert it."""
    cache: dict = field(default_factory=dict)
    restore: dict = field(default_factory=dict)   # pid -> (centroid, face_count) before
    created: list = field(default_factory=list)


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
        .where(ImageRecord.scan_status != "missing",
               ImageRecord.stage == "new",
               ImageRecord.infer_status == "inferred",
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


def _box_iou(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0.0:
        return 0.0
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0.0 else 0.0


class InferencePipeline:
    """Run face + semantic inference over all pending images."""

    def __init__(self, config: CuratorConfig | None = None,
                 backend: BackendState | None = None):
        self.cfg = config or CuratorConfig()
        self.bs = backend or detect_backend(self.cfg)
        self.bs.retry_downgraded()
        self.quality = QualityEngine(self.cfg.quality)

    _cluster_stage = "library"

    def _faces_stale(self, rec: ImageRecord) -> bool:
        return (self.bs.face_backend != "stub" and self.cfg.face.enabled
                and rec.faces_schema_version < self.cfg.face.schema_version)

    @staticmethod
    def _retire_vanished_trash(session) -> int:
        """A trashed file someone deleted by hand is gone for good: drop it from the trash."""
        gone = 0
        for rec in session.exec(sselect(ImageRecord).where(
                ImageRecord.stage == "trash", ImageRecord.route_status != "purged")).all():
            if not Path(rec.path).exists():
                rec.route_status = "purged"
                gone += 1
        if gone:
            session.commit()
        return gone

    # ------------------------------------------------------------ selection
    @staticmethod
    def _stage_scope(stage: str):
        """Rows a run may touch: unfinished photos, or the trash (whose rows are marked missing)."""
        if stage == "trash":
            return and_(ImageRecord.stage == "trash", ImageRecord.route_status != "purged")
        return and_(ImageRecord.stage == "new", ImageRecord.scan_status != "missing")

    def _select_records(self, repo: PhotoRepository, only_paths: list[str] | None,
                        retry_failed: bool, limit: int, stage: str = "new") -> list[ImageRecord]:
        s = repo.s
        if only_paths is not None:
            paths = list(dict.fromkeys(only_paths))
            recs: list[ImageRecord] = []
            for i in range(0, len(paths), _IN_BATCH):
                part = paths[i:i + _IN_BATCH]
                recs.extend(s.exec(
                    sselect(ImageRecord)
                    .where(col(ImageRecord.path).in_(part))
                    .where(self._stage_scope(stage))
                ).all())
            recs.sort(key=lambda r: r.id or 0)
            return recs[:limit] if limit else recs
        statuses = _PENDING_STATUSES + (("failed",) if retry_failed else ())
        wanted = col(ImageRecord.infer_status).in_(statuses)
        if self.bs.face_backend != "stub" and self.cfg.face.enabled:
            # Photos analysed by an older face pipeline: their boxes cannot be reused.
            wanted = or_(wanted, and_(ImageRecord.infer_status == "inferred",
                                      ImageRecord.faces_schema_version < self.cfg.face.schema_version))
        stmt = sselect(ImageRecord).where(
            self._stage_scope(stage),
            wanted
        ).order_by(col(ImageRecord.id))
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
            cancel_check: Callable[[], bool] | None = None,
            stage: str = "new",
            reanalyze: bool = False,
            ) -> InferResult:
        """Analyse photos of one stage. Finished photos are never analysed; ``stage="trash"``
        with ``reanalyze`` redoes every trashed photo whose file is still there."""
        if stage not in ("new", "trash"):
            raise ValueError(f"cannot analyse photos in the {stage} stage")
        # People found in the trash stay in their own clusters, apart from the library's.
        self._cluster_stage = "trash" if stage == "trash" else "library"
        db = db_path or self.cfg.db_path
        result = InferResult()
        t0 = time.monotonic()

        with get_session(db) as session:
            # Keep ORM rows usable across the per-chunk commits below.
            session.expire_on_commit = False
            repo = PhotoRepository(session)
            if stage == "trash":
                result.missing = self._retire_vanished_trash(session)
            if reanalyze:
                session.exec(sa_update(ImageRecord).where(
                    self._stage_scope(stage),
                    ImageRecord.infer_status == "inferred",
                ).values(infer_status="pending", faces_schema_version=0))
                session.commit()
            if stage == "new" and reanalyze_stub and self.bs.semantic_backend != "stub":
                # A real model is available now: redo what the stub analysed.
                result.reanalyzed_stub = mark_stub_rows_for_reanalysis(session)
                if result.reanalyzed_stub:
                    log.info("re-analysing %d photos first scored by the stub",
                             result.reanalyzed_stub)
                    session.commit()
            # Scores from another head, or placeholder scores, are brought in line first.
            result.aesthetic_synced = sync_aesthetic(session)
            if result.aesthetic_synced:
                session.commit()
            pending = self._select_records(repo, only_paths, retry_failed, limit, stage)

            todo: list[ImageRecord] = []
            for rec in pending:
                if rec.infer_status == "inferred" and not self._faces_stale(rec):
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
                              progress, tasks, total_steps, cancel_check=cancel_check)
                if cancel_check and cancel_check():
                    raise InterruptedError("Inference cancelled by user")
                if on_progress:
                    on_progress(len(todo), total_steps,
                                f"Loading CLIP semantic model for {len(todo)} images…")
                self._phase_b(repo, chunks, result, _decode, _on_oom, on_progress,
                              progress, tasks, total_steps, len(todo),
                              cancel_check=cancel_check)
                # New photos get their personal taste score from the active model, if any.
                result.preference_scored = apply_preferences(session, only_missing=True)
                session.commit()
                if on_progress:
                    on_progress(total_steps, total_steps,
                                f"Inference complete: {result.processed} processed")
            finally:
                if progress is not None:
                    progress.stop()
                # Final sweep so VRAM is reclaimed even if a phase raised early.
                gc.collect()
                empty_cuda_cache(self.bs)

        result.duration_s = time.monotonic() - t0
        return result

    # -------------------------------------------------------------- phase A
    def _phase_a(self, repo, chunks, result, decode, on_oom, on_progress,
                 progress, tasks, total_steps,
                 cancel_check: Callable[[], bool] | None = None) -> None:
        """Decode -> faces + quality, committed per chunk, then free pixels."""
        dev = self.cfg.device
        face_max = dev.face_max_side
        face_batch = max(1, int(dev.face_batch_size))
        face_engine = FaceEngine(self.cfg, backend=self.bs)
        done = 0
        cap = min(max(1, int(dev.infer_chunk_size)), _PHASE_A_MAX_FRAMES)
        chunks = [c[i:i + cap] for c in chunks for i in range(0, len(c), cap)]
        # Loaded once so a cluster created or grown earlier in this run is visible to later
        # images, without re-reading every centroid from the DB on each face.
        centroids: dict[int, ClusterCentroid] = {
            c.id: ClusterCentroid(c.id, dequantize_embedding(c.centroid_json) or [], c.face_count)
            for c in repo.all_person_clusters()
            if c.centroid_json and c.face_count > 0 and c.stage == self._cluster_stage
        }
        try:
            for chunk in chunks:
                if cancel_check and cancel_check():
                    raise InterruptedError("Inference cancelled by user")
                # Rows already in "faces_done" only need Phase B.
                a_recs = [r for r in chunk
                          if r.infer_status in ("pending", "failed")
                          or (r.infer_status == "inferred" and self._faces_stale(r))]
                stale_ids = {r.id for r in a_recs if r.infer_status == "inferred"}
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
                    if cancel_check and cancel_check():
                        repo.s.commit()
                        frames.clear()
                        raise InterruptedError("Inference cancelled by user")
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
                        undo = _PeopleUndo()
                        try:
                            emb_ok = self._analyze_crops(face_engine, rec, fr, faces)
                            counts = classify_group([faces], self.cfg.face)
                            q = self.quality.score(fr.bgr)
                            repo.upsert_quality(QualityMetric(
                                image_id=fr.image_id,
                                **{k: v for k, v in q.items() if k != "needs_review"},
                            ))
                            old_ignored = [
                                of for of in repo.faces_for_image(fr.image_id)
                                if getattr(of, "ignored", False) and of.x1 is not None
                            ]
                            ignored_flags = [False] * len(faces)
                            for i, f in enumerate(faces):
                                if getattr(f, "x1", None) is not None:
                                    box = (f.x1, f.y1, f.x2, f.y2)
                                    if any(_box_iou(box, (oi.x1, oi.y1, oi.x2, oi.y2)) >= 0.5 for oi in old_ignored):
                                        ignored_flags[i] = True

                            active_faces = [f for f, ign in zip(faces, ignored_flags) if not ign]
                            if emb_ok:
                                active_pids = self._assign_people(
                                    repo, centroids, fr.image_id, active_faces, undo)
                            else:
                                active_pids = self._keep_people(repo, fr.image_id, active_faces)

                            act_idx = 0
                            person_ids = []
                            for ign in ignored_flags:
                                if ign:
                                    person_ids.append(None)
                                else:
                                    person_ids.append(active_pids[act_idx])
                                    act_idx += 1

                            repo.replace_faces(
                                fr.image_id,
                                [self._face_row(fr.image_id, f, fr.scale, fr.bgr.shape,
                                               embedding=f.embedding, person_id=pid,
                                               ignored=ign)
                                 for f, pid, ign in zip(faces, person_ids, ignored_flags)],
                            )
                            rec.blink_count = counts[0]
                            rec.has_blinks = counts[0] > 0
                            if self.bs.face_backend != "stub" and self.cfg.face.enabled:
                                # Only a real face pass counts; a stub or --no-faces run must
                                # leave the photo eligible for detection later.
                                rec.faces_schema_version = self.cfg.face.schema_version
                            if fr.image_id in stale_ids:
                                result.faces_redone += 1  # CLIP is already done; stays "inferred"
                            else:
                                # Phase B promotes this to "inferred".
                                rec.infer_status = "faces_done"
                        except Exception as e:
                            log.warning("persist failed for %s: %s", rec.path, e)
                            self._undo_people(repo, centroids, undo)
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
                 progress, tasks, total_steps, n_todo,
                 cancel_check: Callable[[], bool] | None = None) -> None:
        """Re-decode each chunk at CLIP size, embed, classify, score, commit."""
        clip_max = self.cfg.device.clip_max_side
        sem_engine = SemanticEngine(self.cfg, backend=self.bs)
        head = get_head()
        done = 0
        try:
            for chunk in chunks:
                if cancel_check and cancel_check():
                    raise InterruptedError("Inference cancelled by user")
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
                    if cancel_check and cancel_check():
                        repo.s.commit()
                        frames.clear()
                        raise InterruptedError("Inference cancelled by user")
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
                        # Placeholder embeddings and the stub head carry no aesthetic signal.
                        aesthetic = (head.score(emb) if backend_name == "clip" and not head.stub
                                     and len(emb) == head.w.shape[0] else None)
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
                            aesthetic=aesthetic,
                            aesthetic_model=head.name if aesthetic is not None else "",
                        ))
                        rec.infer_status = "inferred"
                        result.quality_computed += 1
                        result.semantic_computed += 1
                        result.processed += 1
                        result.processed_ids.append(int(rec.id))
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

    def _analyze_crops(self, engine: FaceEngine, rec: ImageRecord, fr: _LoadedFrame,
                       faces: list) -> bool:
        """Stage 2: measure faces and compute identity embeddings on the full-resolution photo
        (or the frame if it already is). False when embeddings could not be computed at all."""
        if not faces:
            return True
        if fr.scale >= 0.999:
            engine.analyze_faces(fr.bgr, faces)
            engine.compute_embeddings(fr.bgr, faces)
            return engine._ensure_recognizer_loaded() is not None
        try:
            full = load_image(rec.path, max_side=None, orientation=rec.orientation)
        except Exception as e:
            log.warning("full-resolution decode failed for %s (%s); eye state stays unknown", rec.path, e)
            return False
        try:
            engine.analyze_faces(full.bgr, faces)
            engine.compute_embeddings(full.bgr, faces)
        finally:
            del full
        return engine._ensure_recognizer_loaded() is not None

    @staticmethod
    def _keep_people(repo: PhotoRepository, image_id: int, faces: Sequence) -> list[int | None]:
        """Carry person ids and embeddings over by box overlap when none could be computed."""
        old = [o for o in repo.faces_for_image(image_id) if o.person_id is not None]
        out: list[int | None] = [None] * len(faces)
        taken: set[int] = set()
        for i, f in enumerate(faces):
            best, best_iou = None, 0.5
            for o in old:
                if o.id in taken:
                    continue
                iw = min(f.x2, o.x2) - max(f.x1, o.x1)
                ih = min(f.y2, o.y2) - max(f.y1, o.y1)
                if iw <= 0 or ih <= 0:
                    continue
                inter = iw * ih
                union = (f.x2 - f.x1) * (f.y2 - f.y1) + (o.x2 - o.x1) * (o.y2 - o.y1) - inter
                if union > 0 and inter / union >= best_iou:
                    best, best_iou = o, inter / union
            if best is not None:
                taken.add(best.id)
                out[i] = best.person_id
                if f.embedding is None and best.embedding_json:
                    f.embedding = dequantize_embedding(best.embedding_json)
        return out

    @staticmethod
    def _live_cluster(repo: PhotoRepository, centroids: dict[int, ClusterCentroid],
                      pid: int) -> ClusterCentroid | None:
        """Refresh a cached cluster from its DB row; None (and uncached) if merged or deleted."""
        p = repo.s.get(PersonCluster, pid, populate_existing=True)
        if p is None or not p.centroid_json:
            centroids.pop(pid, None)
            return None
        c = ClusterCentroid(pid, dequantize_embedding(p.centroid_json) or [], p.face_count)
        centroids[pid] = c
        return c

    @staticmethod
    def _undo_people(repo: PhotoRepository, centroids: dict[int, ClusterCentroid],
                     undo: _PeopleUndo) -> None:
        """Revert the cluster changes one failed image made, in the DB and the cache."""
        try:
            for pid in undo.created:
                p = repo.s.get(PersonCluster, pid)
                if p is not None:
                    repo.s.delete(p)
            for pid, (centroid, count) in undo.restore.items():
                if repo.s.get(PersonCluster, pid) is not None:
                    repo.update_person_cluster(pid, centroid, count)
        except Exception as e:
            log.warning("could not revert person clusters: %s", e)
        if undo.cache:
            centroids.clear()
            centroids.update(undo.cache)

    def _assign_people(self, repo: PhotoRepository, centroids: dict[int, ClusterCentroid],
                       image_id: int, faces: Sequence,
                       undo: _PeopleUndo | None = None) -> list[int | None]:
        """Resolve a person for each face (same order as ``faces``), mutating ``centroids``.

        A face that remaps to one of this same photo's previous faces keeps that person's
        identity without changing the cluster's face_count or centroid (it is a re-detection
        of the same occurrence, not a new one). The same holds when remap misses (its threshold
        is stricter) but a face still lands, via fresh clustering, on a person this same photo
        already had before: still a continuing occurrence, not a new one. Only a person this
        photo did not already have before is a genuinely new occurrence and grows the cluster it
        joins. Any old identity this photo had that no new face resolves to (remapped or not) has
        lost its occurrence here and the cluster shrinks by one. Faces with no embedding (too
        small or blurry) are left unassigned.
        """
        old = [OldFace(f.person_id, dequantize_embedding(f.embedding_json))
               for f in repo.faces_for_image(image_id)
               if f.person_id is not None and f.embedding_json]
        old_person_ids = {o.person_id for o in old}
        new = [NewFace(i, f.embedding) for i, f in enumerate(faces) if f.embedding is not None]
        remap = remap_person_ids(old, new, self.cfg.face.cluster_remap_threshold)

        undo = undo if undo is not None else _PeopleUndo()
        undo.cache = dict(centroids)
        result: list[int | None] = [None] * len(faces)
        claimed_in_image: set[int] = set()
        resolved_person_ids: set[int] = set()
        # Faces that continue an earlier identity are resolved first, so a face that merely clusters
        # nearby cannot take that person and push the true continuation onto someone else.
        for i in sorted(range(len(faces)), key=lambda k: 0 if k in remap else 1):
            f = faces[i]
            if f.embedding is None:
                continue
            if (i in remap and remap[i] not in claimed_in_image
                    and self._live_cluster(repo, centroids, remap[i]) is not None):
                pid = remap[i]
            else:
                while True:
                    candidates = [c for c in centroids.values() if c.person_id not in claimed_in_image]
                    pid = assign_cluster(f.embedding, candidates, self.cfg.face.cluster_join_threshold)
                    # A cluster merged or deleted mid-run is dropped from the cache and skipped.
                    if pid is None or self._live_cluster(repo, centroids, pid) is not None:
                        break
                if pid is None:
                    created = repo.create_person_cluster(f.embedding, stage=self._cluster_stage)
                    pid = created.id
                    undo.created.append(pid)
                    centroids[pid] = ClusterCentroid(pid, list(f.embedding), 1)
                else:
                    c = centroids[pid]
                    undo.restore.setdefault(pid, (c.centroid, c.face_count))
                    new_centroid = update_centroid(c.centroid, c.face_count, f.embedding)
                    # A person this photo already had keeps their count: this is the same
                    # occurrence continuing (remap's stricter threshold just missed it), not a
                    # second one. Only a person new to this photo grows the cluster.
                    new_count = c.face_count if pid in old_person_ids else c.face_count + 1
                    repo.update_person_cluster(pid, new_centroid, new_count)
                    centroids[pid] = ClusterCentroid(pid, new_centroid, new_count)
            result[i] = pid
            claimed_in_image.add(pid)
            resolved_person_ids.add(pid)

        for lost_pid in old_person_ids - resolved_person_ids:
            c = self._live_cluster(repo, centroids, lost_pid)
            if c is not None:
                undo.restore.setdefault(lost_pid, (c.centroid, c.face_count))
                new_count = max(0, c.face_count - 1)
                if new_count == 0:
                    centroids.pop(lost_pid, None)
                else:
                    centroids[lost_pid] = ClusterCentroid(lost_pid, c.centroid, new_count)
                repo.update_person_cluster(lost_pid, c.centroid, new_count)
            else:
                repo.decrement_person_cluster(lost_pid)  # no centroid cached; no-op if gone
        return result

    @staticmethod
    def _face_row(image_id: int, f, scale: float = 1.0, shape: tuple = (0, 0),
                 embedding: list[float] | None = None, person_id: int | None = None,
                 ignored: bool = False) -> FaceDetection:
        h, w = shape[:2]
        return FaceDetection(
            image_id=image_id,
            # Legacy pixel box in the decoded frame, kept for older readers.
            bbox_x1=f.x1 * w, bbox_y1=f.y1 * h, bbox_x2=f.x2 * w, bbox_y2=f.y2 * h,
            x1=f.x1, y1=f.y1, x2=f.x2, y2=f.y2,
            confidence=f.confidence,
            landmarks=_json_sem([(round(x, 5), round(y, 5)) for x, y in (f.landmarks or [])]),
            scale=float(scale),
            face_sharpness=f.face_sharpness,
            eye_closed_score=f.eye_closed_score,
            eye_status=f.eye_status,
            is_blink=bool(getattr(f, "is_blink", False)),
            embedding_json=quantize_embedding(embedding) if embedding else None,
            person_id=person_id,
            ignored=ignored,
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
                  reanalyze_stub: bool = True,
                  cancel_check: Callable[[], bool] | None = None,
                  stage: str = "new", reanalyze: bool = False) -> InferResult:
    cfg = config or CuratorConfig.load(db_path)
    return InferencePipeline(cfg).run(db_path=db_path, only_paths=only_paths,
                                      show_progress=show_progress,
                                      retry_failed=retry_failed, limit=limit,
                                      reanalyze_stub=reanalyze_stub,
                                      cancel_check=cancel_check,
                                      stage=stage, reanalyze=reanalyze)
