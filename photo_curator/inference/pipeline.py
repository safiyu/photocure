"""Sequential GPU inference pipeline (Phase A: faces, Phase B: CLIP).

Orchestrates the two model families without ever holding both in VRAM at once:
    1. Phase A - FaceEngine (InsightFace/SCRFD or fallback).
    2. Memory sweep - ``gc.collect()`` + ``torch.cuda.empty_cache()``.
    3. Phase B - SemanticEngine (OpenCLIP).

Quality metrics are computed on the same decoded frame and persisted alongside
the inference results. Every image is marked ``inferred`` (or ``failed``) so the
scan is fully resumable.
"""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

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
from .backends import BackendState, detect_backend
from .face_engine import FaceEngine
from .semantic_engine import SemanticEngine

log = logging.getLogger("photo_curator.pipeline")

# Longest side used for the *quality* decode (cheap, resolution-normalized).
QUALITY_MAX_SIDE = 1280
# Longest side for face detection (needs enough detail for landmarks).
FACE_MAX_SIDE = 1280


@dataclass
class InferResult:
    processed: int = 0
    faces_total: int = 0
    quality_computed: int = 0
    semantic_computed: int = 0
    failures: int = 0
    duration_s: float = 0.0
    skipped_already_inferred: int = 0
    errors: list = field(default_factory=list)


@dataclass
class _LoadedFrame:
    image_id: int
    bgr: np.ndarray


class InferencePipeline:
    """Run face + semantic inference over all pending images."""

    def __init__(self, config: CuratorConfig | None = None,
                 backend: Optional[BackendState] = None):
        self.cfg = config or CuratorConfig()
        self.bs = backend or detect_backend(self.cfg)
        self.quality = QualityEngine(self.cfg.quality)

    # ------------------------------------------------------------------ run
    def run(self, db_path: Path | str | None = None,
            only_paths: Optional[List[str]] = None,
            on_progress: Optional[Callable[[int, int, str], None]] = None,
            show_progress: bool = True,
            ) -> InferResult:
        db = db_path or self.cfg.db_path
        result = InferResult()
        t0 = time.monotonic()

        with get_session(db) as session:
            repo = PhotoRepository(session)
            if only_paths is not None:
                pending = [r for r in repo.all_images() if r.path in set(only_paths)]
            else:
                pending = repo.images_pending_inference()

            # Skip already-inferred.
            todo: List[ImageRecord] = []
            for rec in pending:
                if rec.infer_status == "inferred":
                    result.skipped_already_inferred += 1
                else:
                    todo.append(rec)

            if not only_paths:
                # images_pending_inference() only returns "pending" rows, so
                # already-inferred images never appear in ``pending``. Count
                # them separately to make the resumption skip observable.
                result.skipped_already_inferred = repo.count_inferred()

            if not todo:
                return result

            face_max = self.cfg.device.face_max_side
            clip_max = self.cfg.device.clip_max_side
            frames: List[_LoadedFrame] = []
            load_failed: list = []

            def _load(rec: ImageRecord) -> Optional[_LoadedFrame]:
                try:
                    img = load_image(rec.path, max_side=face_max,
                                     orientation=rec.orientation)
                    return _LoadedFrame(image_id=rec.id, bgr=img.bgr)
                except Exception as e:
                    log.warning("decode failed for %s: %s", rec.path, e)
                    return None

            tasks = {}
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
                tasks["decode"] = progress.add_task("Decode", total=len(todo))
                tasks["faces"] = progress.add_task("Faces (Phase A)", total=0)
                tasks["clip"] = progress.add_task("CLIP (Phase B)", total=0)
                progress.start()
            else:
                progress = None

            for i, rec in enumerate(todo):
                fr = _load(rec)
                if fr is None:
                    load_failed.append(rec)
                else:
                    frames.append(fr)
                if show_progress:
                    progress.update(tasks["decode"], completed=i + 1)

            if show_progress:
                progress.update(tasks["faces"], total=len(frames))
                progress.update(tasks["clip"], total=len(frames))

            # ------------------------------------------------ Phase A: faces
            face_engine = FaceEngine(self.cfg, backend=self.bs)
            face_by_id: dict = {}
            try:
                # Face engine already detects per-image; feed the batch.
                faces_batch = face_engine.detect_faces_batch(
                    [f.bgr for f in frames])
                for fr, faces in zip(frames, faces_batch):
                    face_by_id[fr.image_id] = faces
                    result.faces_total += len(faces)
                    if show_progress:
                        progress.advance(tasks["faces"])
            finally:
                face_engine.release()
                gc.collect()
                if self.bs.cuda_available and self.bs.torch is not None:
                    try:
                        self.bs.torch.cuda.empty_cache()
                    except Exception:
                        pass

            # ------------------------------------------------ Phase B: CLIP
            sem_engine = SemanticEngine(self.cfg, backend=self.bs)
            sem_by_id: dict = {}
            try:
                sems_batch = sem_engine.classify_and_embed([f.bgr for f in frames])
                for fr, sem in zip(frames, sems_batch):
                    sem_by_id[fr.image_id] = sem
                    if show_progress:
                        progress.advance(tasks["clip"])
            finally:
                sem_engine.release()
                gc.collect()
                if self.bs.cuda_available and self.bs.torch is not None:
                    try:
                        self.bs.torch.cuda.empty_cache()
                    except Exception:
                        pass

            # ------------------------------------------------ persist
            rec_by_id = {r.id: r for r in todo}
            for fr in frames:
                rec = rec_by_id.get(fr.image_id)
                if rec is None:
                    continue
                try:
                    # Quality on the decoded frame.
                    q = self.quality.score(fr.bgr)
                    repo.upsert_quality(QualityMetric(
                        image_id=fr.image_id,
                        **{k: v for k, v in q.items() if k != "needs_review"},
                    ))
                    # Faces.
                    faces = face_by_id.get(fr.image_id, [])
                    repo.replace_faces(
                        fr.image_id,
                        [self._face_row(fr.image_id, f) for f in faces],
                    )
                    # Semantic.
                    sem = sem_by_id.get(fr.image_id)
                    if sem is not None:
                        repo.upsert_semantic(SemanticScore(
                            image_id=fr.image_id,
                            model_name=(
                                self.cfg.taxonomy.clip_model_name
                                if self.bs.semantic_backend == "clip" else "stub"
                            ),
                            top1_category=sem.top1.value,
                            top1_score=sem.top1_score,
                            top2_category=sem.top2.value,
                            top2_score=sem.top2_score,
                            score_margin=sem.margin,
                            distribution=_json_sem(sem.distribution),
                            embedding=quantize_embedding(sem.embedding or []),
                        ))
                    rec.infer_status = "inferred"
                    result.quality_computed += 1
                    result.semantic_computed += 1
                    result.processed += 1
                    if on_progress:
                        on_progress(result.processed, len(frames), rec.path)
                except Exception as e:
                    log.warning("persist failed for %s: %s", rec.path, e)
                    rec.infer_status = "failed"
                    result.failures += 1
                    result.errors.append(f"{rec.path}: {e}")

            for rec in load_failed:
                rec.infer_status = "failed"
                result.failures += 1
                result.errors.append(f"{rec.path}: decode failure")

            if show_progress and progress is not None:
                progress.stop()

        result.duration_s = time.monotonic() - t0
        return result

    @staticmethod
    def _face_row(image_id: int, f) -> FaceDetection:
        lm = f.landmarks or []
        return FaceDetection(
            image_id=image_id,
            bbox_x1=f.x1, bbox_y1=f.y1, bbox_x2=f.x2, bbox_y2=f.y2,
            confidence=f.confidence,
            landmarks=_json_sem([(round(x, 5), round(y, 5)) for x, y in lm]),
            eye_aspect_ratio=f.eye_aspect_ratio,
            face_sharpness=f.face_sharpness,
        )


def _json_sem(obj) -> str:
    import json
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return "[]"


def run_inference(config: CuratorConfig | None = None,
                  db_path: Path | str | None = None,
                  only_paths: Optional[List[str]] = None,
                  show_progress: bool = True) -> InferResult:
    cfg = config or CuratorConfig.load(db_path)
    return InferencePipeline(cfg).run(db_path=db_path, only_paths=only_paths,
                                      show_progress=show_progress)
