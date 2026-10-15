"""Face detection engine (InsightFace SCRFD primary, OpenCV-Haar fallback, stub).

Produces per-face bounding boxes, confidence, 5-point landmarks, an
eye-openness (EAR) score when 106-point landmarks exist, an optional
recognition embedding and a face-crop sharpness (Laplacian variance).

Backends:
- ``insightface``: SCRFD detector (needs the ``buffalo_l`` model bundle; the
  CUDAExecutionProvider is used when CUDA is present).
- ``cv2``: OpenCV Haar cascade (ships with the opencv wheel, no download) -
  gives boxes only, so eye openness is reported as unknown (None).
- ``stub``: deterministic, zero-dependency backend used in CI so the pipeline
  and tests always run. It reports no faces (so face routing is skipped) but is
  fully exercised for correctness of the surrounding logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections.abc import Sequence

import cv2
import numpy as np

from ..config import CuratorConfig
from .backends import (
    BackendState,
    detect_backend,
    empty_cuda_cache,
    is_oom,
    select_onnx_providers,
)

log = logging.getLogger("photocure.face")

# 5-point landmark order for InsightFace/SCRFD:
#   0 right eye, 1 left eye, 2 nose, 3 right mouth, 4 left mouth
# (x, y) normalized to 0..1 in the *original* image space.


@dataclass
class FaceInfo:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    eye_aspect_ratio: float | None = None
    face_sharpness: float | None = None
    # L2-normalized recognition embedding (InsightFace recognition module).
    embedding: list[float] | None = None

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)


def eye_aspect_ratio(points: Sequence[Sequence[float]]) -> float:
    """Canonical 6-point EAR (spec section 2.D).

    ``points`` must be the 6 eye-contour points in order
    ``[p1, p2, p3, p4, p5, p6]`` (p1/p4 are the horizontal pair, p2/p6 and
    p3/p5 the vertical pairs)::

        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    if len(points) < 6:
        return 0.0
    p = [np.asarray(p, dtype=float) for p in points[:6]]
    denom = 2.0 * float(np.linalg.norm(p[0] - p[3]))
    if denom < 1e-6:
        return 0.0
    num = float(np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4]))
    return float(num / denom)


# Haar cascades give no calibrated score, so report a value below the face gate.
HAAR_CONFIDENCE = 0.5

# 106-point (2d106det) eye contours ordered [corner, top, top, corner, bottom, bottom].
# These indices follow the commonly used InsightFace layout and are guarded by a
# plausibility check; verify against the model bundle when changing it.
_LEFT_EYE_106 = (35, 41, 40, 42, 39, 37)
_RIGHT_EYE_106 = (89, 95, 94, 96, 93, 91)
_EAR_PLAUSIBLE = (0.02, 0.8)


def eye_openness_from_106(lm106) -> float | None:
    """Mean canonical EAR of both eyes from 106 landmarks, or None if unusable."""
    try:
        pts = np.asarray(lm106, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 106 or pts.shape[1] < 2:
            return None
        ears = [eye_aspect_ratio([pts[i][:2] for i in idx])
                for idx in (_LEFT_EYE_106, _RIGHT_EYE_106)]
    except Exception:
        return None
    if any(e <= 0.0 for e in ears):
        return None
    ear = float(np.mean(ears))
    lo, hi = _EAR_PLAUSIBLE
    return ear if lo <= ear <= hi else None


def _face_attr(face, name: str):
    """Read ``name`` from an InsightFace Face (dict subclass) or plain object."""
    try:
        return face[name]
    except Exception:
        pass
    try:
        return getattr(face, name)
    except Exception:
        return None


def _face_sharpness(bgr: np.ndarray, f: FaceInfo) -> float:
    """Laplacian variance of the face crop (focus proxy for the face region)."""
    h, w = bgr.shape[:2]
    x0, y0 = max(0, int(f.x1)), max(0, int(f.y1))
    x1, y1 = min(w, int(f.x2)), min(h, int(f.y2))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    crop = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


class FaceEngine:
    """Batched face detector with automatic backend selection."""

    def __init__(self, config: CuratorConfig | None = None,
                 backend: BackendState | None = None):
        self.cfg = config or CuratorConfig()
        self.bs = backend or detect_backend(self.cfg)
        self._insightface = None
        self._haar: cv2.CascadeClassifier | None = None

    # ------------------------------------------------------------- lifecycle
    def _ensure_loaded(self) -> None:
        if self.bs.face_backend == "insightface":
            if self._insightface is None:
                self._insightface = self._load_insightface()
                if self._insightface is None:
                    log.warning("InsightFace unavailable; falling back to cv2")
                    self.bs.face_backend = "cv2"
                    self._ensure_loaded()
                    return
        elif self.bs.face_backend == "cv2":
            if self._haar is None:
                self._haar = self._load_haar()
                if self._haar is None:
                    self.bs.face_backend = "stub"

    def _load_insightface(self):
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            log.warning("InsightFace import failed (%s: %s); falling back",
                        e.__class__.__name__, e)
            return None
        providers = select_onnx_providers(self.bs.accelerator)
        # ctx_id below zero would force the CPU provider in InsightFace.
        ctx_id = 0 if any(p != "CPUExecutionProvider" for p in providers) else -1
        threads = int(getattr(self.cfg.device, "torch_threads", 0) or 0)
        size = self.cfg.device.face_max_side
        module_sets = (["detection", "landmark_2d_106", "recognition"],
                       ["detection"])
        for modules in module_sets:
            try:
                kwargs = {"name": "buffalo_l", "allowed_modules": modules,
                          "providers": providers}
                app = None
                if threads > 0:
                    # Best effort: not every insightface build forwards session options.
                    try:
                        import onnxruntime as ort
                        so = ort.SessionOptions()
                        so.intra_op_num_threads = threads
                        app = FaceAnalysis(sess_options=so, **kwargs)
                    except Exception as e:
                        log.info("onnxruntime thread option not applied (%s)", e)
                        app = None
                if app is None:
                    app = FaceAnalysis(**kwargs)
                app.prepare(ctx_id=ctx_id, det_size=(size, size))
                log.info("InsightFace ready (modules=%s, ctx_id=%s)", modules, ctx_id)
                return app
            except Exception as e:
                log.warning("InsightFace load with %s failed (%s: %s)",
                            modules, e.__class__.__name__, e)
        return None

    def _load_haar(self):
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cc = cv2.CascadeClassifier(path)
            if cc.empty():
                return None
            log.info("OpenCV Haar cascade ready")
            return cc
        except Exception:
            return None

    def release(self) -> None:
        """Drop model references (lets the pipeline swap to the CLIP phase)."""
        self._insightface = None
        self._haar = None
        empty_cuda_cache(self.bs)

    # -------------------------------------------------------------- detection
    def detect_faces_batch(self,
                           images: Sequence[np.ndarray],
                           ) -> list[list[FaceInfo]]:
        """Detect faces in a list of BGR images; returns per-image face lists."""
        self._ensure_loaded()
        results: list[list[FaceInfo]] = []
        if self.bs.face_backend == "stub":
            results = [[] for _ in images]
            return results
        for bgr in images:
            results.append(self._detect_one(bgr))
        return results

    def _detect_one(self, bgr: np.ndarray) -> list[FaceInfo]:
        self._ensure_loaded()
        h, w = bgr.shape[:2]
        if self.bs.face_backend == "insightface" and self._insightface is not None:
            return self._detect_insightface(bgr, w, h)
        if self.bs.face_backend == "cv2" and self._haar is not None:
            return self._detect_haar(bgr, w, h)
        return []

    def _detect_insightface(self, bgr: np.ndarray, w: int, h: int) -> list[FaceInfo]:
        out: list[FaceInfo] = []
        try:
            faces = self._insightface.get(bgr)
        except Exception as e:
            if is_oom(e):
                raise  # let the pipeline shrink the batch
            log.warning("insightface.get failed: %s", e)
            return out
        for f in faces:
            x1, y1, x2, y2 = [float(v) for v in _face_attr(f, "bbox")]
            conf = float(_face_attr(f, "det_score"))
            kps = _face_attr(f, "kps")
            lm = [(float(p[0]), float(p[1])) for p in np.asarray(kps)] \
                if kps is not None else []
            fi = FaceInfo(x1, y1, x2, y2, conf, landmarks=lm)
            lm106 = _face_attr(f, "landmark_2d_106")
            if lm106 is not None:
                fi.eye_aspect_ratio = eye_openness_from_106(lm106)
            emb = _face_attr(f, "normed_embedding")
            if emb is not None:
                fi.embedding = [float(v) for v in np.asarray(emb).ravel()]
            fi.face_sharpness = _face_sharpness(bgr, fi)
            out.append(fi)
        return out

    def _detect_haar(self, bgr: np.ndarray, w: int, h: int) -> list[FaceInfo]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = self._haar.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(48, 48))
        out: list[FaceInfo] = []
        for (x, y, bw, bh) in boxes:
            fi = FaceInfo(float(x), float(y), float(x + bw), float(y + bh),
                          confidence=HAAR_CONFIDENCE)
            fi.face_sharpness = _face_sharpness(bgr, fi)
            out.append(fi)
        return out


def detect_faces(images: Sequence[np.ndarray],
                 config: CuratorConfig | None = None) -> list[list[FaceInfo]]:
    return FaceEngine(config).detect_faces_batch(images)
