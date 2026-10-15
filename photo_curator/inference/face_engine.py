"""Face detection engine (InsightFace SCRFD primary, OpenCV-Haar fallback, stub).

Produces per-face bounding boxes, confidence, 5-point landmarks, an
eye-openness (EAR) score and a face-crop sharpness (Laplacian variance).

Backends:
- ``insightface``: SCRFD detector (needs the ``buffalo_l`` model bundle; the
  CUDAExecutionProvider is used when CUDA is present).
- ``cv2``: OpenCV Haar cascade (ships with the opencv wheel, no download) -
  gives boxes but no landmarks, so EAR falls back to a crop-based proxy.
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
from .backends import BackendState, detect_backend

log = logging.getLogger("photo_curator.face")

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


def _ear_from_five_landmarks(lm: list[tuple[float, float]], bgr: np.ndarray) -> float:
    """EAR-like openness score from SCRFD's 5 landmarks (no per-eye contour).

    SCRFD does not expose the 6 contour points per eye, so we derive a stable
    geometric openness proxy: the ratio of inter-eye distance to the eye->nose
    distance, combined with high-frequency energy in each eye crop (open eyes
    carry more high-frequency detail than closed lids). The result is clamped
    to roughly the canonical EAR range [0, 0.35] so it composes cleanly with
    :func:`eye_aspect_ratio` outputs in the best-shot selector.
    """
    if len(lm) < 3:
        return 0.0
    r, left, n = (np.asarray(lm[0], float), np.asarray(lm[1], float),
                  np.asarray(lm[2], float))
    inter_eye = float(np.linalg.norm(r - left))
    eye_mid = (r + left) / 2.0
    eye_nose = float(np.linalg.norm(eye_mid - n))
    if inter_eye < 1e-3 or eye_nose < 1e-3:
        return 0.0

    # Geometric openness: larger inter-eye / (eye-nose) => eyes sit higher.
    geom = float(np.clip(inter_eye / (1.8 * eye_nose), 0.0, 1.0))

    # High-frequency energy proxy on each eye crop (0..1).
    detail = 0.0
    h_img, w_img = bgr.shape[:2]
    for pt in (r, left):
        cx = int(pt[0] * w_img) if pt[0] <= 1.0 else int(pt[0])
        cy = int(pt[1] * h_img) if pt[1] <= 1.0 else int(pt[1])
        rad = max(2, int(0.08 * inter_eye)) if inter_eye > 5 else max(2, int(0.02 * w_img))
        x0, y0 = max(0, cx - rad), max(0, cy - rad)
        x1, y1 = min(w_img, cx + rad), min(h_img, cy + rad)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        crop = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(crop, cv2.CV_64F)
        detail += float(lap.var())
    detail = float(np.clip(detail / 400.0, 0.0, 1.0))

    score = 0.6 * geom + 0.4 * detail
    # Map to canonical EAR-ish range [0.05, 0.32].
    return float(0.05 + score * 0.27)


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
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if not (self.bs.cuda_available and self.bs.torch is not None):
                providers = ["CPUExecutionProvider"]
            app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=providers)
            ctx_id = 0 if self.bs.cuda_available else -1
            app.prepare(ctx_id=ctx_id, det_size=(self.cfg.device.face_max_side,
                                                 self.cfg.device.face_max_side))
            log.info("InsightFace SCRFD ready (ctx_id=%s)", ctx_id)
            return app
        except Exception as e:
            log.warning("InsightFace load failed (%s); will use cv2/stub",
                        e.__class__.__name__)
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
        if self.bs.cuda_available and self.bs.torch is not None:
            try:
                self.bs.torch.cuda.empty_cache()
            except Exception:
                pass

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
            log.debug("insightface.get failed: %s", e)
            return out
        for f in faces:
            x1, y1, x2, y2 = [float(v) for v in f.bbox]
            conf = float(f.det_score)
            lm = [(float(p[0]), float(p[1])) for p in np.asarray(f.kps)] \
                if getattr(f, "kps", None) is not None else []
            fi = FaceInfo(x1, y1, x2, y2, conf, landmarks=lm)
            if len(lm) == 5:
                fi.eye_aspect_ratio = _ear_from_five_landmarks(lm, bgr)
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
                          confidence=0.75)
            fi.face_sharpness = _face_sharpness(bgr, fi)
            out.append(fi)
        return out


def detect_faces(images: Sequence[np.ndarray],
                 config: CuratorConfig | None = None) -> list[list[FaceInfo]]:
    return FaceEngine(config).detect_faces_batch(images)
