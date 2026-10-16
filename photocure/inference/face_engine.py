"""Face detection and eye measurement with permissively licensed models.

Stage 1 finds faces with OpenCV's YuNet on a frame no larger than ``face_max_side``.
Stage 2 maps each box back to the full-resolution, EXIF-oriented photo, cuts a crop with a
margin, measures its sharpness, and asks MediaPipe's Face Landmarker how closed the eyes are.

Backends: ``yunet`` (the bundled detector) and ``stub`` (no faces, for tests and CI).
Without MediaPipe, faces are still found and every eye state stays ``unknown``. Unknown is
never treated as open or closed.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import models
from ..config import CuratorConfig
from .backends import BackendState, detect_backend, empty_cuda_cache

log = logging.getLogger("photocure.face")


@dataclass
class FaceInfo:
    # Box and landmarks as fractions (0..1) of the EXIF-oriented image.
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    # YuNet's 5 points: right eye, left eye, nose, right mouth corner, left mouth corner.
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    face_sharpness: float | None = None
    # (eyeBlinkLeft + eyeBlinkRight) / 2, or None when it could not be measured.
    eye_closed_score: float | None = None
    eye_status: str = "unknown"
    is_blink: bool = False

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)


def crop_face(bgr: np.ndarray, box: tuple[float, float, float, float], margin: float) -> np.ndarray:
    """Crop a normalized box from ``bgr`` with ``margin`` (a fraction of the box size) on each side."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    px1, py1 = max(0, int(round((x1 - mx) * w))), max(0, int(round((y1 - my) * h)))
    px2, py2 = min(w, int(round((x2 + mx) * w))), min(h, int(round((y2 + my) * h)))
    return bgr[py1:py2, px1:px2]


def shrink_to(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr
    k = max_side / longest
    return cv2.resize(bgr, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)


def standardized_sharpness(crop: np.ndarray, width: int) -> float:
    """Laplacian variance of the crop resized to a fixed width, so faces of any size compare."""
    h, w = crop.shape[:2]
    if w < 8 or h < 8:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FaceEngine:
    def __init__(self, config: CuratorConfig | None = None, backend: BackendState | None = None):
        self.cfg = config or CuratorConfig()
        self.bs = backend or detect_backend(self.cfg)
        self._yunet = None
        self._landmarker = None
        self._landmarker_failed = False

    # ------------------------------------------------------------- lifecycle
    def _ensure_loaded(self) -> None:
        if self.bs.face_backend != "yunet" or self._yunet is not None:
            return
        path = models.model_path(models.YUNET)
        if path is None or not hasattr(cv2, "FaceDetectorYN"):
            log.warning("YuNet is unavailable (model missing, altered, or OpenCV too old); "
                        "no faces will be detected")
            self.bs.face_backend = "stub"
            return
        f = self.cfg.face
        self._yunet = cv2.FaceDetectorYN.create(str(path), "", (320, 320), f.detect_score_min,
                                                f.nms_threshold, f.max_faces)

    def _load_landmarker(self):
        if self._landmarker is not None or self._landmarker_failed:
            return self._landmarker
        path = models.model_path(models.LANDMARKER)
        try:
            if path is None:
                raise FileNotFoundError("face_landmarker.task is missing or altered")
            import mediapipe as mp  # optional extra
            from mediapipe.tasks.python import BaseOptions, vision
            f = self.cfg.face
            opts = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=f.landmark_num_faces,
                output_face_blendshapes=True,
                min_face_detection_confidence=f.landmark_min_confidence,
                min_face_presence_confidence=f.landmark_min_confidence)
            self._landmarker = vision.FaceLandmarker.create_from_options(opts)
            self._mp = mp
        except Exception as e:
            self._landmarker_failed = True
            log.info("Eye state will be reported as unknown (%s: %s)", e.__class__.__name__, e)
        return self._landmarker

    @property
    def blink_available(self) -> bool:
        return self._load_landmarker() is not None

    def release(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
        self._yunet = self._landmarker = None
        self._landmarker_failed = False
        empty_cuda_cache(self.bs)

    # ------------------------------------------------------ stage 1: detect
    def detect_faces_batch(self, images: Sequence[np.ndarray]) -> list[list[FaceInfo]]:
        """Find faces in each BGR image. Boxes are fractions, so any decode size works."""
        if not self.cfg.face.enabled:
            return [[] for _ in images]
        self._ensure_loaded()
        if self.bs.face_backend != "yunet" or self._yunet is None:
            return [[] for _ in images]
        return [self._detect_one(bgr) for bgr in images]

    def _detect_one(self, bgr: np.ndarray) -> list[FaceInfo]:
        small = shrink_to(bgr, int(self.cfg.device.face_max_side))
        h, w = small.shape[:2]
        self._yunet.setInputSize((w, h))
        _, rows = self._yunet.detect(small)
        out: list[FaceInfo] = []
        if rows is None:
            return out

        def clip(v: float) -> float:
            return float(min(1.0, max(0.0, v)))

        for r in rows:
            x, y, bw, bh = (float(v) for v in r[:4])
            lm = [(clip(float(r[4 + 2 * i]) / w), clip(float(r[5 + 2 * i]) / h)) for i in range(5)]
            out.append(FaceInfo(clip(x / w), clip(y / h), clip((x + bw) / w), clip((y + bh) / h),
                                confidence=float(r[14]), landmarks=lm))
        return out

    # --------------------------------------------------- stage 2: crop + eyes
    def analyze_faces(self, full_bgr: np.ndarray, faces: Sequence[FaceInfo]) -> None:
        """Fill sharpness and eye_closed_score from the full-resolution frame, in place."""
        f = self.cfg.face
        H, W = full_bgr.shape[:2]
        for face in faces:
            if face.w * W < f.min_px:  # too few pixels to judge eyes; never guess
                face.face_sharpness, face.eye_closed_score, face.eye_status = None, None, "unknown"
                continue
            crop = crop_face(full_bgr, (face.x1, face.y1, face.x2, face.y2), f.crop_margin)
            if crop.size == 0:
                continue
            face.face_sharpness = standardized_sharpness(crop, f.sharpness_width)
            face.eye_closed_score = self._eye_closed_score(shrink_to(crop, f.landmark_max_side))

    def _eye_closed_score(self, crop: np.ndarray) -> float | None:
        lm = self._load_landmarker()
        if lm is None:
            return None
        try:
            rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            res = lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        except Exception as e:
            log.warning("landmarker failed on a crop (%s: %s)", e.__class__.__name__, e)
            return None
        if not res.face_landmarks or not res.face_blendshapes:
            return None
        # A crop can include a neighbour: take the face nearest the centre.
        centres = [(np.mean([p.x for p in pts]) - 0.5) ** 2 + (np.mean([p.y for p in pts]) - 0.5) ** 2
                   for pts in res.face_landmarks]
        scores = {c.category_name: c.score for c in res.face_blendshapes[int(np.argmin(centres))]}
        if "eyeBlinkLeft" not in scores or "eyeBlinkRight" not in scores:
            return None
        return float((scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2.0)


def detect_faces(images: Sequence[np.ndarray], config: CuratorConfig | None = None) -> list[list[FaceInfo]]:
    return FaceEngine(config).detect_faces_batch(images)
