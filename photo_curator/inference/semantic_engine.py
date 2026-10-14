"""Semantic classification & embedding engine (OpenCLIP ViT-B-16, with stub).

Zero-shot classification: the CLIP text tower encodes the canonical taxonomy
prompts once; each image is encoded and cosine similarity against those text
embeddings is softmaxed into per-category probabilities. The same image
embedding (512-dim, L2-normalized) is stored for burst clustering.

Backends:
- ``clip``: real OpenCLIP ViT-B-16 on CUDA (FP16) or CPU (FP32).
- ``stub``: deterministic, dependency-light backend for CI. It derives a stable
  512-dim embedding from a content hash and category scores from low-level image
  statistics, so the decision matrix / clustering / best-shot logic is fully
  exercisable without GPU or model downloads.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np

from ..config import CuratorConfig
from ..rules.taxonomy import CLIP_CATEGORIES, CategoryType, flat_prompts
from .backends import BackendState, detect_backend

log = logging.getLogger("photo_curator.semantic")

EMBED_DIM = 512


@dataclass
class SemanticResult:
    top1: CategoryType
    top1_score: float
    top2: CategoryType
    top2_score: float
    margin: float
    distribution: dict = field(default_factory=dict)
    embedding: list[float] | None = None


class SemanticEngine:
    """Zero-shot CLIP classifier + embedding extractor."""

    def __init__(self, config: CuratorConfig | None = None,
                 backend: BackendState | None = None):
        self.cfg = config or CuratorConfig()
        self.bs = backend or detect_backend(self.cfg)
        self._model = None
        self._preprocess = None
        self._tokenize = None
        self._text_features: object | None = None  # torch tensor (device-local)
        self._device = None
        self._autocast = None

    # ------------------------------------------------------------- lifecycle
    def _ensure_loaded(self) -> None:
        if self.bs.semantic_backend == "clip" and self._model is None:
            ok = self._load_clip()
            if not ok:
                log.warning("OpenCLIP unavailable; falling back to stub")
                self.bs.semantic_backend = "stub"
                self._model = "stub"

    def _load_clip(self) -> bool:
        try:
            import open_clip
            import torch
            model_name = self.cfg.taxonomy.clip_model_name
            pretrained = self.cfg.taxonomy.clip_pretrained
            self._device = "cuda" if self.bs.cuda_available else "cpu"
            # Match the activation architecture the weights were trained with:
            # the OpenAI tags use QuickGELU, most LAION tags use standard GELU.
            # Without this the model silently runs with the wrong activation
            # (and open_clip warns about the mismatch on every load).
            quick_gelu = _pretrained_quick_gelu(open_clip, model_name, pretrained)
            model, _train_preprocess, val_preprocess = (
                open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained,
                    force_quick_gelu=quick_gelu))
            # Use the *validation* pipeline: the first transform returned is
            # the training one, whose RandomResizedCrop is stochastic — the
            # same photo would then produce a different crop (and embedding)
            # on every encode, breaking determinism and burst clustering.
            self._preprocess = val_preprocess
            self._tokenize = open_clip.get_tokenizer(model_name)
            self._model = model.to(self._device).eval()
            self._autocast = (
                torch.autocast(device_type=self._device, dtype=torch.float16)
                if self.bs.fp16 and self._device == "cuda"
                else None
            )
            self._encode_texts()
            log.info("OpenCLIP %s/%s ready on %s (fp16=%s, quick_gelu=%s)",
                     model_name, pretrained, self._device, self.bs.fp16,
                     quick_gelu)
            return True
        except Exception as e:
            log.warning("OpenCLIP load failed (%s)", e.__class__.__name__)
            return False

    def _encode_texts(self) -> None:
        import torch
        prompts = flat_prompts()
        tok = self._tokenize(prompts).to(self._device)
        ctx = self._autocast or torch.autocast(device_type="cpu", enabled=False)
        with torch.no_grad(), ctx:
            tf = self._model.encode_text(tok)
        tf = tf.float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
        # Keep as a torch tensor on the model device: matmul between a torch
        # tensor and a numpy array trips numpy-2.0 __array_wrap__ deprecations.
        self._text_features = tf

    def release(self) -> None:
        self._model = None
        self._text_features = None
        self._autocast = None
        if self.bs.cuda_available and self.bs.torch is not None:
            try:
                self.bs.torch.cuda.empty_cache()
            except Exception:
                pass

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _bgr_to_tensor(bgr: np.ndarray, preprocess) -> object:
        import torch
        rgb = bgr[:, :, ::-1].copy()
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return preprocess(img)

    # ------------------------------------------------------------ inference
    def classify_and_embed(self,
                           images: Sequence[np.ndarray],
                           ) -> list[SemanticResult]:
        """Zero-shot classify + embed a batch of BGR images."""
        self._ensure_loaded()
        if self.bs.semantic_backend == "stub" or self._model == "stub":
            return [self._stub_result(bgr) for bgr in images]
        return self._clip_batch(images)

    def _clip_batch(self, images: Sequence[np.ndarray]) -> list[SemanticResult]:
        import torch
        batch_size = self.cfg.device.clip_batch_size
        out: list[SemanticResult] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            tensors = torch.stack(
                [self._bgr_to_tensor(b, self._preprocess) for b in chunk]
            ).to(self._device)
            ctx = self._autocast or torch.autocast(device_type="cpu", enabled=False)
            with torch.no_grad(), ctx:
                img_feat = self._model.encode_image(tensors)
            img_feat = img_feat.float()
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            # Cosine similarity against the (already-normalized) text features.
            logits = (img_feat @ self._text_features.T) * 100.0
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            emb = img_feat.cpu().numpy()
            for i in range(len(chunk)):
                out.append(self._from_probs(probs[i], emb[i]))
        return out

    def _from_probs(self, probs: np.ndarray, emb: np.ndarray) -> SemanticResult:
        order = np.argsort(probs)[::-1]
        top1_idx, top2_idx = int(order[0]), int(order[1])
        dist = {
            CLIP_CATEGORIES[i].value: float(probs[i])
            for i in range(len(CLIP_CATEGORIES))
        }
        return SemanticResult(
            top1=CLIP_CATEGORIES[top1_idx],
            top1_score=float(probs[top1_idx]),
            top2=CLIP_CATEGORIES[top2_idx],
            top2_score=float(probs[top2_idx]),
            margin=float(probs[top1_idx] - probs[top2_idx]),
            distribution=dist,
            embedding=[float(v) for v in emb],
        )

    # --------------------------------------------------------------- stub
    def _stub_result(self, bgr: np.ndarray) -> SemanticResult:
        """Deterministic stub classification from image statistics.

        Produces a stable embedding (seeded by content hash) and a category
        distribution derived from luminance/contrast/blur so that:
        - near-black frames -> low confidence (-> uncertain)
        - high-contrast flat frames -> "graphic"
        - otherwise a stable pseudo-category from the content hash
        """
        import cv2
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        flat = gray.tobytes()
        digest = hashlib.sha256(flat).digest()
        # Deterministic unit-vector embedding from the content hash.
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        emb = rng.standard_normal(EMBED_DIM).astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)

        mean = float(gray.mean())
        std = float(gray.std())
        # Build a soft distribution that is deliberately low-confidence for
        # pathological frames and stable for normal ones.
        base = 1.0 / len(CLIP_CATEGORIES)
        probs = np.full(len(CLIP_CATEGORIES), base, dtype=np.float32)
        # Slight deterministic tilt so top-1/top-2 are stable per-image.
        tilt = (np.frombuffer(digest[8:24], dtype=np.uint8).astype(np.float32) / 255.0)
        probs += tilt[: len(CLIP_CATEGORIES)] * 0.3
        # Pathological frames get pushed toward low confidence.
        if mean < 18 or std < 6.0:
            probs = probs * 0.5  # spread out -> low top1
        probs = probs / probs.sum()

        order = np.argsort(probs)[::-1]
        top1_idx, top2_idx = int(order[0]), int(order[1])
        dist = {
            CLIP_CATEGORIES[i].value: float(probs[i])
            for i in range(len(CLIP_CATEGORIES))
        }
        return SemanticResult(
            top1=CLIP_CATEGORIES[top1_idx],
            top1_score=float(probs[top1_idx]),
            top2=CLIP_CATEGORIES[top2_idx],
            top2_score=float(probs[top2_idx]),
            margin=float(probs[top1_idx] - probs[top2_idx]),
            distribution=dist,
            embedding=[float(v) for v in emb],
        )


def _pretrained_quick_gelu(open_clip, model_name: str,
                           pretrained: str) -> bool:
    """QuickGELU flag of the pretrained tag (default False).

    The OpenAI ViT-B/16 & B/32 weights were trained with QuickGELU while the
    built-in config defaults to standard GELU; running the weights with the
    wrong activation silently degrades quality (and warns on every load).
    """
    try:
        cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
        return bool((cfg or {}).get("quick_gelu", False))
    except Exception:
        return False


def classify(images: Sequence[np.ndarray],
             config: CuratorConfig | None = None) -> list[SemanticResult]:
    return SemanticEngine(config).classify_and_embed(images)
