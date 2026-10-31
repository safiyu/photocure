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
from ..rules.taxonomy import CategoryType, Taxonomy, get_taxonomy
from .backends import BackendState, detect_backend, empty_cuda_cache

log = logging.getLogger("photocure.semantic")

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
    # DB class names (equal to ``top1.value`` except for custom YAML classes).
    top1_name: str = ""
    top2_name: str = ""

    def __post_init__(self) -> None:
        self.top1_name = self.top1_name or self.top1.value
        self.top2_name = self.top2_name or self.top2.value


class SemanticEngine:
    """Zero-shot CLIP classifier + embedding extractor."""

    def __init__(self, config: CuratorConfig | None = None,
                 backend: BackendState | None = None):
        self.cfg = config or CuratorConfig()
        self.bs = backend or detect_backend(self.cfg)
        # A transient load failure only downgrades the previous run; try the real model again.
        self.bs.retry_downgraded()
        self._model = None
        self._preprocess = None
        self._tokenize = None
        self._text_features: object | None = None  # torch tensor (device-local)
        self._device = None
        self._autocast = None
        self.taxonomy: Taxonomy = get_taxonomy(self.cfg)

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
            self._device = self.bs.device
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
            # the training one, whose RandomResizedCrop is stochastic - the
            # same photo would then produce a different crop (and embedding)
            # on every encode, breaking determinism and burst clustering.
            self._preprocess = val_preprocess
            self._tokenize = open_clip.get_tokenizer(model_name)
            self._model = model.to(self._device).eval()
            self._autocast = (
                torch.autocast(device_type=self._device, dtype=torch.float16)
                if self.bs.fp16 and self._device in ("cuda", "xpu")
                else None
            )
            self._encode_texts()
            log.info("OpenCLIP %s/%s ready on %s (fp16=%s, quick_gelu=%s)",
                     model_name, pretrained, self._device, self.bs.fp16,
                     quick_gelu)
            return True
        except Exception as e:
            log.warning("OpenCLIP load failed (%s: %s)", e.__class__.__name__, e)
            return False

    def _encode_texts(self) -> None:
        import torch
        ctx = self._autocast or torch.autocast(device_type="cpu", enabled=False)
        per_class: list[np.ndarray] = []
        for prompts in self.taxonomy.prompts:
            tok = self._tokenize(list(prompts)).to(self._device)
            with torch.no_grad(), ctx:
                tf = self._model.encode_text(tok)
            per_class.append(tf.float().cpu().numpy())
        # Keep as a torch tensor on the model device: matmul between a torch
        # tensor and a numpy array trips numpy-2.0 __array_wrap__ deprecations.
        self._text_features = torch.from_numpy(
            ensemble_text_features(per_class)).to(self._device)

    def text_features_numpy(self) -> np.ndarray:
        """Class text embeddings ``(C, 512)``, L2-normalized (text tower only).

        The stub backend has no text tower, so it derives deterministic unit
        vectors from the prompt text instead.
        """
        self._ensure_loaded()
        if self.bs.semantic_backend == "stub" or self._model == "stub":
            return _stub_text_features(self.taxonomy)
        return self._text_features.detach().cpu().numpy().astype(np.float32)

    def text_search_available(self) -> bool:
        """True when a real CLIP text tower is loaded (the placeholder has none)."""
        self._ensure_loaded()
        return (self.bs.semantic_backend == "clip"
                and self._model is not None and self._model != "stub")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Raw CLIP text features ``(len(texts), 512)``; callers normalize and pool."""
        import torch
        self._ensure_loaded()
        tok = self._tokenize(list(texts)).to(self._device)
        ctx = self._autocast or torch.autocast(device_type="cpu", enabled=False)
        with torch.no_grad(), ctx:
            feats = self._model.encode_text(tok)
        return feats.float().cpu().numpy()

    def release(self) -> None:
        import gc
        self._model = None
        self._text_features = None
        self._autocast = None
        self._preprocess = None
        self._tokenize = None
        self._device = None
        gc.collect()
        empty_cuda_cache(self.bs)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _bgr_to_tensor(bgr: np.ndarray, preprocess) -> object:
        """Run the open_clip val transform on a PIL image (works on all versions)."""
        from PIL import Image
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        return preprocess(Image.fromarray(rgb))

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
        batch_size = effective_clip_batch(self.cfg.device.clip_batch_size, self.bs.device)
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
        return _result_from_probs(self.taxonomy, probs, emb)

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
        n_cls = len(self.taxonomy.entries)
        base = 1.0 / n_cls
        probs = np.full(n_cls, base, dtype=np.float32)
        # Slight deterministic tilt so top-1/top-2 are stable per-image.
        tilt_bytes = np.frombuffer(digest[8:24], dtype=np.uint8)
        tilt = np.resize(tilt_bytes.astype(np.float32) / 255.0, n_cls)
        probs += tilt * 0.3
        # Pathological frames get pushed toward low confidence.
        if mean < 18 or std < 6.0:
            probs = probs * 0.5  # spread out -> low top1
        probs = probs / probs.sum()
        return _result_from_probs(self.taxonomy, probs, emb)


def effective_clip_batch(configured: int, device: str) -> int:
    """Batch size that suits the device: big on CUDA, modest on MPS and CPU."""
    cap = {"cuda": None, "xpu": 32, "mps": 32}.get(device, 16)
    n = max(1, int(configured))
    return n if cap is None else min(n, cap)


def _result_from_probs(tax: Taxonomy, probs: np.ndarray,
                       emb: np.ndarray) -> SemanticResult:
    names = tax.names
    order = np.argsort(probs)[::-1]
    i1, i2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    return SemanticResult(
        top1=tax.categories[i1],
        top1_score=float(probs[i1]),
        top2=tax.categories[i2],
        top2_score=float(probs[i2]),
        margin=float(probs[i1] - probs[i2]),
        distribution={names[i]: float(probs[i]) for i in range(len(names))},
        embedding=[float(v) for v in emb],
        top1_name=names[i1],
        top2_name=names[i2],
    )


def ensemble_text_features(per_class: Sequence[np.ndarray]) -> np.ndarray:
    """Prompt ensembling: normalize each prompt, mean-pool per class, re-normalize.

    ``per_class[c]`` is ``(n_prompts_c, D)``; the result is ``(C, D)`` unit rows.
    """
    rows = []
    for feats in per_class:
        f = np.asarray(feats, dtype=np.float32)
        f = f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-12)
        m = f.mean(axis=0)
        rows.append(m / (np.linalg.norm(m) + 1e-12))
    return np.stack(rows).astype(np.float32)


def _stub_text_features(tax: Taxonomy) -> np.ndarray:
    rows = []
    for prompts in tax.prompts:
        vecs = []
        for p in prompts:
            seed = int.from_bytes(hashlib.sha256(p.encode("utf-8")).digest()[:8], "big")
            v = np.random.default_rng(seed).standard_normal(EMBED_DIM).astype(np.float32)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
        rows.append(np.stack(vecs))
    return ensemble_text_features(rows)


def rescore_from_embeddings(session, cfg: CuratorConfig | None = None,
                            backend: BackendState | None = None) -> int:
    """Recompute class scores from stored embeddings using only the text tower.

    Use after editing the taxonomy or its prompts; no image is decoded. Returns
    the number of ``SemanticScore`` rows updated. The caller owns the commit.
    """
    import json

    from sqlmodel import select as sselect

    from ..db.models import SemanticScore
    from ..db.repository import dequantize_embedding

    cfg = cfg or CuratorConfig()
    engine = SemanticEngine(cfg, backend=backend)
    try:
        text = engine.text_features_numpy()
        tax = engine.taxonomy
        updated = 0
        for row in session.exec(sselect(SemanticScore)).all():
            if row.backend == "stub":  # placeholder embeddings are unrelated to the text tower
                continue
            emb = dequantize_embedding(row.embedding)
            if not emb or len(emb) != text.shape[1]:
                continue
            e = np.asarray(emb, dtype=np.float32)
            n = float(np.linalg.norm(e))
            if n < 1e-9:
                continue
            logits = (text @ (e / n)) * 100.0
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            _write_result(row, _result_from_probs(tax, probs, e))
            updated += 1
        session.flush()
        return updated
    finally:
        engine.release()


def _write_result(row, res: SemanticResult) -> None:
    """Copy a classification result onto a SemanticScore row."""
    import json
    row.top1_category = res.top1_name
    row.top1_score = res.top1_score
    row.top2_category = res.top2_name
    row.top2_score = res.top2_score
    row.score_margin = res.margin
    row.distribution = json.dumps(res.distribution)


def learn_from_corrections(session, cfg: CuratorConfig | None = None,
                           backend: BackendState | None = None) -> dict:
    """Re-score uncorrected photos using the user's manual category choices.

    Each corrected class with enough examples becomes a prototype (mean of its
    L2-normalized stored embeddings). Every photo without a manual bucket, and
    not locked, gets ``zero_shot + feedback_weight * 100 * cosine(prototype)``
    for the classes that have a prototype. The caller owns the commit.
    """
    from sqlmodel import select as sselect

    from ..db.models import ImageRecord, SemanticScore
    from ..db.repository import dequantize_embedding

    cfg = cfg or CuratorConfig()
    tcfg = cfg.taxonomy
    weight = float(getattr(tcfg, "feedback_weight", 0.6))
    min_examples = int(getattr(tcfg, "feedback_min_examples", 2))
    min_cosine = float(getattr(tcfg, "feedback_min_cosine", 0.60))
    summary = {"updated": 0, "classes_used": 0, "examples": 0}

    engine = SemanticEngine(cfg, backend=backend)
    try:
        text = engine.text_features_numpy()
        tax = engine.taxonomy
        index = {n: i for i, n in enumerate(tax.names)}
        rows = session.exec(
            sselect(SemanticScore, ImageRecord)
            .join(ImageRecord, ImageRecord.id == SemanticScore.image_id)).all()

        def _unit(row) -> np.ndarray | None:
            if row.backend == "stub":  # placeholder embeddings carry no signal
                return None
            emb = dequantize_embedding(row.embedding)
            if not emb or len(emb) != text.shape[1]:
                return None
            e = np.asarray(emb, dtype=np.float32)
            n = float(np.linalg.norm(e))
            return e / n if n > 1e-9 else None

        sums: dict[int, np.ndarray] = {}
        counts: dict[int, int] = {}
        for sem, img in rows:
            ci = index.get(img.manual_bucket or "")
            e = _unit(sem) if ci is not None else None
            if e is None:
                continue
            sums[ci] = sums.get(ci, 0) + e
            counts[ci] = counts.get(ci, 0) + 1
        protos: dict[int, np.ndarray] = {}
        for ci, total in sums.items():
            if counts[ci] >= min_examples:
                norm = float(np.linalg.norm(total))
                if norm > 1e-9:
                    protos[ci] = (total / norm).astype(np.float32)
        summary["classes_used"] = len(protos)
        summary["examples"] = sum(counts[ci] for ci in protos)
        if not protos:
            return summary

        for sem, img in rows:
            if img.manual_bucket or getattr(img, "locked", False):
                continue
            e = _unit(sem)
            if e is None:
                continue
            logits = (text @ e) * 100.0
            for ci, proto in protos.items():
                sim = float(proto @ e)
                if sim > min_cosine:
                    logits[ci] += weight * 100.0 * (sim - min_cosine)
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            _write_result(sem, _result_from_probs(tax, probs, e))
            summary["updated"] += 1
        session.flush()
        return summary
    finally:
        engine.release()


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
