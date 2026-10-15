"""Export the CLIP image tower (and class text features) to ONNX.

Needs ``torch`` and ``open_clip`` (``pip install -e ".[gpu]"``). The export
writes:

- ``clip_image.onnx``: image tower, dynamic batch, input ``image`` (N,3,S,S),
  output ``embedding`` (N,512, not normalized).
- ``text_features.npy`` and ``classes.json``: the ensembled, normalized class
  text embeddings, so inference needs no text tower.
- Copies of any InsightFace ``buffalo_l`` ONNX files found under
  ``~/.insightface/models`` (they are already ONNX).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import CuratorConfig

_INSTALL_HINT = ("ONNX export needs torch and open_clip. "
                 'Install them with: pip install -e ".[gpu]"')


def export_models(out_dir: str | Path, cfg: CuratorConfig | None = None,
                  opset: int = 17) -> dict[str, Path]:
    """Write the ONNX artifacts into ``out_dir`` and return their paths."""
    try:
        import open_clip
        import torch
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e

    import numpy as np

    from .semantic_engine import SemanticEngine, _pretrained_quick_gelu

    cfg = cfg or CuratorConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    name = cfg.taxonomy.clip_model_name
    pretrained = cfg.taxonomy.clip_pretrained
    model, _train, _val = open_clip.create_model_and_transforms(
        name, pretrained=pretrained,
        force_quick_gelu=_pretrained_quick_gelu(open_clip, name, pretrained))
    model = model.eval().cpu()

    size = model.visual.image_size
    side = size[0] if isinstance(size, (tuple, list)) else int(size)
    dummy = torch.zeros(1, 3, side, side)
    image_path = out / "clip_image.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model.visual, dummy, str(image_path),
            input_names=["image"], output_names=["embedding"],
            dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=opset)
    artifacts["clip_image"] = image_path

    engine = SemanticEngine(cfg)
    try:
        engine._model = model
        engine._tokenize = open_clip.get_tokenizer(name)
        engine._device = "cpu"
        engine._autocast = None
        engine._encode_texts()
        feats = engine._text_features.detach().cpu().numpy().astype(np.float32)
    finally:
        engine._model = None
    feats_path = out / "text_features.npy"
    np.save(feats_path, feats)
    classes_path = out / "classes.json"
    classes_path.write_text(json.dumps(engine.taxonomy.names, indent=2),
                            encoding="utf-8")
    artifacts["text_features"] = feats_path
    artifacts["classes"] = classes_path

    face_src = Path.home() / ".insightface" / "models" / "buffalo_l"
    if face_src.is_dir():
        for onnx in sorted(face_src.glob("*.onnx")):
            dst = out / onnx.name
            shutil.copy2(onnx, dst)
            artifacts[f"insightface_{onnx.stem}"] = dst
    return artifacts
