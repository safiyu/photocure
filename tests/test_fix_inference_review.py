"""Regression tests for review findings in inference, quality and bench."""
from __future__ import annotations

import numpy as np
from PIL import Image

from photocure import bench
from photocure.config import CuratorConfig
from photocure.inference import semantic_engine as se
from photocure.quality.image_loader import _to_uint8


def test_to_uint8_scales_12_bit_data_by_its_range():
    a = np.linspace(800, 3600, 100).astype(np.uint16)
    out = _to_uint8(a)
    assert out.max() >= 200 and 40 < out.mean() < 200


def test_to_uint8_full_16_bit_still_divides_by_257():
    a = np.array([0, 65535], dtype=np.uint16)
    assert _to_uint8(a).tolist() == [0, 255]


def test_stub_rows_are_never_rescored_or_learned(tmp_path):
    from photocure.db.models import ImageRecord, SemanticScore
    from photocure.db.repository import quantize_embedding
    from photocure.db.session import get_session, init_db
    db = tmp_path / "t.db"
    init_db(db)
    v = np.random.default_rng(1).standard_normal(512).astype(np.float32)
    with get_session(db) as s:
        for name in ("a", "b", "c"):
            img = ImageRecord(path=f"/p/{name}.jpg", filename=f"{name}.jpg",
                              infer_status="inferred", manual_bucket="food" if name != "c" else None)
            s.add(img)
            s.flush()
            s.add(SemanticScore(image_id=img.id, backend="stub", top1_category="uncertain",
                                embedding=quantize_embedding(v.tolist())))
        cfg = CuratorConfig()
        assert se.rescore_from_embeddings(s, cfg) == 0
        out = se.learn_from_corrections(s, cfg)
        assert out["updated"] == 0 and out["classes_used"] == 0


def test_bench_passes_exif_orientation(tmp_path, monkeypatch):
    p = tmp_path / "a.jpg"
    im = Image.new("RGB", (40, 20), "white")
    exif = Image.Exif()
    exif[274] = 6
    im.save(p, exif=exif)
    assert bench._orientation(p) == 6
    seen = []
    import photocure.quality.image_loader as il
    real = il.load_image

    def spy(path, max_side=None, orientation=None):
        seen.append(orientation)
        return real(path, max_side=max_side, orientation=orientation)

    monkeypatch.setattr(il, "load_image", spy)
    bench.score_labelled_photos([(p, "open")], CuratorConfig())
    assert seen == [6]
