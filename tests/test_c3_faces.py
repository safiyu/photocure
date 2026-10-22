"""C3 acceptance: YuNet + MediaPipe face pipeline, burst-relative blink flags, safe crop endpoint."""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from photocure import models
from photocure.clustering.best_shot_selector import BestShotSelector, ShotCandidate
from photocure.clustering.blinks import apply_blinks, classify_blinks, iou, track_faces
from photocure.config import CuratorConfig
from photocure.db import models as M
from photocure.db.session import get_session, init_db
from photocure.inference.backends import BackendState, reset_backend_cache
from photocure.inference.face_engine import FaceEngine, FaceInfo
from photocure.inference.pipeline import InferencePipeline

from .test_inference import _scan
from .world import build_world

PORTRAIT = Path(__file__).parent / "fixtures" / "face_portrait.jpg"
ROOT = Path(__file__).resolve().parent.parent


def _bs(face="yunet"):
    return BackendState(device="cpu", fp16=False, torch=None, cuda_available=False,
                        memory_fraction=0.75, face_backend=face, semantic_backend="stub")


@pytest.fixture(autouse=True)
def _fresh_backend():
    reset_backend_cache()
    yield
    reset_backend_cache()


# ------------------------------------------------------------ classify_blinks
def test_one_high_score_among_stable_frames_is_exactly_one_closed():
    out = classify_blinks([[(0, .10), (1, .12), (2, .90), (3, .11), (4, .10)]])
    assert [out[(0, i)] for i in range(5)] == ["open", "open", "closed", "open", "open"]


def test_two_frames_fall_back_to_the_absolute_rules():
    out = classify_blinks([[(0, .10), (1, .80)]])
    assert out == {(0, 0): "open", (0, 1): "closed"}
    assert classify_blinks([[(0, .10), (1, .60)]])[(0, 1)] == "possible"  # no baseline, below closed_hi


def test_narrow_eyed_subject_with_constant_moderate_score_is_never_closed():
    out = classify_blinks([[(i, .45) for i in range(6)]])
    assert set(out.values()) == {"open"}


def test_relative_rule_catches_a_blink_below_the_absolute_threshold():
    out = classify_blinks([[(0, .12), (1, .14), (2, .13), (3, .58)]])
    assert out[(0, 3)] == "closed" and out[(0, 0)] == "open"


def test_missing_score_is_unknown_and_never_open_or_closed():
    out = classify_blinks([[(0, None), (1, .1), (2, .1), (3, .1)]])
    assert out[(0, 0)] == "unknown"


# ------------------------------------------------------------------- tracker
B = (0.40, 0.30, 0.55, 0.50)


def _shift(b, dx):
    return (b[0] + dx, b[1], b[2] + dx, b[3])


def test_consecutive_frames_with_overlap_form_one_track():
    assert track_faces([[B], [_shift(B, .01)], [_shift(B, .02)]]) == [[(0, 0), (1, 0), (2, 0)]]


def test_a_pan_or_jump_breaks_the_track():
    tracks = track_faces([[B], [_shift(B, .01)], [_shift(B, .40)]])
    assert tracks == [[(0, 0), (1, 0)], [(2, 0)]]
    assert iou(B, _shift(B, .40)) < 0.45


def test_two_faces_keep_their_own_tracks():
    a, b = (0.1, 0.1, 0.25, 0.3), (0.6, 0.1, 0.75, 0.3)
    tracks = track_faces([[a, b], [_shift(b, .01), _shift(a, .01)]])
    assert sorted(tracks) == [[(0, 0), (1, 1)], [(0, 1), (1, 0)]]


def test_a_track_cannot_skip_a_frame():
    assert track_faces([[B], [], [B]]) == [[(0, 0)], [(2, 0)]]


def _face(box, score, status="unknown"):
    return SimpleNamespace(x1=box[0], y1=box[1], x2=box[2], y2=box[3], eye_closed_score=score,
                           eye_status=status, is_blink=False)


def _rec(i, faces, sec):
    return SimpleNamespace(id=i, faces=faces, blink_count=0, has_blinks=False,
                           taken_at=datetime(2024, 5, 1, 12, 0, sec, tzinfo=timezone.utc))


def test_tracks_never_cross_a_burst_boundary():
    # A lone 0.6 is "possible". Merged with the other burst's frames it would become "closed".
    a = [_rec(1, [_face(B, .10)], 0), _rec(2, [_face(B, .10)], 1)]
    b = [_rec(3, [_face(B, .10)], 2), _rec(4, [_face(B, .60)], 3)]
    apply_blinks(a + b, [a, b], CuratorConfig().face)
    assert b[1].faces[0].eye_status == "possible" and not b[1].has_blinks


def test_blink_is_judged_against_the_subjects_own_burst_baseline():
    frames = [_rec(i, [_face(B, s)], i) for i, s in enumerate((.12, .13, .14, .58, .13))]
    flagged = apply_blinks(frames, [frames], CuratorConfig().face)
    assert flagged == 1 and [r.has_blinks for r in frames] == [False, False, False, True, False]
    assert frames[3].blink_count == 1 and frames[3].faces[0].is_blink


def test_photos_outside_bursts_use_single_frame_rules():
    lone = _rec(9, [_face(B, .85)], 0)
    apply_blinks([lone], [], CuratorConfig().face)
    assert lone.has_blinks and lone.faces[0].eye_status == "closed"


def test_legacy_faces_without_a_box_are_unknown_and_ignored():
    legacy = SimpleNamespace(x1=None, y1=None, x2=None, y2=None, eye_closed_score=None,
                             eye_status="open", is_blink=True)
    r = _rec(1, [legacy], 0)
    apply_blinks([r], [], CuratorConfig().face)
    assert legacy.eye_status == "unknown" and not legacy.is_blink and not r.has_blinks


# -------------------------------------------------------------- best shot
def test_blink_penalty_is_per_face_and_clamped():
    sel = BestShotSelector(CuratorConfig())
    base = dict(image_id=1, sharpness_norm=.8, face_sharpness_norm=.8, width=4000, height=3000,
                has_face=True, eye_open=1.0)
    clean = sel.score(ShotCandidate(**base))[0]
    assert sel.score(ShotCandidate(**base, blink_count=2))[0] == pytest.approx(clean - .30)
    assert sel.score(ShotCandidate(**base, blink_count=9))[0] == pytest.approx(clean - .45)


def test_best_shot_still_chosen_when_every_frame_blinks():
    sel = BestShotSelector(CuratorConfig())
    cands = [ShotCandidate(image_id=i, sharpness_norm=s, width=4000, height=3000, blink_count=1)
             for i, s in ((1, .3), (2, .9), (3, .5))]
    assert sel.pick_best_shot(cands).image_id == 2


def test_a_sharp_frame_with_closed_eyes_loses_to_a_slightly_softer_open_one():
    sel = BestShotSelector(CuratorConfig())
    blink = ShotCandidate(image_id=1, sharpness_norm=.85, face_sharpness_norm=.85, has_face=True,
                          eye_open=.1, blink_count=1, width=4000, height=3000)
    open_ = ShotCandidate(image_id=2, sharpness_norm=.80, face_sharpness_norm=.80, has_face=True,
                          eye_open=.95, width=4000, height=3000)
    assert sel.pick_best_shot([blink, open_]).image_id == 2


# ---------------------------------------------------------- engine: stage 1+2
def _cfg():
    c = CuratorConfig()
    c.device.device = "cpu"
    return c


def test_yunet_finds_the_face_in_the_portrait_with_normalised_boxes():
    img = cv2.imread(str(PORTRAIT))
    faces = FaceEngine(_cfg(), backend=_bs()).detect_faces_batch([img])[0]
    assert len(faces) == 1
    f = faces[0]
    assert 0.0 <= f.x1 < f.x2 <= 1.0 and 0.0 <= f.y1 < f.y2 <= 1.0
    assert f.confidence > 0.7 and len(f.landmarks) == 5
    assert 0.25 < (f.x1 + f.x2) / 2 < 0.6  # the face is in the upper middle of the frame


def test_detection_is_scale_free():
    img = cv2.imread(str(PORTRAIT))
    big = cv2.resize(img, None, fx=3, fy=3)
    e = FaceEngine(_cfg(), backend=_bs())
    a, b = e.detect_faces_batch([img])[0][0], e.detect_faces_batch([big])[0][0]
    assert abs(a.x1 - b.x1) < 0.03 and abs(a.y2 - b.y2) < 0.03


def test_no_faces_flag_disables_analysis_entirely():
    cfg = _cfg()
    cfg.face.enabled = False
    assert FaceEngine(cfg, backend=_bs()).detect_faces_batch([cv2.imread(str(PORTRAIT))]) == [[]]


def test_compute_embeddings_fills_a_vector_for_a_real_face_and_skips_tiny_ones():
    engine = FaceEngine(_cfg(), backend=_bs())
    bgr = cv2.imread(str(PORTRAIT))
    faces = engine.detect_faces_batch([bgr])[0]
    engine.analyze_faces(bgr, faces)
    engine.compute_embeddings(bgr, faces)
    assert faces and faces[0].embedding and len(faces[0].embedding) > 0

    tiny = FaceInfo(0.0, 0.0, 0.01, 0.01, confidence=0.9, face_sharpness=None)
    engine.compute_embeddings(bgr, [tiny])
    assert tiny.embedding is None


class _Cat:
    def __init__(self, name, score):
        self.category_name, self.score = name, score


class _FakeLandmarker:
    def __init__(self, sets, centres=None):
        self.sets, self.seen = sets, []
        self.centres = centres or [(.5, .5)] * len(sets)

    def detect(self, image):
        self.seen.append(image.data.shape)
        return SimpleNamespace(
            face_landmarks=[[SimpleNamespace(x=x, y=y)] * 3 for x, y in self.centres],
            face_blendshapes=[[_Cat("eyeBlinkLeft", l), _Cat("eyeBlinkRight", r)] for l, r in self.sets])


def _engine(fake):
    e = FaceEngine(_cfg(), backend=_bs())
    e._landmarker = fake
    e._mp = SimpleNamespace(Image=lambda image_format, data: SimpleNamespace(data=data),
                            ImageFormat=SimpleNamespace(SRGB=1))
    return e


def test_eye_closed_score_is_the_mean_of_both_blink_blendshapes():
    e = _engine(_FakeLandmarker([(.8, .6)]))
    face = FaceInfo(.2, .2, .6, .6, .9)
    e.analyze_faces(np.full((1000, 1000, 3), 120, np.uint8), [face])
    assert face.eye_closed_score == pytest.approx(.7) and face.face_sharpness is not None


def test_crops_are_downscaled_to_512_before_the_landmarker():
    fake = _FakeLandmarker([(.1, .1)])
    e = _engine(fake)
    e.analyze_faces(np.full((3000, 4000, 3), 90, np.uint8), [FaceInfo(.1, .1, .8, .9, .9)])
    assert fake.seen and max(fake.seen[0][:2]) <= 512


def test_faces_narrower_than_min_px_are_unknown_and_skip_the_landmarker():
    fake = _FakeLandmarker([(.9, .9)])
    e = _engine(fake)
    face = FaceInfo(.5, .5, .52, .56, .9)  # 20 px wide on a 1000 px frame
    e.analyze_faces(np.full((1000, 1000, 3), 90, np.uint8), [face])
    assert face.eye_closed_score is None and face.eye_status == "unknown"
    assert face.face_sharpness is None and fake.seen == []


def test_the_centred_face_is_used_when_a_neighbour_is_in_the_crop():
    fake = _FakeLandmarker([(.9, .9), (.1, .1)], centres=[(.1, .1), (.5, .5)])
    e = _engine(fake)
    face = FaceInfo(.2, .2, .6, .6, .9)
    e.analyze_faces(np.full((1000, 1000, 3), 120, np.uint8), [face])
    assert face.eye_closed_score == pytest.approx(.1)


def test_a_failing_landmarker_leaves_eye_state_unknown_not_open():
    class Boom:
        def detect(self, image):
            raise RuntimeError("tflite exploded")
    e = _engine(Boom())
    face = FaceInfo(.2, .2, .6, .6, .9)
    e.analyze_faces(np.full((1000, 1000, 3), 120, np.uint8), [face])
    assert face.eye_closed_score is None and face.eye_status == "unknown"


def test_real_landmarker_when_mediapipe_is_installed():
    pytest.importorskip("mediapipe")
    e = FaceEngine(_cfg(), backend=_bs())
    img = cv2.imread(str(PORTRAIT))
    faces = e.detect_faces_batch([img])[0]
    e.analyze_faces(img, faces)
    if e._landmarker_failed and any(k in getattr(e, "_landmarker_error", "") for k in ("libEGL", "libGLES", "libGL")):
        pytest.skip(f"System lacks OpenGL/EGL/GLES libraries: {e._landmarker_error}")
    assert not e._landmarker_failed, f"Landmarker initialization failed: {getattr(e, '_landmarker_error', 'unknown error')}"
    assert faces[0].eye_closed_score is not None and faces[0].eye_closed_score < 0.4  # eyes open
    e.release()


# ------------------------------------------------------ bundled model integrity
def test_manifest_lists_the_bundled_models_with_matching_hashes():
    bundled = {models.YUNET, models.LANDMARKER, models.SFACE, models.AESTHETIC}
    assert all(models.verify_model(f) for f in bundled)
    assert {m["file"] for m in models.load_manifest()} == bundled
    assert all(len(m["sha256"]) == 64 and m["upstream_url"].startswith("https://")
               and m["license"] for m in models.load_manifest())


def test_an_altered_model_is_refused(tmp_path):
    d = tmp_path / "models"
    shutil.copytree(models.MODELS_DIR, d, ignore=shutil.ignore_patterns("__pycache__"))
    assert models.verify_model(models.YUNET, d)
    with (d / models.YUNET).open("r+b") as f:
        f.seek(100)
        f.write(b"\x00\x01\x02")
    assert not models.verify_model(models.YUNET, d)
    assert models.model_path(models.YUNET, d) is None
    assert models.YUNET not in {m["file"] for m in models.active_models(d)}


def test_engine_degrades_to_no_faces_when_the_detector_model_is_not_trusted(monkeypatch):
    monkeypatch.setattr(models, "verify_model", lambda *a, **k: False)
    bs = _bs()
    out = FaceEngine(_cfg(), backend=bs).detect_faces_batch([cv2.imread(str(PORTRAIT))])
    assert out == [[]] and bs.face_backend == "stub"


# --------------------------------------------------------- pipeline integration
def test_provisional_blink_flags_are_set_by_infer_alone(tmp_db, tmp_path, monkeypatch):
    cfg, _ = _scan(tmp_path, tmp_db, 2)
    monkeypatch.setattr(FaceEngine, "detect_faces_batch",
                        lambda self, imgs: [[FaceInfo(.2, .2, .5, .6, .9)] for _ in imgs])

    def analyze(self, bgr, faces):
        for f in faces:
            f.eye_closed_score = .9
    monkeypatch.setattr(FaceEngine, "analyze_faces", analyze)
    InferencePipeline(cfg, backend=_bs()).run(db_path=tmp_db, show_progress=False)
    with get_session(tmp_db) as s:
        recs = s.exec(select(M.ImageRecord)).all()
        assert all(r.has_blinks and r.blink_count == 1 for r in recs)
        faces = s.exec(select(M.FaceDetection)).all()
        assert all(f.is_blink and f.eye_status == "closed" and f.eye_closed_score == .9 for f in faces)


def test_old_face_rows_are_redone_without_repeating_clip(tmp_db, tmp_path, monkeypatch):
    cfg, _ = _scan(tmp_path, tmp_db, 2)
    pipe = InferencePipeline(cfg)  # stub backends: first pass just completes inference
    pipe.run(db_path=tmp_db, show_progress=False)
    with get_session(tmp_db) as s:
        for r in s.exec(select(M.ImageRecord)).all():
            assert r.infer_status == "inferred" and r.faces_schema_version == 0
    monkeypatch.setattr(FaceEngine, "detect_faces_batch",
                        lambda self, imgs: [[FaceInfo(.2, .2, .5, .6, .9)] for _ in imgs])
    monkeypatch.setattr(FaceEngine, "analyze_faces", lambda self, bgr, faces: None)
    from photocure.inference.semantic_engine import SemanticEngine
    monkeypatch.setattr(SemanticEngine, "classify_and_embed",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("CLIP must not rerun")))
    res = InferencePipeline(cfg, backend=_bs()).run(db_path=tmp_db, show_progress=False)
    assert res.faces_redone == 2 and res.failures == 0
    with get_session(tmp_db) as s:
        for r in s.exec(select(M.ImageRecord)).all():
            assert r.infer_status == "inferred" and r.faces_schema_version == cfg.face.schema_version
        assert len(s.exec(select(M.FaceDetection)).all()) == 2


def test_cluster_flags_blinks_per_burst_before_choosing_the_best_shot(tmp_path, monkeypatch):
    from photocure.clustering.burst_detector import BurstGroupResult
    from photocure.services import Planner
    w = build_world(tmp_path, n=4)
    w.cfg.burst.min_group_size = 2
    with get_session(w.db) as s:
        for i, (rid, score) in enumerate(zip(w.ids, (.1, .1, .6, .1))):
            rec = s.get(M.ImageRecord, rid)
            rec.taken_at = datetime(2024, 5, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=i)
            rec.width, rec.height = 4000, 3000
            s.add(M.FaceDetection(image_id=rid, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1,
                                  confidence=.95, x1=.4, y1=.3, x2=.55, y2=.5,
                                  eye_closed_score=score, face_sharpness=300))
            s.add(M.QualityMetric(image_id=rid, sharpness_norm=.8))
    monkeypatch.setattr("photocure.services.BurstDetector.cluster_records",
                        lambda self, recs: [BurstGroupResult(image_ids=list(w.ids))])
    assert Planner(w.cfg, db_path=w.db).cluster() == 1
    with get_session(w.db) as s:
        recs = {r.id: r for r in s.exec(select(M.ImageRecord)).all()}
        # .6 against a ~.1 baseline in a 4-frame burst is a blink; as a lone frame it would not be.
        assert [recs[i].has_blinks for i in w.ids] == [False, False, True, False]
        assert recs[w.ids[2]].is_alternate and not recs[w.ids[2]].is_best_shot


# ----------------------------------------------------------------------- API
@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_CACHE_DIR", str(tmp_path / "cache"))
    from photocure.scanner.file_scanner import FileScanner
    from photocure.ui.app import create_app
    db = tmp_path / "api.db"
    cfg = _cfg()
    cfg.db_path = db
    init_db(db, drop=True)
    lib = tmp_path / "lib"
    lib.mkdir()
    shutil.copy(PORTRAIT, lib / "IMG_20240501_120000.jpg")
    FileScanner(cfg).scan_directory(lib, db_path=db, show_progress=False)
    face = FaceEngine(cfg, backend=_bs()).detect_faces_batch([cv2.imread(str(PORTRAIT))])[0][0]
    with get_session(db) as s:
        rec = s.exec(select(M.ImageRecord)).first()
        rec.infer_status = "inferred"
        fd = M.FaceDetection(image_id=rec.id, bbox_x1=0, bbox_y1=0, bbox_x2=1, bbox_y2=1,
                             confidence=face.confidence, x1=face.x1, y1=face.y1, x2=face.x2, y2=face.y2,
                             eye_closed_score=.05, eye_status="open")
        legacy = M.FaceDetection(image_id=rec.id, bbox_x1=1, bbox_y1=1, bbox_x2=2, bbox_y2=2, confidence=.9)
        s.add(fd)
        s.add(legacy)
        s.flush()
        ids = (rec.id, fd.id, legacy.id)
    app = create_app(str(db))
    client = TestClient(app, headers={"X-PCUR-Token": app.state.token})
    return SimpleNamespace(client=client, app=app, anon=TestClient(app), image=ids[0], face=ids[1],
                           legacy=ids[2], cache=tmp_path / "cache")


def test_face_listing_gives_boxes_status_and_crop_urls(api):
    items = api.client.get(f"/api/images/{api.image}/faces").json()["items"]
    good = next(i for i in items if i["id"] == api.face)
    assert good["eye_status"] == "open" and good["url"].startswith(f"/api/faces/{api.face}.jpg?v=")
    legacy = next(i for i in items if i["id"] == api.legacy)
    assert legacy["box"] is None and legacy["url"] is None


def test_crop_endpoint_serves_a_cached_immutable_jpeg(api):
    url = api.client.get(f"/api/images/{api.image}/faces").json()["items"][0]["url"]
    r = api.client.get(url)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["cache-control"] == "public, max-age=31536000, immutable"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert 0 < max(img.shape[:2]) <= 512
    cached = list((api.cache / "faces").glob(f"{api.face}_*.jpg"))
    assert len(cached) == 1 and not list((api.cache / "faces").glob("*.tmp"))
    assert api.client.get(url).content == r.content  # served from the cache


@pytest.mark.parametrize("v", ["../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "zzzzzzzzzzzz", "abc",
                               "a" * 65, "0123456789abcdef/../x", "0123456789ABCDEF"])
def test_crop_endpoint_rejects_malformed_cache_keys(api, v):
    assert api.client.get(f"/api/faces/{api.face}.jpg", params={"v": v}).status_code == 400
    assert not (api.cache / "faces").exists() or not list((api.cache / "faces").glob("*"))


def test_cache_key_is_a_cache_buster_only_and_never_part_of_the_path(api):
    a = api.client.get(f"/api/faces/{api.face}.jpg?v=0123456789abcdef")
    b = api.client.get(f"/api/faces/{api.face}.jpg")
    assert a.status_code == b.status_code == 200 and a.content == b.content
    assert len(list((api.cache / "faces").glob("*.jpg"))) == 1  # name comes from the database row


def test_crop_endpoint_needs_auth_and_handles_unknown_and_legacy_faces(api):
    assert api.anon.get(f"/api/faces/{api.face}.jpg").status_code == 401
    assert api.client.get("/api/faces/99999.jpg").status_code == 404
    assert api.client.get(f"/api/faces/{api.legacy}.jpg").status_code == 404
    assert api.client.get(f"/api/faces/{api.face}.png").status_code == 404
    assert api.client.get("/api/faces/abc.jpg").status_code == 404


# ----------------------------------------------------------- reject blinks
@pytest.fixture()
def burst_api(tmp_path, monkeypatch):
    monkeypatch.setenv("PCUR_CACHE_DIR", str(tmp_path / "cache"))
    from photocure.ui.app import create_app
    w = build_world(tmp_path, n=5)
    with get_session(w.db) as s:
        bg = M.BurstGroup(name="Burst_test", image_count=5, best_shot_id=w.ids[0])
        s.add(bg)
        s.flush()
        for i, rid in enumerate(w.ids):
            r = s.get(M.ImageRecord, rid)
            r.burst_id, r.is_best_shot, r.is_alternate = bg.id, i == 0, i != 0
            r.has_blinks = i in (0, 1, 2)   # the best shot and two alternates blink
            r.blink_count = 1 if r.has_blinks else 0
        s.get(M.ImageRecord, w.ids[2]).locked = True
        burst_id = bg.id
    app = create_app(str(w.db))
    client = TestClient(app, headers={"X-PCUR-Token": app.state.token})
    return SimpleNamespace(client=client, ids=w.ids, burst=burst_id, db=w.db)


def _buckets(b):
    with get_session(b.db) as s:
        return {r.id: r.manual_bucket for r in s.exec(select(M.ImageRecord)).all()}


def test_reject_blinks_moves_only_blinking_alternates(burst_api):
    detail = burst_api.client.get(f"/api/burst/{burst_api.burst}").json()
    assert detail["can_reject_blinks"] and detail["reject_count"] == 1
    r = burst_api.client.post(f"/api/bursts/{burst_api.burst}/reject-blinks").json()
    assert r["moved"] == 1 and r["ids"] == [burst_api.ids[1]]
    b = _buckets(burst_api)
    assert b[burst_api.ids[1]] == "junk_blinks"
    assert b[burst_api.ids[0]] is None   # best shot untouched
    assert b[burst_api.ids[2]] is None   # locked photo untouched
    assert b[burst_api.ids[3]] is None and b[burst_api.ids[4]] is None  # no blink: untouched
    assert burst_api.client.get(f"/api/burst/{burst_api.burst}").json()["can_reject_blinks"] is False


def test_reject_blinks_is_reversible_and_touches_no_files(burst_api):
    with get_session(burst_api.db) as s:
        before = sorted(Path(r.path).exists() for r in s.exec(select(M.ImageRecord)).all())
    burst_api.client.post(f"/api/bursts/{burst_api.burst}/reject-blinks")
    back = burst_api.client.post(f"/api/bursts/{burst_api.burst}/restore-blinks").json()
    assert back["restored"] == 1 and _buckets(burst_api)[burst_api.ids[1]] is None
    with get_session(burst_api.db) as s:
        assert sorted(Path(r.path).exists() for r in s.exec(select(M.ImageRecord)).all()) == before


def test_nothing_is_offered_or_moved_when_every_frame_blinks(burst_api):
    with get_session(burst_api.db) as s:
        for r in s.exec(select(M.ImageRecord)).all():
            r.has_blinks, r.blink_count = True, 1
    assert burst_api.client.get(f"/api/burst/{burst_api.burst}").json()["can_reject_blinks"] is False
    r = burst_api.client.post(f"/api/bursts/{burst_api.burst}/reject-blinks").json()
    assert r["moved"] == 0 and set(_buckets(burst_api).values()) == {None}


def test_reject_blinks_needs_auth_and_a_real_burst(burst_api):
    assert TestClient(burst_api.client.app).post(f"/api/bursts/{burst_api.burst}/reject-blinks").status_code == 401
    assert burst_api.client.post("/api/bursts/9999/reject-blinks").status_code == 404


# ------------------------------------------------------------ repository rules
def test_no_insightface_in_the_running_package():
    out = subprocess.run([sys.executable, "-I", "-c",
                          "import sys, photocure, photocure.cli, photocure.ui.app, photocure.inference.pipeline;"
                          "assert 'insightface' not in sys.modules"],
                         capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": ""})
    assert out.returncode == 0, out.stderr


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_no_insightface_text_outside_the_allowed_files():
    r = subprocess.run(["git", "grep", "-iq", "insightface", "--", ".", ":!tests", ":!CHANGELOG*",
                        ":!implementation_plan.md", ":!NOTICE", ":!docs",
                        ":!.github/workflows/ci.yml"], cwd=ROOT)
    assert r.returncode == 1  # 1 means no match


def test_server_warns_at_startup_about_an_altered_model(tmp_path, monkeypatch, caplog):
    import logging

    from photocure.ui.app import create_app
    monkeypatch.setenv("PCUR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(models, "verify_model", lambda name, *a, **k: name != models.LANDMARKER)
    app = create_app(str(tmp_path / "s.db"))
    with caplog.at_level(logging.WARNING, logger="photocure.ui"):
        with TestClient(app):  # runs the startup hooks
            pass
    assert any(models.LANDMARKER in r.getMessage() and "MANIFEST" in r.getMessage() for r in caplog.records)
    assert not any(models.YUNET in r.getMessage() for r in caplog.records)
