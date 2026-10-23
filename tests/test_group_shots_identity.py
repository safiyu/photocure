"""The face matrix must follow people, not screen positions."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from photocure.clustering.group_shots import FaceObs, _subjects, burst_face_matrix


def _box(cx: float, cy: float = 0.4, w: float = 0.08, h: float = 0.12):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _obs(cx, pid=None, emb=None, face_id=None):
    return FaceObs(box=_box(cx), person_id=pid, emb=emb, face_id=face_id)


def _pids(subject: dict[int, FaceObs]):
    return {o.person_id for o in subject.values()}


def test_people_who_shifted_places_keep_their_own_column():
    """Frame 1 misses the fifth person and everyone from the third on stands one slot further left."""
    frame0 = [_obs(0.1, 1), _obs(0.3, 2), _obs(0.7, 3), _obs(0.9, 4)]
    frame1 = [_obs(0.1, 1), _obs(0.3, 2), _obs(0.5, 3), _obs(0.7, 4), _obs(0.9, 5)]
    subjects = _subjects([frame0, frame1])
    assert len(subjects) == 5
    for s in subjects:
        assert len(_pids(s)) == 1, "a column mixed two different people"
    by_pid = {next(iter(_pids(s))): s for s in subjects}
    assert set(by_pid) == {1, 2, 3, 4, 5}
    assert set(by_pid[3]) == {0, 1} and set(by_pid[4]) == {0, 1}
    assert set(by_pid[5]) == {1}


def test_embeddings_decide_when_there_are_no_person_clusters():
    def vec(i):
        v = [0.0] * 8
        v[i] = 1.0
        return tuple(v)

    frame0 = [_obs(0.1, emb=vec(0)), _obs(0.7, emb=vec(1)), _obs(0.9, emb=vec(2))]
    frame1 = [_obs(0.1, emb=vec(0)), _obs(0.5, emb=vec(1)), _obs(0.7, emb=vec(2))]
    subjects = _subjects([frame0, frame1])
    assert len(subjects) == 3
    for s in subjects:
        embs = {o.emb for o in s.values()}
        assert len(embs) == 1


def test_two_different_people_never_merge_just_because_they_stand_close():
    subjects = _subjects([[_obs(0.5, 1)], [_obs(0.5, 2)]])
    assert len(subjects) == 2


def test_the_same_person_is_followed_across_a_long_move():
    subjects = _subjects([[_obs(0.1, 7)], [_obs(0.45, 7)], [_obs(0.9, 7)]])
    assert len(subjects) == 1 and set(subjects[0]) == {0, 1, 2}


def test_without_any_identity_nearby_faces_still_track_by_position():
    frame0 = [_obs(0.2), _obs(0.6)]
    frame1 = [_obs(0.21), _obs(0.61)]
    subjects = _subjects([frame0, frame1])
    assert len(subjects) == 2 and all(len(s) == 2 for s in subjects)


def _face(fid, cx, pid=None, emb=None):
    x1, y1, x2, y2 = _box(cx)
    return SimpleNamespace(id=fid, x1=x1, y1=y1, x2=x2, y2=y2, person_id=pid, eye_status="open",
                           eye_closed_score=0.1, face_sharpness=500.0, is_blink=False,
                           embedding_json=json.dumps(list(emb)) if emb else None)


def _member(i, faces):
    t = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc) + timedelta(seconds=i)
    return SimpleNamespace(id=i + 1, filename=f"f{i}.jpg", taken_at=t, faces=faces, quality=None,
                           is_best_shot=i == 0, delete_approved=False, has_blinks=False,
                           width=100, height=100)


def test_the_matrix_columns_follow_people_in_the_api_payload():
    f0 = _member(0, [_face(1, 0.1, 1), _face(2, 0.3, 2), _face(3, 0.7, 3), _face(4, 0.9, 4)])
    f1 = _member(1, [_face(5, 0.1, 1), _face(6, 0.3, 2), _face(7, 0.5, 3), _face(8, 0.7, 4), _face(9, 0.9, 5)])
    m = burst_face_matrix([f0, f1])
    assert len(m["subjects"]) == 5
    pid_of = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5}
    for col in range(5):
        seen = {pid_of[row["faces"][col]["face_id"]] for row in m["rows"] if row["faces"][col]}
        assert len(seen) == 1, f"column {col} mixes people: {seen}"
