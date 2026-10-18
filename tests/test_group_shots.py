"""Group scorer: the frame where the weakest face is strongest, or a pair to composite."""
from photocure.clustering.group_shots import FaceObs, score_group


def face(x, status="open", closed=0.1, sharp=100.0, pid=None, y=0.3):
    return FaceObs(box=(x, y, x + 0.1, y + 0.2), eye_status=status, eye_closed=closed,
                   sharpness=sharp, person_id=pid)


A, B, C = 0.1, 0.4, 0.7


def test_everyone_open_is_perfect_and_picks_the_sharpest_frame():
    frames = [
        [face(A, sharp=80), face(B, sharp=80), face(C, sharp=80)],
        [face(A, sharp=100), face(B, sharp=100), face(C, sharp=100)],
        [face(A, sharp=120), face(B, sharp=120), face(C, sharp=120)],
    ]
    m = score_group(frames)
    assert m.status == "perfect" and m.subject_count == 3 and m.best == 2 and m.composite is None


def test_one_blinker_per_frame_picks_the_best_minimum_and_offers_a_composite():
    frames = [
        [face(A, "closed", 0.9), face(B), face(C)],
        [face(A), face(B, "possible", 0.6), face(C)],
        [face(A), face(B), face(C, "closed", 0.9)],
    ]
    m = score_group(frames)
    assert m.best == 1 and m.status == "composite"
    assert m.composite[0] == 1 and m.composite[1] in (0, 2)


def test_unfixable_burst_is_best_effort():
    frames = [[face(A, "closed", 0.9), face(B)], [face(A, "closed", 0.9), face(B)]]
    m = score_group(frames)
    assert m.status == "best_effort" and m.best in (0, 1) and m.composite is None


def test_lone_faces_and_lone_frames_are_not_groups():
    assert score_group([[face(A)], [face(A)], [face(A)]]).status == "not_group"
    assert score_group([[face(A), face(B)]]).status == "not_group"
    assert score_group([]).status == "not_group"
    assert score_group([[], []]).status == "not_group"


def test_a_subject_missing_from_a_frame_is_penalised():
    frames = [[face(A), face(B)], [face(A)], [face(A), face(B)]]
    m = score_group(frames)
    assert m.frames[1].min_quality is not None and m.frames[1].min_quality <= 0.31
    assert m.frames[1].subjects[1].state == "missing"
    assert m.best in (0, 2)


def test_people_sharing_a_cluster_id_in_one_frame_stay_separate():
    frames = [[face(A, pid=5), face(B, pid=5)], [face(A, pid=5), face(B, pid=5)]]
    assert score_group(frames).subject_count == 2


def test_person_id_bridges_a_gap_where_the_tracker_lost_the_face():
    frames = [[face(A, pid=7), face(B)], [face(B)], [face(A, pid=7), face(B)]]
    m = score_group(frames)
    assert m.subject_count == 2
    assert [s.state for s in m.frames[1].subjects].count("missing") == 1


def test_faces_with_no_box_and_no_identity_are_ignored():
    ghost = FaceObs(box=None, eye_status="closed", eye_closed=1.0, sharpness=None, person_id=None)
    frames = [[face(A), face(B), ghost], [face(A), face(B), ghost]]
    m = score_group(frames)
    assert m.subject_count == 2 and m.status == "perfect"


def test_tiebreak_prefers_the_higher_value_then_the_earlier_frame():
    frames = [[face(A), face(B)], [face(A), face(B)]]
    assert score_group(frames).best == 0
    assert score_group(frames, tiebreak=[0.1, 0.9]).best == 1


def test_a_face_the_detector_missed_for_one_frame_is_still_the_same_subject():
    frames = [[face(A), face(B)], [face(A)], [face(A), face(B)], [face(A), face(B)]]
    m = score_group(frames)
    assert m.subject_count == 2 and m.frames[1].subjects[1].state == "missing"


def test_bridging_never_joins_two_faces_that_share_a_frame():
    frames = [[face(A), face(A + 0.01)], [face(A), face(A + 0.01)], [face(A), face(A + 0.01)]]
    assert score_group(frames).subject_count == 2
