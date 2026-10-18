"""The analysis counter shows photos, not the pipeline's two-steps-per-photo total."""
from photocure.ui.app import infer_progress_detail


def test_no_total_means_no_detail():
    assert infer_progress_detail(0, 0) == ""


def test_start_of_phase_one():
    assert infer_progress_detail(0, 684) == "0 / 342 photos"


def test_counts_photos_through_phase_one():
    assert infer_progress_detail(100, 684) == "100 / 342 photos"
    assert infer_progress_detail(342, 684) == "342 / 342 photos"


def test_phase_two_restarts_the_photo_count():
    assert infer_progress_detail(343, 684) == "1 / 342 photos"
    assert infer_progress_detail(684, 684) == "342 / 342 photos"


def test_never_reports_more_photos_than_the_library_batch():
    for done in range(0, 685, 17):
        n, _, total = infer_progress_detail(done, 684).partition(" / ")
        assert int(n) <= int(total.split()[0]) == 342
