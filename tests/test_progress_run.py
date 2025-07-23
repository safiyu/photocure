"""Every task gets its own run id, so a dismissed progress card stays dismissed."""
from photocure.ui.app import ProgressManager


def test_each_started_task_gets_a_new_run_id():
    pm = ProgressManager()
    before = pm.get()["run"]
    pm.start("Scan")
    first = pm.get()["run"]
    pm.finish("Scan complete")
    assert pm.get()["run"] == first, "finishing must not change the run id"
    pm.start("Scan")
    assert pm.get()["run"] == first + 1 > before


def test_error_and_stop_keep_the_run_id():
    pm = ProgressManager()
    pm.start("Scan")
    run = pm.get()["run"]
    pm.error("boom")
    assert pm.get()["run"] == run
    pm.start("Scan")
    run = pm.get()["run"]
    pm.stop("Stopped")
    assert pm.get()["run"] == run


def test_reset_does_not_reuse_a_run_id():
    pm = ProgressManager()
    pm.start("Scan")
    used = pm.get()["run"]
    pm.reset()
    pm.start("Scan")
    assert pm.get()["run"] > used
