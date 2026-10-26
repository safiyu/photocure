"""C1 acceptance: photocure stop only touches a real Photocure server."""
import os
import subprocess
import sys
import threading
import time

import pytest
from typer.testing import CliRunner

from photocure import cli as cli_mod
from photocure.lifecycle import is_photocure_server, pid_file_path, process_cmdline

runner = CliRunner()


@pytest.mark.parametrize("cmd,expected", [
    ("/usr/bin/python3 /home/u/.venv/bin/photocure serve --port 8080", True),
    ("python -m photocure start --no-open", True),
    ("/usr/bin/photocure setup --yes", True),
    ("vim /home/u/notes/photocure.txt", False),
    ("tail -f photocure.log", False),
    ("python photocure_helper.py serve", False),
    ("/usr/bin/photocure stats", False),
    ("", False),
    (None, False),
])
def test_cmdline_matching(cmd, expected):
    assert is_photocure_server(cmd) is expected


@pytest.fixture()
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    return tmp_path


def _spawn(*marker):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", *marker])
    # Right after the spawn, ps can still show the parent's command line; wait for the marker.
    end = time.time() + 10
    while time.time() < end and marker[-1] not in (process_cmdline(proc.pid) or ""):
        time.sleep(0.05)
    return proc


def _alive(p):
    return p.poll() is None


def test_decoy_with_similar_name_survives(runtime):
    decoy = _spawn("photocure.log")
    try:
        pf = pid_file_path(8101)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(str(decoy.pid))
        r = runner.invoke(cli_mod.app, ["stop", "--port", "8101", "--force"])
        assert r.exit_code == 1
        time.sleep(0.3)
        assert _alive(decoy)
    finally:
        decoy.kill()
        decoy.wait()


def test_real_server_process_is_stopped_and_pid_file_removed(runtime):
    server = _spawn("photocure", "serve")
    # A real server is reaped by its shell or init; here pytest is the parent, so reap it.
    reaper = threading.Thread(target=server.wait, daemon=True)
    reaper.start()
    try:
        pf = pid_file_path(8102)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(str(server.pid))
        r = runner.invoke(cli_mod.app, ["stop", "--port", "8102"])
        assert r.exit_code == 0, r.output
        server.wait(timeout=5)
        assert not pf.exists()
    finally:
        if _alive(server):
            server.kill()


def test_stop_with_nothing_running_exits_zero(runtime):
    r = runner.invoke(cli_mod.app, ["stop", "--port", "8103"])
    assert r.exit_code == 0 and "No active Photocure server found on port 8103" in r.output


def test_kill_is_stop_force(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli_mod, "stop", lambda port, force: seen.update(port=port, force=force))
    r = runner.invoke(cli_mod.app, ["kill", "--port", "8104"])
    assert r.exit_code == 0 and seen == {"port": 8104, "force": True}


def test_db_restore_roundtrip_and_refusals(tmp_path, monkeypatch):
    import sqlite3
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    db = tmp_path / "p.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t(x)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    con.close()
    from photocure.db.session import snapshot_database
    snap = snapshot_database(db, tmp_path / "bk")
    con = sqlite3.connect(db)
    con.execute("INSERT INTO t VALUES (2)")
    con.commit()
    con.close()
    r = runner.invoke(cli_mod.app, ["db", "restore", str(snap), "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert sqlite3.connect(db).execute("SELECT count(*) FROM t").fetchone()[0] == 1
    assert (tmp_path / "p.db.pre-restore").exists()
    bad = tmp_path / "bad.bak.db"
    bad.write_bytes(b"not a database" * 100)
    r = runner.invoke(cli_mod.app, ["db", "restore", str(bad), "--db", str(db)])
    assert r.exit_code != 0
    assert sqlite3.connect(db).execute("SELECT count(*) FROM t").fetchone()[0] == 1
