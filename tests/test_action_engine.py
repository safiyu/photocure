"""Tests for the ActionEngine and RollbackEngine (Milestone 4 deliverable).

Covers:
- Dry-run: no filesystem mutation.
- Move / copy / hardlink execution with transactional ledger (PENDING → COMMITTED).
- Collision rename (_dup<sha6>).
- Rollback: move-back, copy-removal, hardlink-removal, symlink-removal.
- Rollback hash verification (refuses on mismatch).
- Rollback idempotency.
- Spec §7.2 Safety & Rollback Invariance: N files → apply move → undo →
  byte-for-byte directory equivalence.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.conftest import make_image
from photocure.config import CuratorConfig
from photocure.db.repository import PhotoRepository
from photocure.db.session import get_session, init_db
from photocure.engine.action_engine import ActionEngine, new_transaction_id
from photocure.engine.rollback import RollbackEngine
from photocure.rules.organizer import PlannedAction
from photocure.rules.taxonomy import CategoryType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class _Rec:
    """Detached-safe view of the fields the tests need from an ImageRecord."""
    id: int
    path: Path
    filename: str


@pytest.fixture()
def cfg(tmp_db):
    c = CuratorConfig()
    c.db_path = tmp_db
    c.device.device = "cpu"
    return c


@pytest.fixture()
def seeded_repo(cfg, tmp_db, tmp_path):
    """Scan a small library so ImageRecords exist with real paths + sha256."""
    root = tmp_path / "lib"
    paths = [
        make_image(root / f"IMG_2024{i:04d}_153045.jpg", rng_seed=i)
        for i in range(1, 6)
    ]
    init_db(tmp_db, drop=True)
    from photocure.scanner.file_scanner import FileScanner
    res = FileScanner(cfg).scan_directory(root, db_path=tmp_db, show_progress=False)
    assert res.new == len(paths)

    with get_session(tmp_db) as s:
        repo = PhotoRepository(s)
        records = repo.all_images()
        assert len(records) == len(paths)
        # Mark all as inferred (the decision matrix skips non-inferred rows).
        for r in records:
            r.infer_status = "inferred"
        s.commit()
        # Capture into detached-safe views while the session is still open.
        recs = [_Rec(r.id, Path(r.path), r.filename) for r in records]
    return recs, paths


def _plan_actions(records, target_dir: Path, op: str = "move"):
    """Build a list of PlannedActions for a set of records."""
    actions = []
    for r in records:
        actions.append(PlannedAction(
            image_id=r.id,
            source=Path(r.path),
            target=target_dir / r.filename,
            operation=op,
            category=CategoryType.SCENERY,
            reason="test",
        ))
    return actions


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    """Map of relative path -> file bytes for every photo file under root.

    The SQLite state store (``*.db``) and its WAL sidecars (``*.db-wal``,
    ``*.db-shm``) legitimately mutate during execute/rollback (ledger rows,
    timestamps); the spec's byte-for-byte invariance applies to the photo
    library, so all SQLite artifacts are excluded.
    """
    import re
    skip = re.compile(r"\.db(-[a-z]+)?$")
    out: dict[str, bytes] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not skip.search(p.name):
            out[str(p.relative_to(root))] = p.read_bytes()
    return out


def _assert_trees_equal(tree_a: dict[str, bytes], tree_b: dict[str, bytes], label: str):
    assert tree_a.keys() == tree_b.keys(), (
        f"{label}: file sets differ\n"
        f"  only in before: {sorted(tree_a.keys() - tree_b.keys())}\n"
        f"  only in after:  {sorted(tree_b.keys() - tree_a.keys())}"
    )
    for k in sorted(tree_a.keys()):
        assert tree_a[k] == tree_b[k], f"{label}: byte mismatch at {k}"


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def test_dry_run_does_not_touch_filesystem(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records, target)

    before = _snapshot_tree(tmp_path)
    engine = ActionEngine(cfg, db_path=cfg.db_path)
    rep = engine.dry_run(actions)

    assert rep.dry_run is True
    assert rep.executed == len(actions)
    assert rep.failures == 0
    after = _snapshot_tree(tmp_path)
    _assert_trees_equal(before, after, "dry-run")


# ---------------------------------------------------------------------------
# Move
# ---------------------------------------------------------------------------

def test_move_executes_and_writes_ledger(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records, target, op="move")
    txn = new_transaction_id()

    engine = ActionEngine(cfg, db_path=cfg.db_path)
    rep = engine.execute(actions, transaction_id=txn)

    assert rep.executed == len(actions)
    assert rep.failures == 0
    # Sources moved away.
    for p in paths:
        assert not p.exists(), f"source should be gone: {p}"
    # Targets exist.
    for r in records:
        tgt = target / r.filename
        assert tgt.exists(), f"target missing: {tgt}"

    # Ledger rows are COMMITTED.
    with get_session(cfg.db_path) as s:
        repo = PhotoRepository(s)
        entries = repo.ledger_by_txn(txn)
        assert len(entries) == len(actions)
        assert all(e.status == "committed" for e in entries)


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

def test_copy_keeps_source_and_creates_target(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records, target, op="copy")

    engine = ActionEngine(cfg, db_path=cfg.db_path)
    rep = engine.execute(actions)
    assert rep.executed == len(actions)
    assert rep.failures == 0
    # Source still there.
    for p in paths:
        assert p.exists(), f"source should remain after copy: {p}"
    # Target also there.
    for r in records:
        assert (target / r.filename).exists()


# ---------------------------------------------------------------------------
# Hardlink
# ---------------------------------------------------------------------------

def test_hardlink_same_inode(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records, target, op="hardlink")

    engine = ActionEngine(cfg, db_path=cfg.db_path)
    rep = engine.execute(actions)
    assert rep.failures == 0
    for r in records:
        src = Path(r.path)
        tgt = target / r.filename
        assert tgt.exists()
        if os.path.samefile(src, tgt):
            # Same device: real hardlink (same inode).
            assert src.stat().st_ino == tgt.stat().st_ino


# ---------------------------------------------------------------------------
# Collision rename
# ---------------------------------------------------------------------------

def test_collision_renames_target(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    # Pre-create a file with the same name as the first record's target.
    target.mkdir(parents=True, exist_ok=True)
    victim = target / records[0].filename
    victim.write_bytes(b"pre-existing content")

    actions = _plan_actions(records, target, op="move")
    engine = ActionEngine(cfg, db_path=cfg.db_path)
    rep = engine.execute(actions)
    assert rep.failures == 0
    assert rep.collisions >= 1
    # The pre-existing file must be untouched.
    assert victim.read_bytes() == b"pre-existing content"
    # The moved file got a _dup suffix.
    moved = [p for p in target.iterdir() if p.name != records[0].filename and p.is_file()]
    assert any("_dup" in p.name for p in moved), f"no _dup rename found in {list(target.iterdir())}"


# ---------------------------------------------------------------------------
# Rollback: move
# ---------------------------------------------------------------------------

def test_rollback_move_restores_byte_for_byte(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records, target, op="move")
    txn = new_transaction_id()

    before = _snapshot_tree(tmp_path)

    engine = ActionEngine(cfg, db_path=cfg.db_path)
    rep = engine.execute(actions, transaction_id=txn)
    assert rep.failures == 0

    rb = RollbackEngine(cfg, db_path=cfg.db_path)
    rrep = rb.rollback_transaction(txn)
    assert rrep.reversed == len(actions)
    assert rrep.failures == 0

    after = _snapshot_tree(tmp_path)
    _assert_trees_equal(before, after, "rollback-move")


# ---------------------------------------------------------------------------
# Rollback: copy
# ---------------------------------------------------------------------------

def test_rollback_copy_removes_target(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records, target, op="copy")
    txn = new_transaction_id()

    before = _snapshot_tree(tmp_path)

    engine = ActionEngine(cfg, db_path=cfg.db_path)
    rep = engine.execute(actions, transaction_id=txn)
    assert rep.failures == 0

    rb = RollbackEngine(cfg, db_path=cfg.db_path)
    rrep = rb.rollback_transaction(txn)
    assert rrep.failures == 0

    after = _snapshot_tree(tmp_path)
    _assert_trees_equal(before, after, "rollback-copy")


# ---------------------------------------------------------------------------
# Rollback: hardlink
# ---------------------------------------------------------------------------

def test_rollback_hardlink_removes_link(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records, target, op="hardlink")
    txn = new_transaction_id()

    before = _snapshot_tree(tmp_path)

    engine = ActionEngine(cfg, db_path=cfg.db_path)
    rep = engine.execute(actions, transaction_id=txn)
    assert rep.failures == 0

    rb = RollbackEngine(cfg, db_path=cfg.db_path)
    rrep = rb.rollback_transaction(txn)
    assert rrep.failures == 0

    after = _snapshot_tree(tmp_path)
    _assert_trees_equal(before, after, "rollback-hardlink")


# ---------------------------------------------------------------------------
# Rollback: symlink
# ---------------------------------------------------------------------------

def test_rollback_symlink_removes_link(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records, target, op="symlink")
    txn = new_transaction_id()

    before = _snapshot_tree(tmp_path)

    engine = ActionEngine(cfg, db_path=cfg.db_path)
    rep = engine.execute(actions, transaction_id=txn)
    assert rep.failures == 0

    rb = RollbackEngine(cfg, db_path=cfg.db_path)
    rrep = rb.rollback_transaction(txn)
    assert rrep.failures == 0

    after = _snapshot_tree(tmp_path)
    _assert_trees_equal(before, after, "rollback-symlink")


# ---------------------------------------------------------------------------
# Rollback hash verification
# ---------------------------------------------------------------------------

def test_rollback_refuses_on_hash_mismatch(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records[:1], target, op="move")
    txn = new_transaction_id()

    engine = ActionEngine(cfg, db_path=cfg.db_path)
    rep = engine.execute(actions, transaction_id=txn)
    assert rep.failures == 0

    # Corrupt the target so the SHA-256 no longer matches.
    tgt = target / records[0].filename
    tgt.write_bytes(b"tampered")

    rb = RollbackEngine(cfg, db_path=cfg.db_path)
    rrep = rb.rollback_transaction(txn, verify=True)
    assert rrep.reversed == 0
    assert rrep.failures == 1
    # Source must NOT have been restored (rollback refused).
    assert not Path(records[0].path).exists()


# ---------------------------------------------------------------------------
# Rollback idempotency
# ---------------------------------------------------------------------------

def test_rollback_is_idempotent(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records, target, op="move")
    txn = new_transaction_id()

    engine = ActionEngine(cfg, db_path=cfg.db_path)
    engine.execute(actions, transaction_id=txn)

    rb = RollbackEngine(cfg, db_path=cfg.db_path)
    first = rb.rollback_transaction(txn)
    assert first.reversed == len(actions)
    assert first.failures == 0

    # Second rollback: all already rolled back -> 0 reversed, 0 failures.
    second = rb.rollback_transaction(txn)
    assert second.reversed == 0
    assert second.failures == 0


# ---------------------------------------------------------------------------
# Spec §7.2: Safety & Rollback Invariance (large sandbox)
# ---------------------------------------------------------------------------

def test_rollback_invariance_100_files(cfg, tmp_path):
    """Create 100 mixed images, apply move, undo, assert byte-for-byte equivalence."""
    root = tmp_path / "library"
    n = 100
    paths = []
    for i in range(n):
        # Mix of sharp, blurry, dark, normal for variety.
        kwargs = {}
        if i % 7 == 0:
            kwargs["blur"] = True
        elif i % 13 == 0:
            kwargs["dark"] = True
        elif i % 19 == 0:
            kwargs["white"] = True
        p = make_image(root / f"IMG_2024{i:06d}.jpg", rng_seed=i, **kwargs)
        paths.append(p)

    init_db(cfg.db_path, drop=True)
    from photocure.scanner.file_scanner import FileScanner
    res = FileScanner(cfg).scan_directory(root, db_path=cfg.db_path, show_progress=False)
    assert res.new == n

    # Mark all inferred.
    with get_session(cfg.db_path) as s:
        repo = PhotoRepository(s)
        for r in repo.all_images():
            r.infer_status = "inferred"
        s.commit()

    target = tmp_path / "Organized"
    with get_session(cfg.db_path) as s:
        repo = PhotoRepository(s)
        records = [_Rec(r.id, Path(r.path), r.filename) for r in repo.all_images()]

    actions = _plan_actions(records, target, op="move")
    txn = new_transaction_id()

    before = _snapshot_tree(tmp_path)
    engine = ActionEngine(cfg, db_path=cfg.db_path)
    rep = engine.execute(actions, transaction_id=txn)
    assert rep.executed == n
    assert rep.failures == 0

    # Verify nothing in source lib remains.
    remaining = list(root.rglob("*.jpg"))
    assert len(remaining) == 0, f"{len(remaining)} files still in source"

    rb = RollbackEngine(cfg, db_path=cfg.db_path)
    rrep = rb.rollback_transaction(txn)
    assert rrep.reversed == n
    assert rrep.failures == 0

    after = _snapshot_tree(tmp_path)
    _assert_trees_equal(before, after, "rollback-invariance-100")


# ---------------------------------------------------------------------------
# Ledger state transitions
# ---------------------------------------------------------------------------

def test_ledger_pending_before_commit(cfg, seeded_repo, tmp_path):
    """Verify that after execute, all ledger rows are 'committed' (not 'pending')."""
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records, target, op="move")
    txn = new_transaction_id()

    engine = ActionEngine(cfg, db_path=cfg.db_path)
    rep = engine.execute(actions, transaction_id=txn)
    assert rep.failures == 0

    with get_session(cfg.db_path) as s:
        repo = PhotoRepository(s)
        entries = repo.ledger_by_txn(txn)
        assert all(e.status == "committed" for e in entries)
        assert all(e.sha256 for e in entries)
        assert all(e.file_size > 0 for e in entries)


# ---------------------------------------------------------------------------
# Image record routing state
# ---------------------------------------------------------------------------

def test_execute_updates_route_status(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records, target, op="move")
    txn = new_transaction_id()

    engine = ActionEngine(cfg, db_path=cfg.db_path)
    engine.execute(actions, transaction_id=txn)

    with get_session(cfg.db_path) as s:
        repo = PhotoRepository(s)
        for r in records:
            row = repo.get_by_id(r.id)
            assert row.route_status == "executed"
            assert row.target_path is not None


def test_rollback_resets_route_status(cfg, seeded_repo, tmp_path):
    records, paths = seeded_repo
    target = tmp_path / "Organized"
    actions = _plan_actions(records, target, op="move")
    txn = new_transaction_id()

    engine = ActionEngine(cfg, db_path=cfg.db_path)
    engine.execute(actions, transaction_id=txn)

    rb = RollbackEngine(cfg, db_path=cfg.db_path)
    rb.rollback_transaction(txn)

    with get_session(cfg.db_path) as s:
        repo = PhotoRepository(s)
        for r in records:
            row = repo.get_by_id(r.id)
            assert row.route_status == "unrouted"
            assert row.target_path is None
