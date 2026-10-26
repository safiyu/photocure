"""SQLite engine factory & session management.

Configures WAL journal mode, ``synchronous=NORMAL``, a 64 MiB page cache and
foreign-key enforcement on every connection.
"""
from __future__ import annotations

import contextlib
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.dialects import sqlite
from sqlmodel import Session, SQLModel, Session as _Session

from . import models  # noqa: F401  (register tables with SQLModel.metadata)

_engine_cache: dict[str, object] = {}


def _apply_pragmas(dbapi_conn, _connection_record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA cache_size=-64000;")  # -64000 == 64 MiB page cache
    cur.execute("PRAGMA foreign_keys=ON;")
    cur.execute("PRAGMA busy_timeout=30000;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.close()


def get_engine(db_path: Path | str = "photocure.db", echo: bool = False):
    """Create (or reuse) a SQLite engine for ``db_path`` with WAL tuning."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path.resolve()}"
    cached = _engine_cache.get(url)
    if cached is not None:
        return cached

    if len(_engine_cache) >= 16:
        to_del = []
        for u in list(_engine_cache.keys()):
            path_str = u.removeprefix("sqlite:///")
            if not Path(path_str).exists():
                to_del.append(u)
        for u in to_del:
            old = _engine_cache.pop(u, None)
            if old is not None:
                try:
                    old.dispose()
                except Exception:
                    pass

    engine = create_engine(
        url,
        echo=echo,
        connect_args={"check_same_thread": False, "timeout": 30},
        pool_pre_ping=True,
    )
    event.listens_for(engine, "connect")(_apply_pragmas)
    _engine_cache[url] = engine
    return engine


SCHEMA_VERSION = 12  # 12: ledger remembers the photo's state before each action; 11: image stage and origin_path, person stage; 10: face ignored column; 9: delete_approved; 8: aesthetic_model, preference, preference_models; 7: file_mtime; 6: geo/favourite columns and indexes; 5: blink/face columns; 4: xmp_written_*


def _sql_type(col) -> str:
    try:
        t = col.type.compile(dialect=sqlite.dialect())
    except Exception:
        t = "TEXT"
    return t


def _default_clause(col) -> str:
    d = getattr(col.default, "arg", None) if col.default is not None else None
    if isinstance(d, bool):
        return f" DEFAULT {int(d)}"
    if isinstance(d, (int, float)):
        return f" DEFAULT {d}"
    if isinstance(d, str):
        return " DEFAULT '" + d.replace("'", "''") + "'"
    if callable(d) and not col.nullable:
        return " DEFAULT " + _callable_default_literal(col, d)
    return ""


def _callable_default_literal(col, fn) -> str:
    """SQL literal for a callable default, evaluated once for existing rows."""
    try:
        v = fn(None) if fn.__code__.co_argcount else fn()
    except Exception:
        v = None
    if isinstance(v, datetime):
        return "'" + v.strftime("%Y-%m-%d %H:%M:%S.%f") + "'"
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return "''" if "CHAR" in _sql_type(col).upper() or "TEXT" in _sql_type(col).upper() else "0"


def migrate(engine) -> int:
    """Add columns missing from existing tables and bump PRAGMA user_version."""
    added = 0
    added_cols: set[tuple[str, str]] = set()
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            rows = conn.exec_driver_sql(f'PRAGMA table_info("{table.name}")').fetchall()
            if not rows:
                continue
            have = {r[1] for r in rows}
            for col in table.columns:
                if col.name in have:
                    continue
                ddl = (f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" '
                       f"{_sql_type(col)}{_default_clause(col)}")
                conn.exec_driver_sql(ddl)
                added += 1
                added_cols.add((table.name, col.name))
        if ("images", "stage") in added_cols:
            # Rows from before stages existed take their stage from how they were routed.
            conn.exec_driver_sql("UPDATE images SET stage='library' WHERE route_status='executed'")
            conn.exec_driver_sql("UPDATE images SET stage='trash' WHERE route_status='trashed'")
            conn.exec_driver_sql(
                "UPDATE images SET origin_path=path WHERE stage='library' AND origin_path IS NULL")
        # Columns added above are not indexed by create_all on an existing table, so create any
        # index the models declare that the database does not have yet.
        for table in SQLModel.metadata.sorted_tables:
            if not conn.exec_driver_sql(f'PRAGMA table_info("{table.name}")').fetchall():
                continue
            for index in table.indexes:
                index.create(conn, checkfirst=True)
        conn.exec_driver_sql(f"PRAGMA user_version={SCHEMA_VERSION}")
    return added


def init_db(db_path: Path | str = "photocure.db", drop: bool = False) -> None:
    """Create all tables. With ``drop=True`` the schema is rebuilt from scratch."""
    engine = get_engine(db_path)
    if drop:
        SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    migrate(engine)


@contextmanager
def get_session(db_path: Path | str = "photocure.db") -> Iterator[_Session]:
    """Context-managed session with commit/rollback semantics.

    Usage::

        with get_session(db) as s:
            s.add(obj)  # committed on clean exit, rolled back on exception
    """
    engine = get_engine(db_path)
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def close_engine(db_path: Path | str = "photocure.db") -> None:
    """Dispose of a cached engine (used by tests & the Web server on shutdown)."""
    url = f"sqlite:///{Path(db_path).resolve()}"
    engine = _engine_cache.pop(url, None)
    if engine is not None:
        engine.dispose()


def snapshot_database(db_path: Path | str = "photocure.db", backup_dir: Path | None = None, keep: int = 10) -> Path:
    """Online SQLite backup with UTC timestamp and rotation (keep N newest)."""
    src = Path(db_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Database file not found: {src}")
    b_dir = backup_dir or (src.parent / "backups")
    b_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = b_dir / f"{src.stem}_{timestamp}.bak.db"
    with contextlib.closing(sqlite3.connect(str(src), timeout=30.0)) as src_conn:
        with contextlib.closing(sqlite3.connect(str(dst))) as dst_conn:
            src_conn.backup(dst_conn, pages=100)
    backups = sorted(b_dir.glob(f"{src.stem}_*.bak.db"))
    while len(backups) > keep:
        oldest = backups.pop(0)
        try:
            oldest.unlink(missing_ok=True)
        except OSError:
            pass
    return dst


def generate_otc() -> str:
    """Generate a 32-byte URL-safe one-time code."""
    return secrets.token_urlsafe(32)
