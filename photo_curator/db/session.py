"""SQLite engine factory & session management.

Configures WAL journal mode, ``synchronous=NORMAL``, a 64 MiB page cache and
foreign-key enforcement on every connection.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator

from sqlalchemy import create_engine, event
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


def get_engine(db_path: Path | str = "photo_curator.db", echo: bool = False):
    """Create (or reuse) a SQLite engine for ``db_path`` with WAL tuning."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path.resolve()}"
    cached = _engine_cache.get(url)
    if cached is not None:
        return cached

    engine = create_engine(
        url,
        echo=echo,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    event.listens_for(engine, "connect")(_apply_pragmas)
    _engine_cache[url] = engine
    return engine


def init_db(db_path: Path | str = "photo_curator.db", drop: bool = False) -> None:
    """Create all tables. With ``drop=True`` the schema is rebuilt from scratch."""
    engine = get_engine(db_path)
    if drop:
        SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session(db_path: Path | str = "photo_curator.db") -> Iterator[_Session]:
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


def close_engine(db_path: Path | str = "photo_curator.db") -> None:
    """Dispose of a cached engine (used by tests & the Web server on shutdown)."""
    url = f"sqlite:///{Path(db_path).resolve()}"
    engine = _engine_cache.pop(url, None)
    if engine is not None:
        engine.dispose()
