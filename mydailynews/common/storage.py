from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


MEMORY_DATABASE_NAME = "memory.sqlite3"
CACHE_DATABASE_NAME = "cache.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stories (
    story_key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS coverage (
    date TEXT NOT NULL,
    brief_name TEXT NOT NULL,
    story_key TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (date, brief_name, story_key)
);
CREATE TABLE IF NOT EXISTS coverage_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    archived_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS json_cache (
    namespace TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    cached_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (namespace, cache_key)
);
CREATE TABLE IF NOT EXISTS http_cache (
    namespace TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    url TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    body TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    etag TEXT NOT NULL DEFAULT '',
    last_modified TEXT NOT NULL DEFAULT '',
    content_type TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (namespace, cache_key)
);
"""


def _sqlite_module() -> Any:
    try:
        import sqlite3
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "This Python installation cannot load the standard-library sqlite3 module; "
            "install or allow a Python build with SQLite support."
        ) from exc
    return sqlite3


@contextmanager
def open_database(path: Path | str, *, immediate: bool = False) -> Iterator[Any]:
    """Open the project's small local database for one atomic operation."""

    sqlite3 = _sqlite_module()
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.executescript(_SCHEMA)
        if immediate:
            connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def backup_database(source: Path | str, target: Path | str) -> None:
    """Create a transactionally consistent SQLite backup using the stdlib API."""

    sqlite3 = _sqlite_module()
    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_connection = sqlite3.connect(source_path, timeout=30.0)
    target_connection = sqlite3.connect(target_path, timeout=30.0)
    try:
        source_connection.execute("PRAGMA busy_timeout = 30000")
        source_connection.backup(target_connection)
        target_connection.commit()
    finally:
        target_connection.close()
        source_connection.close()


def metadata_value(connection: Any, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row is not None else None


def claim_migration(connection: Any, key: str) -> bool:
    cursor = connection.execute(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, 'in_progress')",
        (key,),
    )
    return bool(cursor.rowcount)


def set_metadata(connection: Any, key: str, value: str = "1") -> None:
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
