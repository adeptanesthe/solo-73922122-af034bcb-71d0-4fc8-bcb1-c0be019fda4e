"""SQLite storage layer.

One database file is shared by every API process (WAL mode, busy timeout).
All writes happen in short IMMEDIATE transactions, so background comparison
never blocks corpus submissions and vice versa.

Tables
------
meta                 single row: current_generation / published_generation
articles             committed corpus (id, side, text, origin, updated_generation);
                     origin is 'manual' (POST /articles/batch) or 'ingest'
                     (approved by dual-source verification)
ingest_reports       latest accepted report per (article_id, source A/B):
                     monotonic revision, content or retraction
edges / article_groups  published verification results, keyed by generation
compute_jobs         pending/running background verification jobs (lease based)
"""

import os
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    current_generation INTEGER NOT NULL,
    published_generation INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    side TEXT NOT NULL CHECK (side IN ('train', 'test')),
    text TEXT NOT NULL CHECK (length(text) BETWEEN 5 AND 2000),
    origin TEXT NOT NULL DEFAULT 'manual' CHECK (origin IN ('manual', 'ingest')),
    updated_generation INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_reports (
    article_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('A', 'B')),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    kind TEXT NOT NULL CHECK (kind IN ('content', 'retraction')),
    side TEXT CHECK (side IN ('train', 'test')),
    text TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (article_id, source)
);

CREATE TABLE IF NOT EXISTS edges (
    generation INTEGER NOT NULL,
    a TEXT NOT NULL,
    b TEXT NOT NULL,
    similarity REAL NOT NULL,
    PRIMARY KEY (generation, a, b)
);

CREATE TABLE IF NOT EXISTS article_groups (
    generation INTEGER NOT NULL,
    group_id INTEGER NOT NULL,
    article_id TEXT NOT NULL,
    side TEXT NOT NULL,
    PRIMARY KEY (generation, group_id, article_id)
);

CREATE TABLE IF NOT EXISTS compute_jobs (
    generation INTEGER PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running')),
    claimed_by TEXT,
    claimed_at REAL
);
"""

# Hard corpus limits (also enforced at the API boundary).
MAX_ARTICLES = 300
MIN_TEXT_LEN = 5
MAX_TEXT_LEN = 2000


def connect(db_path: str) -> sqlite3.Connection:
    """Open a connection in autocommit mode; transactions are explicit."""
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db(db_path: str) -> None:
    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        # Migration for databases created before the origin column existed:
        # every pre-existing article came through the manual batch API.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(articles)")}
        if "origin" not in columns:
            conn.execute(
                "ALTER TABLE articles ADD COLUMN origin TEXT NOT NULL"
                " DEFAULT 'manual' CHECK (origin IN ('manual', 'ingest'))"
            )
        conn.execute(
            "INSERT OR IGNORE INTO meta (id, current_generation, published_generation)"
            " VALUES (1, 0, 0)"
        )
        conn.commit()
    finally:
        conn.close()


def get_meta(conn: sqlite3.Connection):
    """Return ``(current_generation, published_generation)``."""
    row = conn.execute(
        "SELECT current_generation, published_generation FROM meta WHERE id = 1"
    ).fetchone()
    return row["current_generation"], row["published_generation"]
