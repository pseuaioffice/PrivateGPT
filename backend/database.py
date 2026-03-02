"""
database.py
PostgreSQL connection management for the RAG backend.

Creates the connection pool on startup and auto-creates the
chat_sessions + chat_messages tables if they don't exist yet.

Falls back gracefully when PostgreSQL is not configured.
"""
import logging
import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

# Module-level connection pool (None if DB is not configured/reachable)
_pool: Optional[pg_pool.SimpleConnectionPool] = None


# ── Schema ──────────────────────────────────────────────────────────────────

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    chat_uuid   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       VARCHAR(300),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    chat_uuid   UUID NOT NULL REFERENCES chat_sessions(chat_uuid) ON DELETE CASCADE,
    role        VARCHAR(10) NOT NULL CHECK (role IN ('user', 'bot')),
    content     TEXT NOT NULL,
    sources     JSONB DEFAULT '[]',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_uuid
    ON chat_messages (chat_uuid, created_at);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated
    ON chat_sessions (updated_at DESC);
"""


# ── Public API ───────────────────────────────────────────────────────────────

def init_db() -> bool:
    """
    Initialise the connection pool and create tables.
    Returns True on success, False if DB is not configured or unreachable.
    """
    global _pool
    dsn = getattr(settings, "DATABASE_URL", None)
    if not dsn:
        logger.info("DATABASE_URL not set — chat history disabled.")
        return False
    try:
        _pool = pg_pool.SimpleConnectionPool(1, 10, dsn)
        conn = _pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLES)
            conn.commit()
        finally:
            _pool.putconn(conn)
        logger.info("PostgreSQL connected. Chat history enabled.")
        return True
    except Exception as e:
        logger.warning("PostgreSQL unavailable (%s) — chat history disabled.", e)
        _pool = None
        return False


def is_available() -> bool:
    return _pool is not None


def get_conn():
    """Get a connection from the pool. Caller must call putconn() when done."""
    if _pool is None:
        raise RuntimeError("Database not initialised")
    return _pool.getconn()


def put_conn(conn) -> None:
    if _pool is not None:
        _pool.putconn(conn)


def close_pool() -> None:
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        logger.info("PostgreSQL pool closed.")
