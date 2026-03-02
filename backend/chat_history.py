"""
chat_history.py
CRUD helpers for chat sessions and messages stored in PostgreSQL.
All functions are no-ops when the database is not available.
"""
import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any

import psycopg2.extras

import database as db

logger = logging.getLogger(__name__)


# ── Sessions ─────────────────────────────────────────────────────────────────

def create_session(chat_uuid: str, title: str | None = None) -> Optional[Dict]:
    """Create a new chat session row. Returns the session dict or None."""
    if not db.is_available():
        return None
    conn = db.get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO chat_sessions (chat_uuid, title)
                VALUES (%s, %s)
                ON CONFLICT (chat_uuid) DO NOTHING
                RETURNING chat_uuid::text, title, created_at, updated_at
                """,
                (chat_uuid, title),
            )
            row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    except Exception as e:
        conn.rollback()
        logger.error("create_session error: %s", e)
        return None
    finally:
        db.put_conn(conn)


def list_sessions(limit: int = 50) -> List[Dict]:
    """Return the most-recent chat sessions, ordered by updated_at DESC."""
    if not db.is_available():
        return []
    conn = db.get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    chat_uuid::text,
                    COALESCE(title, 'Untitled Chat') AS title,
                    created_at,
                    updated_at
                FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error("list_sessions error: %s", e)
        return []
    finally:
        db.put_conn(conn)


def touch_session(chat_uuid: str, title: str | None = None) -> None:
    """Update updated_at (and optionally title) for session."""
    if not db.is_available():
        return
    conn = db.get_conn()
    try:
        with conn.cursor() as cur:
            if title:
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET updated_at = NOW(), title = COALESCE(title, %s)
                    WHERE chat_uuid = %s
                    """,
                    (title, chat_uuid),
                )
            else:
                cur.execute(
                    "UPDATE chat_sessions SET updated_at = NOW() WHERE chat_uuid = %s",
                    (chat_uuid,),
                )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("touch_session error: %s", e)
    finally:
        db.put_conn(conn)


def delete_session(chat_uuid: str) -> bool:
    """Delete a session and all its messages (CASCADE)."""
    if not db.is_available():
        return False
    conn = db.get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_sessions WHERE chat_uuid = %s", (chat_uuid,))
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    except Exception as e:
        conn.rollback()
        logger.error("delete_session error: %s", e)
        return False
    finally:
        db.put_conn(conn)


# ── Messages ─────────────────────────────────────────────────────────────────

def save_message(
    chat_uuid: str,
    role: str,
    content: str,
    sources: List[Dict] | None = None,
) -> Optional[Dict]:
    """Persist a single message. Returns saved row or None."""
    if not db.is_available():
        return None
    conn = db.get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (chat_uuid, role, content, sources)
                VALUES (%s, %s, %s, %s)
                RETURNING id, chat_uuid::text, role, content, sources, created_at
                """,
                (chat_uuid, role, content, json.dumps(sources or [])),
            )
            row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    except Exception as e:
        conn.rollback()
        logger.error("save_message error: %s", e)
        return None
    finally:
        db.put_conn(conn)


def get_messages(chat_uuid: str) -> List[Dict]:
    """Return all messages for a session ordered by created_at ASC."""
    if not db.is_available():
        return []
    conn = db.get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, chat_uuid::text, role, content, sources, created_at
                FROM chat_messages
                WHERE chat_uuid = %s
                ORDER BY created_at ASC
                """,
                (chat_uuid,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error("get_messages error: %s", e)
        return []
    finally:
        db.put_conn(conn)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_title(text: str, max_len: int = 60) -> str:
    """Derive a chat title from the first user message."""
    t = text.strip().replace("\n", " ")
    return t[:max_len] + ("…" if len(t) > max_len else "")


def serialize_dt(obj: Any) -> Any:
    """JSON-serialise datetime objects returned from psycopg2."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
