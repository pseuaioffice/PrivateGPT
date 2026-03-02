"""
vector_store.py
Pure-Python vector store built on SQLite + JSON — zero Pydantic dependency.
100% compatible with Python 3.14+.

Stores:
  • document text
  • metadata (JSON)
  • embedding vectors (JSON list of floats)

Similarity search uses cosine similarity computed in plain Python.
"""
import json
import logging
import math
import os
import sqlite3
import uuid
from typing import List, Optional

from langchain_core.documents import Document

from config import settings
from embeddings import HuggingFaceServerlessEmbeddings

logger = logging.getLogger(__name__)

_embed_fn: Optional[HuggingFaceServerlessEmbeddings] = None
_db_path: Optional[str] = None


# ── Embedding helper ────────────────────────────────────────────────────────

def _get_embed_fn() -> HuggingFaceServerlessEmbeddings:
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = HuggingFaceServerlessEmbeddings()
    return _embed_fn


# ── SQLite database helpers ─────────────────────────────────────────────────

def _get_db_path() -> str:
    global _db_path
    if _db_path is None:
        persist_dir = settings.CHROMA_PERSIST_DIR
        os.makedirs(persist_dir, exist_ok=True)
        _db_path = os.path.join(persist_dir, "vectors.db")
    return _db_path


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path())
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id       TEXT PRIMARY KEY,
            text     TEXT NOT NULL,
            metadata TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


# ── Math helpers ────────────────────────────────────────────────────────────

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine(a: List[float], b: List[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


# ── Public API ──────────────────────────────────────────────────────────────

class ChromaVectorStore:
    """Thin wrapper that mimics the langchain VectorStore interface."""

    def as_retriever(self, search_type="similarity", search_kwargs=None):
        k = (search_kwargs or {}).get("k", 5)
        return _SQLiteRetriever(k=k)


class _SQLiteRetriever:
    def __init__(self, k: int = 5):
        self.k = k

    def invoke(self, query: str) -> List[Document]:
        return similarity_search(query, k=self.k)


_store_instance: Optional[ChromaVectorStore] = None


def get_vector_store(reset: bool = False) -> ChromaVectorStore:
    global _store_instance
    if reset:
        clear_collection()
    if _store_instance is None or reset:
        _store_instance = ChromaVectorStore()
    return _store_instance


def add_documents(docs: List[Document]) -> int:
    """Embed and persist documents to the SQLite vector store."""
    if not docs:
        logger.warning("add_documents called with empty list — skipping.")
        return 0

    # Sanitise: remove surrogate / non-UTF-8 characters that break json.dumps
    def _clean(t: str) -> str:
        return t.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    texts = [_clean(d.page_content) for d in docs]
    metas = [d.metadata or {} for d in docs]

    logger.info("Embedding %d chunk(s)…", len(texts))
    embeddings = _get_embed_fn().embed_documents(texts)

    conn = _get_conn()
    with conn:
        for text, meta, emb in zip(texts, metas, embeddings):
            safe_meta = {k: str(v) for k, v in meta.items()}
            conn.execute(
                "INSERT INTO documents (id, text, metadata, embedding) VALUES (?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    text,
                    json.dumps(safe_meta),
                    json.dumps(emb),
                ),
            )
    conn.close()

    total = collection_count()
    logger.info("Added %d chunk(s). Store now has %d vector(s).", len(docs), total)
    return len(docs)


def similarity_search(query: str, k: int = 5) -> List[Document]:
    """Return the top-k most relevant chunks by cosine similarity."""
    if collection_count() == 0:
        return []

    query_emb = _get_embed_fn().embed_query(query)

    conn = _get_conn()
    rows = conn.execute("SELECT text, metadata, embedding FROM documents").fetchall()
    conn.close()

    scored = []
    for text, meta_json, emb_json in rows:
        emb = json.loads(emb_json)
        score = _cosine(query_emb, emb)
        scored.append((score, text, json.loads(meta_json)))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    docs = [Document(page_content=text, metadata=meta) for _, text, meta in top]
    logger.info("Similarity search returned %d result(s).", len(docs))
    return docs


def collection_count() -> int:
    """Return total number of stored vectors."""
    try:
        conn = _get_conn()
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def clear_collection() -> None:
    """Delete all stored vectors."""
    try:
        conn = _get_conn()
        with conn:
            conn.execute("DELETE FROM documents")
        conn.close()
        logger.info("Vector store cleared.")
    except Exception as e:
        logger.error("Failed to clear vector store: %s", e)
