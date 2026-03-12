"""
main.py
FastAPI application — RAG Engine (Qwen2.5-7B)

Endpoints:
  GET  /health                   — liveness check
  GET  /status                   — vector store stats
  POST /chat                     — ask a question (RAG)
  POST /index                    — (re-)index the documents folder
  POST /upload                   — upload a document file
  DELETE /index                  — clear the entire collection
"""
# ---------------------------------------------------------------------------
# Suppress known harmless third-party warnings BEFORE any other imports
# ---------------------------------------------------------------------------
import warnings

# langchain_core imports pydantic.v1 shim internally — this is a warning only;
# our code does NOT use pydantic.v1 so functionality is unaffected.
warnings.filterwarnings(
    "ignore",
    message=".*Core Pydantic V1 functionality.*",
    category=UserWarning,
)

# requests/urllib3 version mismatch — cosmetic only, requests works fine.
warnings.filterwarnings(
    "ignore",
    message=".*urllib3.*chardet.*charset_normalizer.*",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message=".*doesn't match a supported version.*",
    category=Warning,
)

import logging
import os
import shutil
import json
import requests
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings
from indexer import index_all_documents, start_file_watcher, stop_file_watcher
from rag_chain import ask
from vector_store import collection_count, get_vector_store, clear_collection, delete_by_filename
from model_manager import model_manager, ModelStatus
import database as db
import chat_history as ch

# ---------------------------------------------------------------------------
# WebSocket Manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Smart startup indexer — only indexes files not yet in the vector store
# ---------------------------------------------------------------------------
def _index_new_files_on_startup() -> None:
    """
    Compare files in the documents folder against what's already indexed.
    Index only the files that are missing from the vector store.
    Existing vectors are preserved — no re-embedding of already-indexed files.
    """
    from pathlib import Path
    import json
    from vector_store import _get_conn, collection_count
    from document_loader import load_documents_from_folder, split_documents
    from vector_store import add_documents

    supported = {".pdf", ".docx", ".txt", ".md", ".csv"}
    docs_dir = Path(settings.DOCUMENTS_DIR)

    # Gather files on disk
    disk_files = {
        f.name for f in docs_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in supported
    }

    if not disk_files:
        logger.info("No documents found in '%s'.", settings.DOCUMENTS_DIR)
        return

    # Gather files already in the vector store
    indexed_files: set = set()
    if collection_count() > 0:
        try:
            conn = _get_conn()
            rows = conn.execute("SELECT metadata FROM documents").fetchall()
            conn.close()
            for (meta_json,) in rows:
                m = json.loads(meta_json)
                fn = m.get("filename", "")
                if fn:
                    indexed_files.add(fn)
        except Exception as e:
            logger.warning("Could not read indexed files list: %s", e)

    new_files = disk_files - indexed_files

    if not new_files:
        logger.info(
            "All %d document(s) already indexed (%d vectors). Nothing to do.",
            len(disk_files), collection_count(),
        )
        return

    logger.info(
        "Found %d new file(s) not yet indexed: %s",
        len(new_files), new_files,
    )

    # Load only the new files
    all_docs = load_documents_from_folder(settings.DOCUMENTS_DIR)
    new_docs = [d for d in all_docs if d.metadata.get("filename") in new_files]

    if new_docs:
        chunks = split_documents(new_docs)
        try:
            added = add_documents(chunks)
            logger.info(
                "Startup indexing complete: %d chunk(s) indexed from %d new file(s). "
                "Total vectors: %d",
                added, len(new_files), collection_count(),
            )
        except Exception as exc:
            # Propagate so the lifespan handler can log a clean warning
            # instead of a full traceback crashing the server.
            raise RuntimeError(
                f"Failed to embed {len(new_files)} new file(s): {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== RAG backend starting up ===")
    settings.validate()

    # Ensure the documents folder exists
    os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)

    # Initialise PostgreSQL (optional — graceful no-op if not configured)
    db.init_db()

    # Load persistent model configuration
    try:
        active_models = model_manager.get_active_models()
        settings.MODEL_PROVIDER = active_models["provider"]
        settings.CHAT_MODEL_LOCAL = active_models["chat_model"]
        settings.EMBEDDING_MODEL_LOCAL = active_models["embedding_model"]
        logger.info(
            "Loaded model config: provider=%s, chat=%s, embedding=%s",
            settings.MODEL_PROVIDER,
            settings.CHAT_MODEL_LOCAL,
            settings.EMBEDDING_MODEL_LOCAL
        )
    except Exception as e:
        logger.warning("Could not load persistent model config: %s", e)

    # Smart startup indexing: find which files are NOT yet indexed and index them.
    # Wrapped in try/except — if the embedding API is rate-limited or unavailable
    # the server still starts and already-indexed documents remain accessible.
    try:
        _index_new_files_on_startup()
    except Exception as exc:
        logger.warning(
            "Startup indexing skipped — embedding API error: %s. "
            "Already-indexed documents are still available. "
            "New files will be indexed when the rate limit resets.",
            exc,
        )

    # Start hot-reload watcher (handles files added while server is running)
    start_file_watcher()

    yield  # app runs here

    stop_file_watcher()
    db.close_pool()
    logger.info("=== RAG backend shut down ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Chatbot API",
    description="LangChain + ChromaDB + Hugging Face Models RAG backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    k: int = 15
    chat_uuid: str | None = None   # optional — if omitted, history is not saved


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    suggestions: List[str] = []


class IndexResponse(BaseModel):
    message: str
    chunks_indexed: int
    total_vectors: int


class StatusResponse(BaseModel):
    total_vectors: int
    documents_folder: str
    chat_model: str
    embedding_model: str
    model_provider: str
    ollama_model: str
    openai_key_set: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["System"])
async def root():
    return {
        "message": "PrivateGPT Backend API is running!",
        "docs": "/docs",
        "health": "/health",
        "status": "/status"
    }


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def status():
    return StatusResponse(
        total_vectors=collection_count(),
        documents_folder=str(Path(settings.DOCUMENTS_DIR).resolve()),
        chat_model=settings.CHAT_MODEL if settings.MODEL_PROVIDER == "huggingface" else settings.CHAT_MODEL_LOCAL,
        embedding_model=settings.EMBEDDING_MODEL if settings.MODEL_PROVIDER == "huggingface" else settings.EMBEDDING_MODEL_LOCAL,
        model_provider=settings.MODEL_PROVIDER,
        ollama_model=settings.CHAT_MODEL_LOCAL,
        openai_key_set=bool(settings.OPENAI_API_KEY)
    )


class ModelSettings(BaseModel):
    provider: str  # 'huggingface' or 'ollama'
    model_name: str | None = None


@app.patch("/settings/model", tags=["System"])
async def update_model_settings(payload: ModelSettings):
    """Update the active AI model provider and model name with persistence."""
    if payload.provider not in ["huggingface", "ollama"]:
        raise HTTPException(status_code=400, detail="Invalid provider. Use 'huggingface' or 'ollama'.")
    
    settings.MODEL_PROVIDER = payload.provider
    model_manager.set_config("provider", payload.provider)
    
    if payload.model_name:
        if payload.provider == "ollama":
            settings.CHAT_MODEL_LOCAL = payload.model_name
            model_manager.set_config("chat_model", payload.model_name)
        else:
            settings.CHAT_MODEL = payload.model_name
            # Note: HuggingFace models are not persisted as they're cloud-based
            
    logger.info("Model settings updated: Provider=%s, Model=%s", settings.MODEL_PROVIDER, payload.model_name)
    return {
        "message": "Model settings updated and persisted.",
        "provider": settings.MODEL_PROVIDER,
        "chat_model": settings.CHAT_MODEL if settings.MODEL_PROVIDER == "huggingface" else settings.CHAT_MODEL_LOCAL
    }


@app.websocket("/ws/ollama")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


class OllamaPullRequest(BaseModel):
    model_name: str

@app.post("/ollama/pull", tags=["System"])
async def pull_ollama_model(request: OllamaPullRequest, background_tasks: BackgroundTasks):
    """Trigger Ollama to pull a model and stream progress via WebSockets."""
    async def _pull_stream():
        try:
            url = f"{settings.OLLAMA_BASE_URL}/api/pull"
            logger.info("Starting Ollama pull for: %s", request.model_name)
            
            # Use requests with stream=True to get progress
            with requests.post(url, json={"name": request.model_name, "stream": True}, stream=True, timeout=None) as res:
                res.raise_for_status()
                for line in res.iter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)
                        
                        progress = 0
                        if total > 0:
                            progress = int((completed / total) * 100)
                        
                        await manager.broadcast({
                            "model": request.model_name,
                            "status": status,
                            "progress": progress
                        })
                        
            logger.info("Successfully finished pull task for: %s", request.model_name)
        except Exception as e:
            logger.error("Failed override pull model %s: %s", request.model_name, e)
            await manager.broadcast({
                "model": request.model_name,
                "status": "error",
                "message": str(e)
            })

    background_tasks.add_task(_pull_stream)
    return {"message": f"Started pull for {request.model_name}."}


@app.get("/ollama/models", tags=["System"])
@app.get("/ollama/tags", tags=["System"])
async def list_ollama_models():
    """List models available in the local Ollama instance with enhanced metadata."""
    try:
        models = model_manager.list_local_models()
        return {
            "models": [
                {
                    "name": m.name,
                    "size": m.size,
                    "modified_at": m.modified_at,
                    "digest": m.digest,
                    "status": m.status
                }
                for m in models
            ],
            "active_models": model_manager.get_active_models()
        }
    except Exception as e:
        logger.error("Failed to list Ollama models: %s", e)
        return {"models": [], "active_models": model_manager.get_active_models()}


class ModelCheckRequest(BaseModel):
    model_name: str


@app.post("/ollama/check", tags=["System"])
async def check_model_status(request: ModelCheckRequest):
    """Check if a specific model is installed in Ollama."""
    try:
        exists = model_manager.check_model_exists(request.model_name)
        return {
            "model": request.model_name,
            "installed": exists,
            "status": ModelStatus.INSTALLED.value if exists else ModelStatus.NOT_INSTALLED.value
        }
    except Exception as e:
        logger.error("Failed to check model status: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ollama/progress/{model_name}", tags=["System"])
async def get_download_progress(model_name: str):
    """Get current download progress for a model."""
    progress = model_manager.get_download_progress(model_name)
    if progress:
        return {
            "model": model_name,
            "status": progress.get("status", "unknown"),
            "progress": progress.get("progress", 0),
            "error": progress.get("error")
        }
    return {"model": model_name, "status": "unknown", "progress": 0}


@app.get("/ollama/pull-stream", tags=["System"])
async def pull_model_stream(model_name: str):
    """
    Stream model download progress using Server-Sent Events (SSE).
    This provides real-time progress updates for the UI.
    """
    async def event_generator():
        try:
            for update in model_manager.pull_model(model_name):
                yield f"data: {json.dumps(update)}\n\n"
            yield f"data: {json.dumps({'model': model_name, 'status': 'success', 'progress': 100})}\n\n"
        except Exception as e:
            logger.error("SSE pull error: %s", e)
            yield f"data: {json.dumps({'model': model_name, 'status': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.patch("/settings/ollama", tags=["System"])
async def update_ollama_settings(chat_model: str | None = None, embedding_model: str | None = None):
    """Specifically update Ollama chat and embedding models with persistence."""
    if chat_model:
        settings.CHAT_MODEL_LOCAL = chat_model
        model_manager.set_config("chat_model", chat_model)
    if embedding_model:
        settings.EMBEDDING_MODEL_LOCAL = embedding_model
        model_manager.set_config("embedding_model", embedding_model)
    
    # Ensure provider is set to ollama
    settings.MODEL_PROVIDER = "ollama"
    model_manager.set_config("provider", "ollama")
    
    return {
        "chat_model": settings.CHAT_MODEL_LOCAL,
        "embedding_model": settings.EMBEDDING_MODEL_LOCAL,
        "provider": settings.MODEL_PROVIDER
    }


@app.post("/chat", response_model=ChatResponse, tags=["RAG"])
async def chat(request: ChatRequest):
    if collection_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload documents and call POST /index first.",
        )
    try:
        result = ask(request.question, k=request.k)
        answer = result["answer"]
        sources = result["sources"]

        # ── Persist to PostgreSQL if chat_uuid provided ──────────────
        cid = request.chat_uuid
        if cid and db.is_available():
            # Ensure session row exists (title = first user message)
            ch.create_session(cid, ch.make_title(request.question))
            ch.save_message(cid, "user", request.question)
            ch.save_message(cid, "bot", answer, sources)
            ch.touch_session(cid)

        return ChatResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            suggestions=result.get("suggestions", []),
        )
    except Exception as exc:
        logger.exception("Error during RAG chain execution")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/index", response_model=IndexResponse, tags=["Indexing"])
async def reindex(reset: bool = False, background_tasks: BackgroundTasks = None):
    """
    Trigger (re-)indexing of the documents folder.
    Pass ?reset=true to wipe the collection before indexing.
    """
    try:
        chunks = index_all_documents(reset=reset)
        total = collection_count()
        return IndexResponse(
            message="Indexing complete.",
            chunks_indexed=chunks,
            total_vectors=total,
        )
    except Exception as exc:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/index", tags=["Indexing"])
async def clear_index():
    """Wipe the entire vector collection."""
    try:
        clear_collection()
        return {"message": "Collection cleared.", "total_vectors": 0}
    except Exception as exc:
        logger.exception("Failed to clear collection")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/upload", tags=["Indexing"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document file (.pdf, .docx, .txt, .md, .csv) to the
    documents folder. The file watcher will auto-index it; you can also
    call POST /index manually.
    """
    allowed = {".pdf", ".docx", ".txt", ".md", ".csv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )

    dest = Path(settings.DOCUMENTS_DIR) / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("Uploaded: %s", dest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    finally:
        file.file.close()

    # Immediately index the uploaded file
    from indexer import _index_single_file
    _index_single_file(str(dest))

    return {
        "message": f"File '{file.filename}' uploaded and indexed.",
        "path": str(dest),
        "total_vectors": collection_count(),
    }


@app.get("/documents", tags=["Indexing"])
async def list_documents():
    """List all files currently in the documents folder."""
    folder = Path(settings.DOCUMENTS_DIR)
    if not folder.exists():
        return {"documents": []}
    files = [
        {"filename": f.name, "size_bytes": f.stat().st_size, "type": f.suffix.lstrip(".")}
        for f in folder.rglob("*")
        if f.is_file()
    ]
    return {"documents": files, "count": len(files)}


# ---------------------------------------------------------------------------
@app.delete("/documents/{filename}", tags=["Index"])
async def delete_document(filename: str):
    """Delete a document file and its associated vectors."""
    doc_path = Path(settings.DOCUMENTS_DIR) / filename
    
    # Check if file exists
    if not doc_path.exists():
        # Even if file is gone, try to clear vectors (maybe it was deleted manually)
        vectors_deleted = delete_by_filename(filename)
        if vectors_deleted > 0:
            return {"message": f"Associated {vectors_deleted} vectors deleted for {filename}."}
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        # 1. Delete the physical file
        os.remove(doc_path)
        # 2. Delete the vectors from the store
        vectors_deleted = delete_by_filename(filename)
        return {
            "message": f"Successfully deleted {filename} and {vectors_deleted} vector(s)."
        }
    except Exception as e:
        logger.error("Error deleting document %s: %s", filename, e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Chat History endpoints (require PostgreSQL)
# ---------------------------------------------------------------------------

@app.get("/chats", tags=["Chat History"])
async def list_chats():
    """Return all chat sessions ordered by most-recent first."""
    if not db.is_available():
        return {"db_enabled": False, "sessions": []}
    sessions = ch.list_sessions()
    # Serialise datetimes
    for s in sessions:
        for k, v in s.items():
            s[k] = ch.serialize_dt(v)
    return {"db_enabled": True, "sessions": sessions}


@app.post("/chats", tags=["Chat History"])
async def create_chat(payload: dict = None):
    """Create a new chat session. Accepts optional {chat_uuid, title}."""
    import uuid as _uuid
    p = payload or {}
    cid = p.get("chat_uuid") or str(_uuid.uuid4())
    title = p.get("title")
    if not db.is_available():
        return {"db_enabled": False, "chat_uuid": cid}
    session = ch.create_session(cid, title)
    if session:
        for k, v in session.items():
            session[k] = ch.serialize_dt(v)
    return {"db_enabled": True, "chat_uuid": cid, "session": session}


@app.get("/chats/{chat_uuid}/messages", tags=["Chat History"])
async def get_chat_messages(chat_uuid: str):
    """Return all messages for a given chat session."""
    if not db.is_available():
        return {"db_enabled": False, "messages": []}
    msgs = ch.get_messages(chat_uuid)
    for m in msgs:
        for k, v in m.items():
            m[k] = ch.serialize_dt(v)
    return {"db_enabled": True, "chat_uuid": chat_uuid, "messages": msgs}


@app.delete("/chats/{chat_uuid}", tags=["Chat History"])
async def delete_chat(chat_uuid: str):
    """Delete a chat session and all its messages."""
    if not db.is_available():
        raise HTTPException(status_code=503, detail="Database not available")
    deleted = ch.delete_session(chat_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": f"Chat {chat_uuid} deleted."}


@app.get("/db-status", tags=["System"])
async def db_status():
    return {"db_enabled": db.is_available()}
