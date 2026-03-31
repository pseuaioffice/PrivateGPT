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
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings
from document_loader import SUPPORTED_EXTENSIONS
from indexer import (
    clear_file_status,
    get_file_statuses,
    index_all_documents,
    register_uploaded_file,
    schedule_index_file,
    start_file_watcher,
    stop_file_watcher,
)
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
    supported = set(SUPPORTED_EXTENSIONS)
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
    from indexer import _index_single_file
    
    success_count = 0
    for fn in new_files:
        filepath = docs_dir / fn
        try:
            _index_single_file(str(filepath))
            success_count += 1
        except Exception as e:
            logger.error("Startup indexing failed for %s: %s", fn, e)

    if success_count > 0:
        logger.info(
            "Startup indexing complete: %d file(s) indexed. Total vectors: %d",
            success_count, collection_count()
        )


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    import traceback as _tb
    logger.info("=== RAG backend starting up ===")

    # Step 1: Validate settings
    try:
        settings.validate()
    except Exception as e:
        logger.warning("Settings validation warning: %s", e)

    # Step 2: Ensure the documents folder exists
    try:
        os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True)
    except Exception as e:
        logger.error("Could not create documents dir: %s\n%s", e, _tb.format_exc())

    # Step 3: Initialise PostgreSQL (optional — graceful no-op if not configured)
    try:
        db.init_db()
    except Exception as e:
        logger.warning("DB init failed (continuing without chat history): %s", e)

    # Step 4: Load persistent model configuration
    try:
        active_models = model_manager.get_active_models()
        settings.MODEL_PROVIDER = active_models.get("provider", "ollama")
        settings.CHAT_MODEL_LOCAL = active_models.get("chat_model", settings.CHAT_MODEL_LOCAL)
        settings.EMBEDDING_MODEL_LOCAL = active_models.get("embedding_model", settings.EMBEDDING_MODEL_LOCAL)
        logger.info(
            "Loaded model config: provider=%s, chat=%s, embedding=%s",
            settings.MODEL_PROVIDER,
            settings.CHAT_MODEL_LOCAL,
            settings.EMBEDDING_MODEL_LOCAL
        )
    except Exception as e:
        logger.warning("Could not load persistent model config (using defaults): %s\n%s", e, _tb.format_exc())


    # Step 5: Smart startup indexing — runs in background so server is immediately ready
    def _background_startup_index():
        try:
            _index_new_files_on_startup()
        except Exception as exc:
            logger.warning(
                "Background startup indexing failed — %s. "
                "Existing documents remain accessible.",
                exc,
            )

    t = threading.Thread(target=_background_startup_index, daemon=True,
                         name="startup-indexer")
    t.start()

    # Step 6: Start hot-reload watcher (handles files added while server is running)
    try:
        start_file_watcher()
    except Exception as e:
        logger.warning("File watcher could not start (continuing without auto-reload): %s", e)

    logger.info("=== RAG backend ready — listening for requests ===")
    yield  # app runs here

    # Shutdown
    try:
        stop_file_watcher()
    except Exception:
        pass
    try:
        db.close_pool()
    except Exception:
        pass
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
    document_name: str | None = None  # optional — search only in this document
    is_strict_mode: bool = False  # if True, strict RAG (only from selected document)


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    suggestions: List[str] = []
    no_context: bool = False
    is_generic: bool = False

class CancelRequest(BaseModel):
    chat_uuid: str


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
        "message": "MyAIAssistant Backend API is running!",
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
    
    async def _progress_broadcaster(update: dict):
        await manager.broadcast(update)

    def _start_pull():
        # model_manager.pull_model is a generator. We need to exhaust it.
        # We'll use the callback to broadcast via WebSocket.
        # Since this runs in a thread, we need to bridge to async broadcast.
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def _run():
            try:
                # We consume the generator here
                for update in model_manager.pull_model(request.model_name):
                    await manager.broadcast(update)
            except Exception as e:
                logger.error("Error in background pull: %s", e)
                await manager.broadcast({
                    "model": request.model_name,
                    "status": "error",
                    "message": str(e)
                })
        
        loop.run_until_complete(_run())
        loop.close()

    background_tasks.add_task(_start_pull)
    return {"message": f"Started pull for {request.model_name} in background."}


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


@app.post("/chat/cancel", tags=["RAG"])
async def cancel_chat(request: CancelRequest):
    from rag_chain import CANCELLED_CHATS
    if request.chat_uuid:
        CANCELLED_CHATS.add(request.chat_uuid)
    return {"message": "Cancellation registered."}

@app.post("/chat", response_model=ChatResponse, tags=["RAG"])
def chat(request: ChatRequest):
    # Allow chat even with no documents for generic/social questions.
    # The RAG chain will handle the no-documents case gracefully.
    from rag_chain import current_chat_uuid, CANCELLED_CHATS
    if request.chat_uuid:
        current_chat_uuid.set(request.chat_uuid)
        if request.chat_uuid in CANCELLED_CHATS:
            CANCELLED_CHATS.remove(request.chat_uuid)
            
    try:
        result = ask(request.question, k=request.k, document_name=request.document_name, is_strict_mode=request.is_strict_mode)
        answer = result["answer"]
        sources = result["sources"]

        # ── Persist to PostgreSQL if chat_uuid provided ──────────────
        cid = request.chat_uuid
        if cid and db.is_available() and not result.get("is_generic", False):
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
            no_context=result.get("no_context", False),
            is_generic=result.get("is_generic", False),
        )
    except Exception as exc:
        logger.exception("Error during RAG chain execution")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if request.chat_uuid and request.chat_uuid in CANCELLED_CHATS:
            CANCELLED_CHATS.remove(request.chat_uuid)


@app.post("/index", response_model=IndexResponse, tags=["Indexing"])
def reindex(reset: bool = False, background_tasks: BackgroundTasks = None):
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
def clear_index():
    """Wipe the entire vector collection."""
    try:
        clear_collection()
        return {"message": "Collection cleared.", "total_vectors": 0}
    except Exception as exc:
        logger.exception("Failed to clear collection")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/upload", tags=["Indexing"])
async def upload_document(request: Request):
    """
    Upload one or more document files into the documents folder and queue them
    for background indexing.
    """
    form = await request.form()
    uploads: list[UploadFile] = []

    for form_file in form.getlist("files"):
        if getattr(form_file, "filename", None) and hasattr(form_file, "file"):
            uploads.append(form_file)

    legacy_file = form.get("file")
    if getattr(legacy_file, "filename", None) and hasattr(legacy_file, "file"):
        uploads.append(legacy_file)

    if not uploads:
        raise HTTPException(status_code=400, detail="No file provided.")

    allowed = set(SUPPORTED_EXTENSIONS)
    docs_dir = Path(settings.DOCUMENTS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)

    accepted_files = []
    rejected_files = []

    for upload in uploads:
        safe_name = os.path.basename(upload.filename or "").strip()
        ext = Path(safe_name).suffix.lower()

        if not safe_name:
            rejected_files.append({"filename": upload.filename or "", "reason": "Empty filename."})
            try:
                upload.file.close()
            except Exception:
                pass
            continue

        if ext not in allowed:
            rejected_files.append(
                {
                    "filename": safe_name,
                    "reason": f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}",
                }
            )
            try:
                upload.file.close()
            except Exception:
                pass
            continue

        dest = docs_dir / safe_name

        try:
            with dest.open("wb") as destination_file:
                shutil.copyfileobj(upload.file, destination_file)

            register_uploaded_file(str(dest))
            schedule_index_file(str(dest))
            accepted_files.append(
                {
                    "filename": safe_name,
                    "path": str(dest),
                    "status": "queued",
                    "type": ext.lstrip("."),
                    "size_bytes": dest.stat().st_size,
                }
            )
            logger.info("Uploaded and queued: %s", dest)
        except Exception as exc:
            logger.error("Upload failed for %s: %s", safe_name, exc)
            rejected_files.append({"filename": safe_name, "reason": str(exc)})
        finally:
            try:
                upload.file.close()
            except Exception:
                pass

    if not accepted_files:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No files were uploaded.",
                "rejected_files": rejected_files,
            },
        )

    return {
        "message": f"Queued {len(accepted_files)} file(s) for indexing.",
        "accepted_files": accepted_files,
        "rejected_files": rejected_files,
        "total_vectors": collection_count(),
    }


@app.get("/documents", tags=["Indexing"])
async def list_documents():
    """List uploaded and indexed files, including current indexing state."""
    import sqlite3
    import json
    from vector_store import _get_db_path

    documents_by_name = {}
    docs_dir = Path(settings.DOCUMENTS_DIR)
    supported = set(SUPPORTED_EXTENSIONS)
    indexed_files = set()
    status_map = get_file_statuses()

    try:
        conn = sqlite3.connect(_get_db_path())
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT metadata FROM documents")
        metadatas = cursor.fetchall()

        for meta_tuple in metadatas:
            try:
                meta = json.loads(meta_tuple[0])
                if "filename" in meta:
                    indexed_files.add(meta["filename"])
            except (json.JSONDecodeError, KeyError):
                continue
        conn.close()
    except Exception as e:
        logger.error(f"Error listing indexed documents: {e}")

    if docs_dir.exists():
        for file_path in docs_dir.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in supported:
                continue

            existing_status = status_map.get(file_path.name, {})
            status = existing_status.get("status", "indexed" if file_path.name in indexed_files else "uploaded")
            documents_by_name[file_path.name] = {
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "type": file_path.suffix.lstrip("."),
                "status": status,
                "error": existing_status.get("error"),
                "updated_at": existing_status.get("updated_at"),
                "chunks_indexed": existing_status.get("chunks_indexed"),
            }

    for filename, state in status_map.items():
        existing = documents_by_name.get(filename, {})
        documents_by_name[filename] = {
            "filename": filename,
            "size_bytes": existing.get("size_bytes", state.get("size_bytes", 0)),
            "type": existing.get("type", state.get("type", Path(filename).suffix.lstrip("."))),
            "status": state.get("status", existing.get("status", "uploaded")),
            "error": state.get("error"),
            "updated_at": state.get("updated_at"),
            "chunks_indexed": state.get("chunks_indexed"),
        }

    for filename in indexed_files:
        if filename not in documents_by_name:
            documents_by_name[filename] = {
                "filename": filename,
                "size_bytes": 0,
                "type": Path(filename).suffix.lstrip(".") or "unknown",
                "status": "indexed_only",
                "error": None,
                "updated_at": None,
                "chunks_indexed": None,
            }

    files = [documents_by_name[name] for name in sorted(documents_by_name)]
    return {"documents": files, "count": len(files)}


# ---------------------------------------------------------------------------
@app.delete("/documents/{filename:path}", tags=["Indexing"])
async def delete_document(filename: str):
    """Delete a document file and its associated vectors."""
    # URL decode the filename (handles spaces and special characters)
    from urllib.parse import unquote
    filename = unquote(filename)
    filename = os.path.basename(filename)
    doc_path = Path(settings.DOCUMENTS_DIR) / filename
    
    logger.info("Attempting to delete document: %s at %s", filename, doc_path)
    
    # Always try to delete vectors first
    vectors_deleted = delete_by_filename(filename)
    clear_file_status(filename)
    
    # Check if file exists on disk
    file_existed = doc_path.exists()
    if file_existed:
        try:
            os.remove(doc_path)
            logger.info("Deleted file: %s", doc_path)
        except Exception as e:
            logger.error("Error deleting file %s: %s", doc_path, e)
    
    # Return appropriate response
    if vectors_deleted > 0 or file_existed:
        return {
            "message": f"Deleted {filename}",
            "file_deleted": file_existed,
            "vectors_deleted": vectors_deleted
        }
    
    # Nothing was deleted - document not found anywhere
    raise HTTPException(status_code=404, detail="Document not found.")


# ---------------------------------------------------------------------------
# Debug: Test document filtering
# ---------------------------------------------------------------------------
@app.get("/debug/filter-test", tags=["Debug"])
async def debug_filter_test(query: str, document_name: str | None = None):
    """Test endpoint to verify document filtering is working.
    
    Returns the chunks found for a query with optional document filtering.
    Useful for debugging why filtering might not be working.
    """
    from vector_store import similarity_search
    
    logger.info("DEBUG: FilterTest - Query: '%s', Document: %s", query, document_name or "All")
    
    docs, scores = similarity_search(query, k=10, document_name=document_name, include_scores=True)
    
    results = []
    for doc, score in zip(docs, scores):
        results.append({
            "filename": doc.metadata.get("filename", "unknown"),
            "relevance_score": round(score, 4),
            "content_preview": doc.page_content[:200],
            "metadata": doc.metadata
        })
    
    return {
        "query": query,
        "filter_document": document_name,
        "chunks_found": len(results),
        "threshold_strict": 0.55,
        "threshold_flexible": 0.35,
        "results": results,
        "note": "Scores >= 0.55 used in strict mode, >= 0.35 in flexible mode"
    }


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


# ---------------------------------------------------------------------------
# Dev entry point — allows: python main.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting MyAIAssistant backend on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info", reload=False)
