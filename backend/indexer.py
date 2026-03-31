"""
indexer.py
Indexes (or re-indexes) all documents in the documents/ folder into the
SQLite vector store. Can also watch the folder for new files and auto-index
them in the background.
"""
import logging
import os
import time
from pathlib import Path
from threading import Lock, Thread

from config import settings
from document_loader import SUPPORTED_EXTENSIONS, load_and_split
from vector_store import (
    add_documents,
    collection_count,
    delete_by_filename,
    get_vector_store,
)

logger = logging.getLogger(__name__)

WATCHED_EXTENSIONS = set(SUPPORTED_EXTENSIONS)

_index_locks = set()
_index_lock_mutex = Lock()
_status_lock = Lock()
_file_statuses = {}

# Watchdog is imported lazily inside start_file_watcher() to avoid hanging at
# import time inside frozen PyInstaller executables where the native OS
# filesystem back-end may not be available.
_observer = None


def _build_status(path: str | None, status: str, error: str | None = None, **extra):
    filepath = Path(path).resolve() if path else None
    payload = {
        "status": status,
        "error": error,
        "updated_at": time.time(),
    }

    if filepath:
        payload.update(
            {
                "path": str(filepath),
                "filename": filepath.name,
                "type": filepath.suffix.lstrip(".").lower(),
                "size_bytes": filepath.stat().st_size if filepath.exists() else 0,
            }
        )

    payload.update(extra)
    return payload


def set_file_status(
    filename: str,
    status: str,
    path: str | None = None,
    error: str | None = None,
    **extra,
) -> None:
    """Store the latest indexing state for a document."""
    with _status_lock:
        current = dict(_file_statuses.get(filename, {}))
        updated = {
            **current,
            **_build_status(path or current.get("path"), status, error=error, **extra),
        }
        updated["filename"] = filename
        updated.setdefault("type", Path(filename).suffix.lstrip(".").lower())
        updated.setdefault("size_bytes", 0)
        _file_statuses[filename] = updated


def register_uploaded_file(filepath: str) -> None:
    filepath_obj = Path(filepath).resolve()
    set_file_status(filepath_obj.name, "queued", path=str(filepath_obj))


def clear_file_status(filename: str) -> None:
    with _status_lock:
        _file_statuses.pop(filename, None)


def get_file_statuses() -> dict:
    with _status_lock:
        return {name: dict(data) for name, data in _file_statuses.items()}


def schedule_index_file(filepath: str, status: str = "queued") -> None:
    """Queue a file for background indexing."""
    filepath_obj = Path(filepath).resolve()
    set_file_status(filepath_obj.name, status, path=str(filepath_obj))
    Thread(
        target=_run_index_single_file,
        args=(str(filepath_obj),),
        daemon=True,
        name=f"index-{filepath_obj.name}",
    ).start()


def _run_index_single_file(filepath: str) -> None:
    try:
        _index_single_file(filepath)
    except Exception:
        logger.exception("Background indexing failed for %s", filepath)


def _wait_for_file_stable(filepath: Path, timeout_seconds: float = 180.0, interval: float = 1.0) -> None:
    """
    Wait until a file stops growing. This helps large PDFs and copied files
    finish writing before loaders try to read them.
    """
    deadline = time.time() + timeout_seconds
    last_size = None
    stable_reads = 0

    while time.time() < deadline:
        if not filepath.exists():
            time.sleep(interval)
            continue

        try:
            current_size = filepath.stat().st_size
        except OSError:
            time.sleep(interval)
            continue

        if current_size == last_size:
            stable_reads += 1
        else:
            stable_reads = 0
            last_size = current_size

        if stable_reads >= 2:
            return

        time.sleep(interval)


def index_all_documents(reset: bool = False) -> int:
    """
    Load and index every document in the documents folder.
    If reset=True, wipe the collection first.
    Returns the number of chunks indexed.
    """
    if reset:
        logger.info("Resetting collection before indexing...")
        get_vector_store(reset=True)

    chunks = load_and_split(settings.DOCUMENTS_DIR)
    if not chunks:
        logger.warning(
            "No documents found in '%s'. Place supported files there and re-index.",
            settings.DOCUMENTS_DIR,
        )
        return 0

    filenames = sorted(
        {
            doc.metadata.get("filename")
            for doc in chunks
            if doc.metadata.get("filename")
        }
    )

    if not reset:
        for filename in filenames:
            delete_by_filename(filename)

    added = add_documents(chunks)
    for filename in filenames:
        file_path = Path(settings.DOCUMENTS_DIR) / filename
        set_file_status(filename, "indexed", path=str(file_path))

    logger.info("Indexed %d chunk(s).", added)
    return added


class _DocumentEventHandler:
    """Fallback stub - replaced with a real watchdog handler at runtime."""


def _make_event_handler():
    """Create a watchdog FileSystemEventHandler dynamically after lazy import."""
    try:
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        return None

    class _Handler(FileSystemEventHandler):
        def _should_handle(self, path: str) -> bool:
            return Path(path).suffix.lower() in WATCHED_EXTENSIONS

        def on_created(self, event):
            if not event.is_directory and self._should_handle(event.src_path):
                logger.info("New file detected: %s - queueing indexing.", event.src_path)
                schedule_index_file(event.src_path)

        def on_modified(self, event):
            if not event.is_directory and self._should_handle(event.src_path):
                logger.info("File modified: %s - queueing re-index.", event.src_path)
                schedule_index_file(event.src_path)

    return _Handler()


def _index_single_file(filepath: str) -> None:
    """Load and add a single file to the existing vector store."""
    from document_loader import LOADER_MAP, _sanitise, split_documents

    filepath_ptr = Path(filepath).resolve()
    abs_path = str(filepath_ptr)
    filename = filepath_ptr.name

    with _index_lock_mutex:
        if abs_path in _index_locks:
            logger.info("Already indexing %s, skipping duplicate request.", filepath_ptr)
            return
        _index_locks.add(abs_path)

    try:
        if not filepath_ptr.exists():
            raise FileNotFoundError(f"File not found: {filepath_ptr}")

        ext = filepath_ptr.suffix.lower()
        loader_cls = LOADER_MAP.get(ext)
        if not loader_cls:
            raise ValueError(f"No loader for file type: {ext}")

        _wait_for_file_stable(filepath_ptr)
        set_file_status(filename, "indexing", path=abs_path)

        logger.info("Indexing single file: %s", filepath_ptr)
        delete_by_filename(filename)

        loader = loader_cls(str(filepath_ptr))
        loaded = loader.load()

        for doc in loaded:
            doc.page_content = _sanitise(doc.page_content)
            doc.metadata.update(
                {
                    "source": abs_path,
                    "filename": filename,
                    "filetype": ext.lstrip("."),
                }
            )

        chunks = split_documents(loaded)
        if chunks:
            add_documents(chunks)
            set_file_status(
                filename,
                "indexed",
                path=abs_path,
                chunks_indexed=len(chunks),
                total_vectors=collection_count(),
            )
            logger.info("Successfully indexed %d chunk(s) from %s.", len(chunks), filepath_ptr)
        else:
            set_file_status(
                filename,
                "indexed",
                path=abs_path,
                chunks_indexed=0,
                total_vectors=collection_count(),
            )
            logger.warning("No chunks created from %s.", filepath_ptr)
    except Exception as exc:
        set_file_status(filename, "error", path=abs_path, error=str(exc))
        logger.error("Failed to index file '%s': %s", filepath_ptr, exc)
        raise
    finally:
        with _index_lock_mutex:
            _index_locks.discard(abs_path)


def start_file_watcher() -> None:
    """
    Start a background thread that watches documents/ for new/modified files.
    Uses PollingObserver as a universal fallback that works in frozen exes.
    """
    global _observer
    docs_dir = settings.DOCUMENTS_DIR
    os.makedirs(docs_dir, exist_ok=True)

    try:
        from watchdog.observers.polling import PollingObserver

        handler = _make_event_handler()
        if handler is None:
            logger.warning("watchdog not available - file watcher disabled.")
            return

        _observer = PollingObserver(timeout=2)
        _observer.schedule(handler, docs_dir, recursive=True)
        _observer.daemon = True
        _observer.start()
        logger.info("File watcher started (PollingObserver) - watching '%s'.", docs_dir)
    except Exception as exc:
        logger.warning("Could not start file watcher: %s. New files require manual re-index.", exc)
        _observer = None


def stop_file_watcher() -> None:
    global _observer
    if _observer:
        try:
            _observer.stop()
            _observer.join(timeout=5)
        except Exception as exc:
            logger.warning("Error stopping file watcher: %s", exc)
        _observer = None
        logger.info("File watcher stopped.")
