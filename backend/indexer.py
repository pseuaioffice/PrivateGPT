"""
indexer.py
Indexes (or re-indexes) all documents in the documents/ folder into ChromaDB.
Can also watch the folder for new files and auto-index them.
"""
import logging
import os
import time
from pathlib import Path
from threading import Thread

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

from config import settings
from document_loader import load_and_split
from vector_store import add_documents, get_vector_store, collection_count

logger = logging.getLogger(__name__)

# Extensions that trigger re-indexing when changed
WATCHED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


def index_all_documents(reset: bool = False) -> int:
    """
    Load and index every document in the documents folder.
    If reset=True, wipe the collection first.
    Returns the number of chunks indexed.
    """
    if reset:
        logger.info("Resetting collection before indexing …")
        get_vector_store(reset=True)

    chunks = load_and_split(settings.DOCUMENTS_DIR)
    if not chunks:
        logger.warning(
            "No documents found in '%s'. "
            "Place PDF / DOCX / TXT files there and re-index.",
            settings.DOCUMENTS_DIR,
        )
        return 0

    added = add_documents(chunks)
    logger.info("Indexed %d chunk(s).", added)
    return added


# ---------------------------------------------------------------------------
# Watchdog — hot-reload when a new file is dropped into documents/
# ---------------------------------------------------------------------------

class _DocumentEventHandler(FileSystemEventHandler):
    def _should_handle(self, path: str) -> bool:
        return Path(path).suffix.lower() in WATCHED_EXTENSIONS

    def on_created(self, event: FileCreatedEvent):  # type: ignore[override]
        if not event.is_directory and self._should_handle(event.src_path):
            logger.info("New file detected: %s — indexing …", event.src_path)
            _index_single_file(event.src_path)

    def on_modified(self, event: FileModifiedEvent):  # type: ignore[override]
        if not event.is_directory and self._should_handle(event.src_path):
            logger.info("File modified: %s — re-indexing …", event.src_path)
            _index_single_file(event.src_path)


def _index_single_file(filepath: str) -> None:
    """Load and add a single file to the existing vector store."""
    from document_loader import load_documents_from_folder, split_documents
    from pathlib import Path

    # Give the OS a moment to finish writing the file
    time.sleep(1)
    try:
        folder = str(Path(filepath).parent)
        docs = load_documents_from_folder(folder)
        # Filter to only the changed file
        target = str(Path(filepath).resolve())
        docs = [d for d in docs if str(Path(d.metadata.get("source", "")).resolve()) == target]
        chunks = split_documents(docs)
        if chunks:
            add_documents(chunks)
            logger.info("Hot-indexed %d chunk(s) from %s.", len(chunks), filepath)
    except Exception as exc:
        logger.error("Hot-index failed for '%s': %s", filepath, exc)


_observer: Observer | None = None


def start_file_watcher() -> None:
    """Start a background thread that watches documents/ for new/modified files."""
    global _observer
    docs_dir = settings.DOCUMENTS_DIR
    os.makedirs(docs_dir, exist_ok=True)

    handler = _DocumentEventHandler()
    _observer = Observer()
    _observer.schedule(handler, docs_dir, recursive=True)
    t = Thread(target=_observer.start, daemon=True)
    t.start()
    logger.info("File watcher started — watching '%s'.", docs_dir)


def stop_file_watcher() -> None:
    global _observer
    if _observer:
        _observer.stop()
        _observer.join()
        _observer = None
        logger.info("File watcher stopped.")
