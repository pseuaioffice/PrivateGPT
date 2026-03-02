"""
document_loader.py
Scans the documents/ folder, loads every supported file, splits the text
into chunks and returns a list of LangChain Document objects.

Supported formats:
  - PDF  (.pdf)
  - Word (.docx)
  - Plain text (.txt, .md, .csv)
"""
import os
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

logger = logging.getLogger(__name__)

# Map file extensions → loader class
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": TextLoader,
}


def _sanitise(text: str) -> str:
    """
    Remove Unicode surrogate characters and any other code points that cannot
    be encoded as UTF-8. These sometimes appear when PDFs contain corrupt or
    scanned text.
    """
    # Encode to UTF-8 replacing surrogates, then decode back
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def load_documents_from_folder(folder: str | None = None) -> List[Document]:
    """
    Walk *folder* (defaults to settings.DOCUMENTS_DIR), load every
    supported file and return the raw Document list (not yet split).
    """
    folder = folder or settings.DOCUMENTS_DIR
    docs: List[Document] = []

    if not os.path.isdir(folder):
        logger.warning("Documents folder '%s' does not exist — creating it.", folder)
        os.makedirs(folder, exist_ok=True)
        return docs

    for root, _, files in os.walk(folder):
        for filename in files:
            filepath = Path(root) / filename
            ext = filepath.suffix.lower()
            loader_cls = LOADER_MAP.get(ext)
            if loader_cls is None:
                logger.debug("Skipping unsupported file: %s", filepath)
                continue
            try:
                logger.info("Loading: %s", filepath)
                loader = loader_cls(str(filepath))
                loaded = loader.load()
                # Sanitise text + attach metadata
                for doc in loaded:
                    doc.page_content = _sanitise(doc.page_content)
                    doc.metadata.update(
                        {
                            "source": str(filepath),
                            "filename": filename,
                            "filetype": ext.lstrip("."),
                        }
                    )
                docs.extend(loaded)
            except Exception as exc:
                logger.error("Failed to load '%s': %s", filepath, exc)

    logger.info("Loaded %d raw document page(s) from '%s'.", len(docs), folder)
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunk(s).", len(chunks))
    return chunks


def load_and_split(folder: str | None = None) -> List[Document]:
    """Convenience function: load + split in one call."""
    docs = load_documents_from_folder(folder)
    if not docs:
        return []
    return split_documents(docs)
