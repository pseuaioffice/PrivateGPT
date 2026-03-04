"""
config.py
Centralised configuration loaded from the .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Hugging Face Models
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    HUGGINGFACE_BASE_URL: str = os.getenv(
        "HUGGINGFACE_BASE_URL", "https://router.huggingface.co/hf-inference/v1"
    )

    # Model names
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "rag_documents"
    )

    # Document folder
    DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", "./documents")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # PostgreSQL (optional — chat history disabled when not set)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Local Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MODEL_PROVIDER: str = "huggingface"  # default
    CHAT_MODEL_LOCAL: str = "qwen"

    def validate(self) -> None:
        if not self.HUGGINGFACE_TOKEN:
            raise EnvironmentError(
                "HUGGINGFACE_TOKEN is not set. "
                "Add it to backend/.env before starting the server."
            )


settings = Settings()
