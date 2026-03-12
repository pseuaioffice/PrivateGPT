"""
embeddings.py
Custom LangChain Embeddings wrapper that calls Hugging Face Serverless via the
OpenAI-compatible inference endpoint.
"""
from typing import List
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
from config import settings


import requests
import logging

logger = logging.getLogger(__name__)

class UniversalEmbeddings(Embeddings):
    """
    Tries to choose between Hugging Face Cloud and local Ollama for embeddings.
    """

    def __init__(self) -> None:
        self._hf_client = InferenceClient(api_key=settings.HUGGINGFACE_TOKEN)
        self._hf_model = settings.EMBEDDING_MODEL
        self._ollama_url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"
        self._ollama_model = settings.EMBEDDING_MODEL_LOCAL

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if settings.MODEL_PROVIDER == "ollama":
            return self._embed_ollama(texts, settings.EMBEDDING_MODEL_LOCAL)
        else:
            return self._embed_huggingface(texts, settings.EMBEDDING_MODEL)

    def _embed_huggingface(self, texts: List[str], model: str) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._hf_client.feature_extraction(
                text=batch,
                model=model,
            )
            if hasattr(response, "tolist"):
                all_embeddings.extend(response.tolist())
            else:
                all_embeddings.extend(list(response))
        return all_embeddings

    def _embed_ollama(self, texts: List[str], model: str) -> List[List[float]]:
        """
        Generate embeddings using local Ollama instance.
        Tries new /api/embed first, falls back to /api/embeddings for older versions.
        """
        # Try new endpoint first (Ollama 0.1.24+)
        try:
            url = f"{settings.OLLAMA_BASE_URL}/api/embed"
            res = requests.post(
                url,
                json={"model": model, "input": texts},
                timeout=30,  # Add timeout for embeddings
            )
            if res.status_code == 200:
                return res.json()["embeddings"]
        except requests.exceptions.Timeout:
            logger.error("Ollama embedding request timed out after 30 seconds")
            raise RuntimeError("Embedding generation timed out. Please try again.")
        except Exception:
            pass
        
        # Fall back to legacy endpoint (older Ollama versions)
        try:
            url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"
            all_embeddings = []
            for text in texts:
                res = requests.post(
                    url,
                    json={"model": model, "prompt": text},
                    timeout=30,  # Add timeout for embeddings
                )
                res.raise_for_status()
                all_embeddings.append(res.json()["embedding"])
            return all_embeddings
        except requests.exceptions.Timeout:
            logger.error("Ollama embedding request timed out after 30 seconds")
            raise RuntimeError("Embedding generation timed out. Please try again.")
        except Exception as e:
            logger.error("Ollama embedding failed for model '%s': %s", model, e)
            raise RuntimeError(f"Ollama embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
