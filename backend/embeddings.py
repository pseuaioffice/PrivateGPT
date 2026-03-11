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
            return self._embed_ollama(texts)
        else:
            return self._embed_huggingface(texts)

    def _embed_huggingface(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._hf_client.feature_extraction(
                text=batch,
                model=self._hf_model,
            )
            if hasattr(response, "tolist"):
                all_embeddings.extend(response.tolist())
            else:
                all_embeddings.extend(list(response))
        return all_embeddings

    def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for text in texts:
            try:
                res = requests.post(
                    self._ollama_url,
                    json={"model": self._ollama_model, "prompt": text},
                    timeout=30,
                )
                res.raise_for_status()
                emb = res.json()["embedding"]
                all_embeddings.append(emb)
            except Exception as e:
                logger.error("Ollama embedding failed for model '%s': %s", self._ollama_model, e)
                # If we fail, we should probably raise so the user knows, 
                # but for documents we want to continue if possible.
                # However, shape mismatch will break Chroma later.
                raise RuntimeError(f"Ollama embedding failed: {e}")
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
