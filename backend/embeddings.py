"""
embeddings.py
Custom LangChain Embeddings wrapper that calls Hugging Face Serverless via the
OpenAI-compatible inference endpoint.
"""
from typing import List
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
from config import settings


class HuggingFaceServerlessEmbeddings(Embeddings):
    """
    Uses the Hugging Face Serverless inference endpoint (OpenAI-compatible) to
    generate embeddings with 'text-embedding-3-small' (or whatever model
    is set in EMBEDDING_MODEL).
    """

    def __init__(self) -> None:
        self._client = InferenceClient(api_key=settings.HUGGINGFACE_TOKEN)
        self._model = settings.EMBEDDING_MODEL

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # Hugging Face serverless has a request-size limit — batch in groups of 16
        all_embeddings: List[List[float]] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.feature_extraction(
                text=batch,
                model=self._model,
            )
            # Response is typically a numpy array
            if hasattr(response, "tolist"):
                all_embeddings.extend(response.tolist())
            else:
                all_embeddings.extend(list(response))
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
