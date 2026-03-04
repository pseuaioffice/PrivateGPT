"""
rag_chain.py
RAG pipeline: retrieve relevant chunks → build prompt → call Hugging Face.

Uses the huggingface_hub InferenceClient directly to interact with models.
"""
import logging
from typing import Any, Dict, List

import requests

from huggingface_hub import InferenceClient
from langchain_core.documents import Document

from config import settings
from vector_store import similarity_search

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional Document Intelligence Assistant. Your goal is to provide \
comprehensive, structured, and accurate answers based strictly on the provided context.

### GUIDELINES:
1. **Comprehensive Coverage**: Ensure you address all aspects of the user's query. If the context covers multiple subjects (e.g., Maths, Physics, Chemistry), include relevant information from all of them.
2. **Structure**: Use clear Markdown headings, sub-headings, and lists to organize complex information.
3. **No Hallucination**: Answer ONLY using the provided context. If the information is not present, say: "I'm sorry, but I couldn't find information about [topic] in the available documents."
4. **Citations**: When possible, refer to the document sources provided in the context format [1], [2], etc.

At the very end of your response, provide exactly 3 short, relevant follow-up \
questions the user could ask next. Prefix this section exactly with '---SUGGESTIONS---', \
and then list the questions on new lines starting with '- '.

### CONTEXT:
{context}
"""

_hf_client: InferenceClient | None = None


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClient(api_key=settings.HUGGINGFACE_TOKEN)
    return _hf_client


def _call_ollama(messages: List[Dict[str, str]]) -> str:
    """Simple wrapper for Ollama local API."""
    try:
        url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": settings.CHAT_MODEL_LOCAL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 4096
            }
        }
        res = requests.post(url, json=payload, timeout=300)
        res.raise_for_status()
        return res.json()["message"]["content"]
    except Exception as e:
        logger.error("Ollama call failed: %s", e)
        raise RuntimeError(f"Local model (Ollama) error: {e}")


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def ask(question: str, k: int = 15) -> Dict[str, Any]:
    """
    Run the RAG pipeline for *question*.
    Returns:
        {
            "answer": str,
            "sources": list[dict]
        }
    """
    docs = similarity_search(question, k=k)
    context = _format_docs(docs) if docs else "No relevant documents found."

    system_msg = SYSTEM_PROMPT.format(context=context)
    msgs = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": question},
    ]

    if settings.MODEL_PROVIDER == "ollama":
        logger.info("Calling local Ollama (%s) with %d chunks.", settings.CHAT_MODEL_LOCAL, len(docs))
        answer = _call_ollama(msgs)
    else:
        logger.info("Calling Hugging Face Cloud (%s) with %d chunks.", settings.CHAT_MODEL, len(docs))
        response = _get_hf_client().chat_completion(
            model=settings.CHAT_MODEL,
            messages=msgs,
            temperature=0.2,
            max_tokens=4096,
        )
        answer = response.choices[0].message.content.strip()

    suggestions = []
    if "---SUGGESTIONS---" in answer:
        parts = answer.split("---SUGGESTIONS---")
        answer = parts[0].strip()
        suggestions_text = parts[1].strip()
        for line in suggestions_text.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                suggestions.append(line.replace('- ', '', 1).strip())
            elif line.startswith(('* ', '1.', '2.', '3.')):
                suggestions.append(line[2:].strip())

    sources = [
        {
            "filename": d.metadata.get("filename", "unknown"),
            "filetype": d.metadata.get("filetype", ""),
            "source":   d.metadata.get("source", ""),
            "snippet":  d.page_content[:300],
        }
        for d in docs
    ]

    return {"answer": answer, "sources": sources, "suggestions": suggestions[:3]}
