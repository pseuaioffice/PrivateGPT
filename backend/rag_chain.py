"""
rag_chain.py
RAG pipeline: retrieve relevant chunks → build prompt → call Hugging Face.

Uses the huggingface_hub InferenceClient directly to interact with models.
"""
import logging
from typing import Any, Dict, List

from huggingface_hub import InferenceClient
from langchain_core.documents import Document

from config import settings
from vector_store import similarity_search

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly \
based on the provided context documents. If the answer cannot be found in the \
context, say so clearly — do NOT make up information.

At the very end of your response, provide exactly 3 short, relevant follow-up \
questions the user could ask next. Prefix this section exactly with '---SUGGESTIONS---', \
and then list the questions on new lines starting with '- '.

Context:
{context}
"""

_hf_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClient(api_key=settings.HUGGINGFACE_TOKEN)
    return _hf_client


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def ask(question: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline for *question*.
    Returns:
        {
            "answer": str,
            "sources": list[dict]
        }
    """
    docs = similarity_search(question, k=5)
    context = _format_docs(docs) if docs else "No relevant documents found."

    system_msg = SYSTEM_PROMPT.format(context=context)

    logger.info("Calling %s with %d retrieved chunk(s).", settings.CHAT_MODEL, len(docs))

    response = _get_client().chat_completion(
        model=settings.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": question},
        ],
        temperature=0.2,
        max_tokens=2048,
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
