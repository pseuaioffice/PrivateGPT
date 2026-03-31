"""
rag_chain.py
RAG pipeline: retrieve relevant chunks → build prompt → call LLM.

Handles three interaction modes:
  1. Generic / social messages (hi, hello, thanks, etc.) - friendly response, no docs needed.
  2. Context-based questions WITH relevant docs found - grounded, cited answers.
  3. Context-based questions with NO relevant docs - graceful notice + general knowledge offer.
"""
import logging
import re
from typing import Any, Dict, List
from contextvars import ContextVar
import json

CANCELLED_CHATS = set()
current_chat_uuid: ContextVar[str] = ContextVar("current_chat_uuid", default="")

import requests

from huggingface_hub import InferenceClient
from openai import OpenAI
from langchain_core.documents import Document

from config import settings
from vector_store import similarity_search

logger = logging.getLogger(__name__)

# ── Generic / social message detection ────────────────────────────────────────
GENERIC_PATTERNS = re.compile(
    r'^(hi|hello|hey|howdy|hiya|greetings|good\s+(morning|afternoon|evening|day)|'
    r'how\s+are\s+you|what(\'s|\s+is)\s+up|sup|yo|thanks|thank\s+you|ty|thx|'
    r'bye|goodbye|see\s+you|ok|okay|sure|great|nice|cool|wow|lol|haha|'
    r'help|what can you do|what are you|who are you|introduce yourself'  
    r')[\.!?]*$',
    re.IGNORECASE
)

GENERIC_SYSTEM = (
    "You are MyAIAssistant, a friendly and expert AI document intelligence assistant. "
    "When users send you social greetings or generic messages (not document questions), "
    "reply warmly, be concise, and gently guide them toward asking questions about their documents. "
    "Keep replies to 2-4 sentences. Do NOT add '---SUGGESTIONS---' for generic messages."
)

CONTEXT_SYSTEM = """You are MyAIAssistant, a professional Document Intelligence Assistant.
Your goal is to provide comprehensive, structured, and accurate answers based on the provided context.

### GUIDELINES:
1. **Comprehensive Coverage**: Address all aspects of the user's query using the context below.
2. **Structure**: Use clear Markdown headings, sub-headings, and bullet lists.
3. **No Hallucination**: Answer ONLY using the provided context. Never invent facts outside the documents.
4. **Context Rejection (CRITICAL)**: If the provided context does NOT clearly contain the answer to the user's question, you MUST reply: "The selected document does not appear to contain information about this." Do NOT try to force an answer from irrelevant text.
5. **Citations**: Reference sources as [1], [2], etc. when relevant.

At the very end of your response, provide exactly 3 short, relevant follow-up \
questions the user could ask next. Prefix this section exactly with '---SUGGESTIONS---', \
then list the questions on new lines starting with '- '.

### CONTEXT FROM YOUR DOCUMENTS:
{context}
"""

NO_CONTEXT_SYSTEM = """You are MyAIAssistant, a professional Document Intelligence Assistant.
The user asked a question but NO relevant information was found in the indexed documents.

### YOUR RESPONSE MUST:
1. ALWAYS start with: "I don't have information about this topic in the uploaded documents."
2. NEVER provide general knowledge answers under ANY circumstances - this is a RAG application, not a general chatbot.
3. ONLY answer questions if you find relevant information in the documents.
4. If no relevant documents are found, you MUST refuse to answer and state that clearly.
5. Suggest user upload relevant documents if they want answers.
6. End with exactly 3 follow-up suggestions prefixed with '---SUGGESTIONS---'.

### STRICT ENFORCEMENT:
- Do NOT provide general knowledge about programming, Python, or any other topics.
- Do NOT be a general-purpose chatbot.
- You are a DOCUMENT-ONLY assistant.
- If documents don't contain the answer, you CANNOT provide it.
- Be firm: "I cannot answer this question from the available documents."

Example: "I don't have information about this topic in the uploaded documents. This appears to be a question about [topic], but the available documents don't cover this subject."
"""

STRICT_NO_CONTEXT_SYSTEM = """You are MyAIAssistant, a professional Document Intelligence Assistant in STRICT MODE.
The user asked a specific question about a selected document, but NO relevant information was found.

### YOUR RESPONSE MUST FOLLOW THESE RULES STRICTLY:
1. **ALWAYS START WITH**: "I don't have this information in the selected document."
2. **NO general knowledge** - You are in strict document-only mode. DO NOT provide any information outside of the document under ANY circumstances.
3. **NO hallucination** - Do not guess or make up answers. Be honest that the document doesn't contain this info.
4. **NO suggestions to upload other docs** - Stay focused on the current document only.
5. **Suggest relevant topics FROM THIS DOCUMENT** - If you can infer what topics the document covers, suggest follow-up questions about those topics only.
6. **Always end with '---SUGGESTIONS---'** followed by 3 questions about content that IS likely in the document.

### STRICT ENFORCEMENT:
- If the selected document doesn't contain information about the user's question, you MUST NOT provide an answer.
- Do not say "based on general knowledge" or "I would normally say" - these are hallucinations.
- Only reference what you know is in the document.
- Be firm and clear that you cannot help with this specific question from this document.
- REFUSE to answer if the information is not in the selected document.

Example response:
"I don't have this information in the selected document. This document appears to focus on [topic], so I cannot answer your question about [user's topic]. You might want to ask me about [related topic from doc] instead."

---SUGGESTIONS---
- What is...
- How does...
- Tell me about...
"""

_hf_client: InferenceClient | None = None
_oa_client: OpenAI | None = None


def _get_hf_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        _hf_client = InferenceClient(api_key=settings.HUGGINGFACE_TOKEN)
    return _hf_client


def _get_oa_client() -> OpenAI:
    global _oa_client
    if _oa_client is None:
        _oa_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _oa_client


def _call_ollama(messages: List[Dict[str, str]], fast_mode: bool = False) -> str:
    """Simple wrapper for Ollama local API with timeout handling and empty-response retry."""
    max_attempts = 2  # retry once if the model returns an empty response

    for attempt in range(1, max_attempts + 1):
        try:
            url = f"{settings.OLLAMA_BASE_URL}/api/chat"
            payload = {
                "model": settings.CHAT_MODEL_LOCAL,
                "messages": messages,
                "stream": True,
                "keep_alive": "15m",  # Keep model in memory for 15 mins
                "options": {
                    "temperature": 0.4 if fast_mode else 0.3,
                    "num_predict": 512 if fast_mode else 1024,
                    "num_ctx": 2048 if fast_mode else 4096,
                    "repeat_penalty": 1.25,
                    "top_k": 40,
                    "top_p": 0.9,
                }
            }
            logger.info(
                "Sending request to Ollama (%s) [fast=%s, attempt=%d]...",
                settings.CHAT_MODEL_LOCAL, fast_mode, attempt,
            )
            # Connect timeout: 10s, Read timeout: 240s for full answers; 90s for fast responses
            read_timeout = 90 if fast_mode else 240
            res = requests.post(url, json=payload, stream=True, timeout=(10, read_timeout))
            res.raise_for_status()

            full_text = []
            got_done = False
            for line in res.iter_lines():
                c_uuid = current_chat_uuid.get("")
                if c_uuid and c_uuid in CANCELLED_CHATS:
                    logger.warning("Generation aborted for chat %s", c_uuid)
                    res.close()
                    raise RuntimeError("Request cancelled by user.")

                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            if content:  # skip empty content chunks
                                full_text.append(content)
                        # Track whether Ollama signalled completion
                        if chunk.get("done") is True:
                            got_done = True
                    except Exception:
                        pass

            result = "".join(full_text).strip()

            if not result:
                # ── Blank response: Ollama responded but generated no text ──────────
                # This happens when the model is still warming up (first inference
                # after load) or when it hits num_predict=0 somehow.
                # Retry once; on second failure return a friendly message.
                if attempt < max_attempts:
                    logger.warning(
                        "Ollama returned an empty response (attempt %d/%d) — retrying...",
                        attempt, max_attempts,
                    )
                    continue
                else:
                    logger.error(
                        "Ollama returned empty response after %d attempt(s). "
                        "done_signal=%s",
                        max_attempts, got_done,
                    )
                    return (
                        "The model returned an empty response. "
                        "This can happen when the model is still loading — "
                        "please try your question again in a moment."
                    )

            logger.info(
                "Ollama responded with %d character(s) (attempt %d, done=%s)",
                len(result), attempt, got_done,
            )
            return result

        except RuntimeError:
            raise
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out (attempt %d)", attempt)
            raise RuntimeError(
                "The model is taking too long to respond. "
                "It might still be loading — please try again in a moment."
            )
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama at %s.", settings.OLLAMA_BASE_URL)
            raise RuntimeError(
                f"Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}. "
                "Please ensure Ollama is running."
            )
        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            raise RuntimeError(f"Local model (Ollama) error: {e}")

    # Should never reach here
    return "Unexpected error — please try again."


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _is_generic_message(question: str) -> bool:
    """Check if the question is a generic/social message that doesn't need document context."""
    cleaned = question.strip()
    return bool(GENERIC_PATTERNS.match(cleaned))


def _parse_suggestions(answer: str):
    """Extract and clean suggestions from answer text.

    Handles edge cases from small models that sometimes:
    - Put ---SUGGESTIONS--- at the very start (no answer text before it)
    - Include extra whitespace or numbering variants
    """
    suggestions: List[str] = []
    if not answer:
        return answer, suggestions

    if "---SUGGESTIONS---" in answer:
        parts = answer.split("---SUGGESTIONS---", 1)
        answer_part = str(parts[0]).strip()
        suggestions_text = str(parts[1]).strip() if len(parts) > 1 else ""

        suggestions_list = str(suggestions_text).split('\n')
        for raw_line in suggestions_list:
            line = str(raw_line).strip()
            if line.startswith('- '):
                suggestions.append(str(line).replace('- ', '', 1).strip())
            elif line.startswith(('* ', '1.', '2.', '3.')):
                # safely strip out any of these prefixes without using text slices
                clean_line = str(line)
                for prefix in ('* ', '1.', '2.', '3.'):
                    if str(line).startswith(prefix):
                        clean_line = str(line).replace(prefix, '', 1)
                        break
                suggestions.append(str(clean_line).strip())
            elif line and not str(line).startswith('#'):
                suggestions.append(line)

        # If the model only produced suggestions and no answer body, return the
        # raw text before the marker (usually empty) unchanged — the caller will
        # detect the empty string and handle it.
        top_suggestions = []
        for i in range(min(3, len(suggestions))):
            top_suggestions.append(suggestions[i])
            
        return answer_part, top_suggestions

    return answer.strip(), suggestions


def _call_llm(msgs: List[Dict[str, str]], fast_mode: bool = False) -> str:
    """Route to the correct LLM provider."""
    if settings.MODEL_PROVIDER == "ollama":
        return _call_ollama(msgs, fast_mode=fast_mode)
    
    model_name = settings.CHAT_MODEL
    is_openai = model_name.lower().startswith("gpt-")
    if is_openai:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OpenAI API Key is missing. Please add OPENAI_API_KEY to your .env file.")
        response = _get_oa_client().chat.completions.create(
            model=model_name, messages=msgs, temperature=0.3, max_tokens=2048,
        )
        return response.choices[0].message.content.strip()
    else:
        response = _get_hf_client().chat_completion(
            model=model_name, messages=msgs, temperature=0.2, max_tokens=4096,
        )
        return response.choices[0].message.content.strip()


def ask(question: str, k: int = 10, document_name: str | None = None, is_strict_mode: bool = False) -> Dict[str, Any]:
    """
    Run the RAG pipeline for *question*.
    
    Args:
        question: The user's question
        k: Number of relevant chunks to retrieve (default 10)
        document_name: If provided, search only within this document
        is_strict_mode: If True, only answer from the selected document (strict RAG).
                       If False, can use general knowledge if document doesn't have answer.
    
    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "suggestions": list[str],
            "no_context": bool,
            "is_generic": bool,
            "is_strict_mode": bool,
        }
    """
    import time
    start_time = time.time()

    # ── Mode 1: Generic / social message ──────────────────────────────────
    if _is_generic_message(question):
        logger.info("Generic message detected — skipping vector search.")
        msgs = [
            {"role": "system", "content": GENERIC_SYSTEM},
            {"role": "user",   "content": question},
        ]
        answer = _call_llm(msgs, fast_mode=True)
        answer, suggestions = _parse_suggestions(answer)
        logger.info("Generic response generated in %.2fs", time.time() - start_time)
        return {
            "answer": answer,
            "sources": [],
            "suggestions": suggestions,
            "no_context": False,
            "is_generic": True,
            "is_strict_mode": is_strict_mode,
        }

    # ── Vector search ──────────────────────────────────────────────────────
    docs, scores = similarity_search(question, k=k, document_name=document_name, include_scores=True)
    search_time = time.time() - start_time
    logger.info(
        "Similarity search took %.2fs (%d chunks found), scores: %s",
        search_time, len(docs), [f"{s:.3f}" for s in scores],
    )

    # ── Strict Relative Thresholding ──────────────────────────────────────
    # To prevent LLM hallucination and repetition loops when answering from garbage:
    # 1. We require a minimum baseline match of 0.25. If nothing hits 0.25, the 
    #    query is fundamentally unrelated to the document.
    # 2. We drop any chunks that are more than 0.10 worse than the best match,
    #    so highly relevant queries aren't diluted by adding tangential chunks.
    if docs and scores:
        top_score = scores[0]
        
        # Determine our cutoff
        absolute_floor = 0.25
        relative_floor = top_score - 0.10
        cutoff = max(absolute_floor, relative_floor)
        
        valid_docs = []
        valid_scores = []
        
        for d, s in zip(docs, scores):
            if s >= cutoff:
                valid_docs.append(d)
                valid_scores.append(s)
                
        docs = valid_docs
        scores = valid_scores
        
        if not docs:
            logger.warning("All %d retrieved chunks fell below the %.2f safety cutoff (top_score was %.3f).", 
                           k, cutoff, top_score)
        else:
            logger.info("Kept %d chunk(s) using cutoff %.3f (top_score: %.3f).", 
                        len(docs), cutoff, top_score)

    # ── Mode 2: No relevant docs found ────────────────────────────────────
    if not docs:
        mode_label = "STRICT MODE" if is_strict_mode else "FLEXIBLE MODE"
        logger.warning(
            "%s: No chunks in vector store for question='%s' doc='%s'",
            mode_label, question, document_name or "All",
        )
        no_context_prompt = STRICT_NO_CONTEXT_SYSTEM if is_strict_mode else NO_CONTEXT_SYSTEM
        msgs = [
            {"role": "system", "content": no_context_prompt},
            {"role": "user",   "content": question},
        ]
        answer = _call_llm(msgs, fast_mode=False)
        answer, suggestions = _parse_suggestions(answer)
        return {
            "answer": answer,
            "sources": [],
            "suggestions": suggestions,
            "no_context": True,
            "is_generic": False,
            "is_strict_mode": is_strict_mode,
        }

    # ── Mode 3: Context-grounded answer ───────────────────────────────────
    mode_label = "STRICT" if is_strict_mode else "FLEXIBLE"
    logger.info(
        "[%s] %d chunk(s) retrieved (top score=%.3f). Calling LLM.",
        mode_label, len(docs), scores[0],
    )

    context    = _format_docs(docs)
    system_msg = CONTEXT_SYSTEM.format(context=context)
    msgs = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": question},
    ]
    llm_start = time.time()
    answer    = _call_llm(msgs, fast_mode=False)
    logger.info("LLM response took %.2fs", time.time() - llm_start)

    answer, suggestions = _parse_suggestions(answer)

    sources = [
        {
            "filename": d.metadata.get("filename", "unknown"),
            "filetype": d.metadata.get("filetype", ""),
            "source":   d.metadata.get("source", ""),
            "snippet":  d.page_content[:300],
        }
        for d in docs
    ]

    logger.info("Total ask() time: %.2fs", time.time() - start_time)
    return {
        "answer":        answer,
        "sources":       sources,
        "suggestions":   suggestions,
        "no_context":    False,
        "is_generic":    False,
        "is_strict_mode": is_strict_mode,
    }
