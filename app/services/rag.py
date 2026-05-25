"""Coordinate indexing, retrieval, and LLM generation."""

from __future__ import annotations

import json

from app.services.chunking import chunk_all_documents
from app.services.embeddings import embed_texts
from app.services.llm import LLMError, generate_response
from app.services.retrieval import format_context, retrieve_context
from app.utils.config import settings
from app.utils.logger import logger
from app.vectorstore.memory_store import vector_store


class SessionStore:
    """In-memory conversation history keyed by sessionId."""

    def __init__(self, max_pairs: int = 5) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._max_messages = max_pairs * 2

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})
        if len(self._sessions[session_id]) > self._max_messages:
            self._sessions[session_id] = self._sessions[session_id][-self._max_messages :]

    def format_history(self, session_id: str) -> str:
        messages = self.get_history(session_id)
        lines = []
        for msg in messages:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)


session_store = SessionStore(max_pairs=settings.max_history_pairs)
_indexed = False


def load_documents() -> list[dict[str, str]]:
    path = settings.docs_path
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_knowledge_base() -> int:
    """Load docs, chunk, embed, and store vectors. Returns chunk count."""
    global _indexed
    documents = load_documents()
    chunks = chunk_all_documents(documents)
    if not chunks:
        logger.warning("No chunks produced from documents")
        _indexed = True
        return 0

    texts = [c["content"] for c in chunks]
    logger.info("Generating embeddings for %d chunks...", len(texts))
    embeddings = embed_texts(texts)

    vector_store.clear()
    vector_store.add_batch(chunks, embeddings)
    _indexed = True
    logger.info("Indexed %d chunks from %d documents", len(chunks), len(documents))
    return len(chunks)


def ensure_indexed() -> None:
    if not _indexed or vector_store.size == 0:
        index_knowledge_base()


def chat(session_id: str, message: str) -> dict:
    """
    Full RAG pipeline: retrieve first, then LLM (or insufficient-context reply).
    """
    ensure_indexed()

    results, sufficient = retrieve_context(message)
    retrieved_count = len(results) if sufficient else 0

    if not sufficient:
        reply = settings.insufficient_message
        session_store.append(session_id, "user", message)
        session_store.append(session_id, "assistant", reply)
        return {
            "reply": reply,
            "tokensUsed": 0,
            "retrievedChunks": 0,
        }

    context = format_context(results)
    history = session_store.format_history(session_id)

    try:
        llm_result = generate_response(context, history, message)
        reply = llm_result.text
        tokens_used = llm_result.tokens_used
    except LLMError as exc:
        raise

    session_store.append(session_id, "user", message)
    session_store.append(session_id, "assistant", reply)

    return {
        "reply": reply,
        "tokensUsed": tokens_used,
        "retrievedChunks": retrieved_count,
    }
