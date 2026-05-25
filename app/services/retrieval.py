"""Similarity search over the vector store."""

from __future__ import annotations

from app.services.embeddings import embed_text
from app.utils.config import settings
from app.utils.logger import logger
from app.vectorstore.memory_store import SearchResult, vector_store


def retrieve_context(query: str) -> tuple[list[SearchResult], bool]:
    """
    Embed the query, run top-K cosine similarity search, apply threshold.

    Returns (results, has_sufficient_context).
    """

    logger.info("User Query: %s", query)

    query_embedding = embed_text(query)

    results = vector_store.search(
        query_embedding,
        top_k=settings.top_k,
    )

    if not results:
        logger.info("Retrieval: no chunks in vector store")
        return [], False

    best_score = results[0].score

    logger.info(
        "Retrieval: best_score=%.4f threshold=%.4f",
        best_score,
        settings.similarity_threshold,
    )

    for i, result in enumerate(results, 1):
        logger.info(
            "Result %d | score=%.4f | title=%s",
            i,
            result.score,
            result.document_title,
        )

    if best_score < settings.similarity_threshold:
        logger.info("Retrieval: insufficient similarity score")
        return results, False

    logger.info("Retrieval: sufficient context found")

    return results, True


def format_context(chunks: list[SearchResult]) -> str:
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Title: {chunk.document_title}\n"
            f"Source: {chunk.source_document}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Content: {chunk.content}"
        )
    return "\n\n".join(parts)
