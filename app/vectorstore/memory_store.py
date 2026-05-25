"""In-memory vector store with cosine similarity search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.logger import logger


@dataclass
class VectorRecord:
    chunk_id: str
    document_title: str
    source_document: str
    content: str
    embedding: np.ndarray


@dataclass
class SearchResult:
    chunk_id: str
    document_title: str
    source_document: str
    content: str
    score: float


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    @property
    def size(self) -> int:
        return len(self._records)

    def add(self, metadata: dict[str, Any], embedding: np.ndarray) -> None:
        self._records.append(
            VectorRecord(
                chunk_id=metadata["chunk_id"],
                document_title=metadata["document_title"],
                source_document=metadata["source_document"],
                content=metadata["content"],
                embedding=embedding.astype(np.float32),
            )
        )

    def add_batch(
        self, chunks: list[dict[str, Any]], embeddings: np.ndarray
    ) -> None:
        for chunk, vector in zip(chunks, embeddings):
            self.add(chunk, vector)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> list[SearchResult]:
        if not self._records:
            return []

        matrix = np.vstack([r.embedding for r in self._records])
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores = cosine_similarity(query, matrix)[0]

        indexed = [
            (idx, float(scores[idx]))
            for idx in range(len(self._records))
        ]
        indexed.sort(key=lambda x: x[1], reverse=True)
        top = indexed[:top_k]

        results: list[SearchResult] = []
        for idx, score in top:
            record = self._records[idx]
            logger.info(
                "Similarity | chunk=%s | title=%s | score=%.4f",
                record.chunk_id,
                record.document_title,
                score,
            )
            results.append(
                SearchResult(
                    chunk_id=record.chunk_id,
                    document_title=record.document_title,
                    source_document=record.source_document,
                    content=record.content,
                    score=score,
                )
            )
        return results

    def clear(self) -> None:
        self._records.clear()


vector_store = InMemoryVectorStore()
