"""Split documents into overlapping chunks (~300-500 tokens)."""

from __future__ import annotations

import re
from typing import Any

from app.utils.config import settings

# Rough estimate: 1 token ≈ 4 characters for English prose
CHARS_PER_TOKEN = 4


def _token_to_chars(tokens: int) -> int:
    return tokens * CHARS_PER_TOKEN


def chunk_document(
    title: str,
    content: str,
    source_index: int,
) -> list[dict[str, Any]]:
    """Split a single document into chunks with metadata."""
    chunk_size = _token_to_chars(settings.chunk_size_tokens)
    overlap = _token_to_chars(settings.chunk_overlap_tokens)

    text = re.sub(r"\s+", " ", content.strip())
    if not text:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"{source_index}-{chunk_index}",
                    "document_title": title,
                    "source_document": title,
                    "content": chunk_text,
                }
            )
            chunk_index += 1

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_all_documents(documents: list[dict[str, str]]) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    for idx, doc in enumerate(documents):
        title = doc.get("title", f"Document {idx}")
        content = doc.get("content", "")
        all_chunks.extend(chunk_document(title, content, idx))
    return all_chunks
