"""Generate text embeddings via Sentence Transformers or OpenAI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from app.utils.config import settings
from app.utils.logger import logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model: "SentenceTransformer | None" = None


def _get_sentence_transformer() -> "SentenceTransformer":
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading Sentence Transformer: %s", settings.sentence_transformer_model)
        _model = SentenceTransformer(settings.sentence_transformer_model)
    return _model


def _embed_openai(texts: list[str]) -> np.ndarray:
    if not settings.embedding_api_key:
        raise ValueError("EMBEDDING_API_KEY or LLM_API_KEY is required for OpenAI embeddings")

    from openai import OpenAI

    client = OpenAI(api_key=settings.embedding_api_key, timeout=60.0)
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
    return np.array(vectors, dtype=np.float32)


def _embed_sentence_transformers(texts: list[str]) -> np.ndarray:
    model = _get_sentence_transformer()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.array([], dtype=np.float32)

    provider = settings.embedding_provider
    if provider == "openai":
        return _embed_openai(texts)
    if provider in ("sentence-transformers", "sentence_transformers", "local"):
        return _embed_sentence_transformers(texts)

    raise ValueError(f"Unsupported embedding provider: {provider}")
