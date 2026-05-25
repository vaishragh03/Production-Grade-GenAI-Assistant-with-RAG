import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


class Settings:
    docs_path: Path = PROJECT_ROOT / os.getenv("DOCS_PATH", "docs.json")

    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.2)

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").lower()
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "") or os.getenv("LLM_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    sentence_transformer_model: str = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
    )

    chunk_size_tokens: int = _int("CHUNK_SIZE_TOKENS", 400)
    chunk_overlap_tokens: int = _int("CHUNK_OVERLAP_TOKENS", 50)
    top_k: int = _int("TOP_K", 3)
    similarity_threshold: float = _float("SIMILARITY_THRESHOLD", 0.45)
    max_history_pairs: int = _int("MAX_HISTORY_PAIRS", 5)

    insufficient_message: str = (
        "I could not find enough information in the knowledge base to answer this question."
    )


settings = Settings()
