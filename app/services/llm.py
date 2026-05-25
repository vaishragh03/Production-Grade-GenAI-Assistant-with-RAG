"""LLM API integration with Gemini error handling and token logging."""

from __future__ import annotations

from dataclasses import dataclass

import google.generativeai as genai

from app.prompts.rag_prompt import (
    RAG_SYSTEM_PROMPT,
    build_user_prompt,
)
from app.utils.config import settings
from app.utils.logger import logger


class LLMError(Exception):
    """Raised when the LLM provider returns an error."""


@dataclass
class LLMResult:
    text: str
    tokens_used: int


def generate_response(
    retrieved_context: str,
    conversation_history: str,
    user_question: str,
) -> LLMResult:

    provider = settings.llm_provider

    if provider == "gemini":
        return _call_gemini(
            retrieved_context,
            conversation_history,
            user_question,
        )

    raise LLMError(
        f"Unsupported LLM provider: {provider}. "
        f"Set LLM_PROVIDER=gemini"
    )


def _call_gemini(
    retrieved_context: str,
    conversation_history: str,
    user_question: str,
) -> LLMResult:

    if not settings.llm_api_key:
        raise LLMError(
            "Invalid or missing API key. "
            "Set LLM_API_KEY in your .env file."
        )

    try:
        genai.configure(api_key=settings.llm_api_key)
        for m in genai.list_models():
          print(m.name)
        model = genai.GenerativeModel(
            "models/gemini-pro"
        )

        user_prompt = build_user_prompt(
            retrieved_context,
            conversation_history,
            user_question,
        )

        response = model.generate_content(
    contents=user_prompt,
    generation_config={
        "temperature": settings.llm_temperature,
    },
)
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)

        error_message = str(exc).lower()

        if "api key" in error_message:
            raise LLMError(
                "Invalid Gemini API key."
            ) from exc

        if "quota" in error_message:
            raise LLMError(
                "Gemini quota exceeded."
            ) from exc

        if "rate limit" in error_message:
            raise LLMError(
                "Gemini rate limit exceeded."
            ) from exc

        raise LLMError(
            f"Gemini language model error: {exc}"
        ) from exc

    text = ""

    if response.text:
        text = response.text.strip()

    tokens = 0

    try:
        usage = response.usage_metadata

        if usage:
            tokens = getattr(
                usage,
                "total_token_count",
                0,
            )

            logger.info(
                "Gemini token usage | total=%s",
                tokens,
            )

    except Exception:
        logger.warning(
            "Could not retrieve Gemini token usage."
        )

    return LLMResult(
        text=text,
        tokens_used=tokens,
    )