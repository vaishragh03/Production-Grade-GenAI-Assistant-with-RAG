RAG_SYSTEM_PROMPT = """You are a helpful assistant for a product knowledge base.

Use ONLY the provided context to answer the user's question.
If the context does not contain enough information, say exactly:
"I could not find enough information in the knowledge base to answer this question."

Do not invent facts. Be concise and accurate."""


def build_user_prompt(
    retrieved_context: str,
    conversation_history: str,
    user_question: str,
) -> str:
    return f"""Use ONLY the provided context to answer.

Context:
{retrieved_context}

Conversation History:
{conversation_history or "(none)"}

Question:
{user_question}"""
