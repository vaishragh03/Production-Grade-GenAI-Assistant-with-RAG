# GenAI RAG Assistant — TRUEAILAB Assignment

Production-style chat assistant that answers questions using **Retrieval-Augmented Generation (RAG)** over a custom document knowledge base.

## Features

- Document loading from `docs.json` with chunking (300–500 token targets)
- Embedding generation (Sentence Transformers or OpenAI)
- In-memory vector store with **cosine similarity** search
- Top-K retrieval with configurable similarity threshold
- OpenAI LLM integration with error handling and token logging
- Session-based conversation history (last 5 message pairs)
- FastAPI backend + HTML/CSS/JS frontend with localStorage sessions

## Project Structure

```
project/
├── app/
│   ├── routes/chat.py
│   ├── services/{chunking,embeddings,retrieval,rag,llm}.py
│   ├── models/schemas.py
│   ├── vectorstore/memory_store.py
│   ├── prompts/rag_prompt.py
│   ├── utils/{config,logger}.py
│   └── main.py
├── frontend/{index.html,styles.css,app.js}
├── docs.json
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

First run downloads the Sentence Transformer model (~90MB).

### 3. Configure environment

Copy `.env.example` to `.env` and set your API key:

```env
LLM_API_KEY=sk-your-openai-key
EMBEDDING_PROVIDER=sentence-transformers
```

Embeddings run locally by default (no key required). Set `EMBEDDING_PROVIDER=openai` to use OpenAI embeddings.

### 4. Run the server

From the project root:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) for the chat UI.

## API

### `GET /health`

```json
{ "status": "healthy" }
```

### `POST /api/chat`

**Request:**

```json
{
  "sessionId": "abc123",
  "message": "How can I reset my password?"
}
```

**Response:**

```json
{
  "reply": "Users can reset their password from Settings > Security.",
  "tokensUsed": 120,
  "retrievedChunks": 3
}
```

**Error example:**

```json
{ "error": "Message field is required" }
```

## RAG Flow

1. **Startup** — Load `docs.json`, chunk documents, generate embeddings, store in vector DB
2. **Query** — Embed user message → cosine similarity → top-K chunks
3. **Threshold** — If best score &lt; `SIMILARITY_THRESHOLD`, return insufficient-context message (no LLM call)
4. **Generate** — Build prompt with context + history + question → LLM → reply

Similarity scores are logged on every retrieval.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | — | OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `LLM_TEMPERATURE` | `0.2` | 0–0.3 recommended |
| `EMBEDDING_PROVIDER` | `sentence-transformers` | or `openai` |
| `TOP_K` | `3` | Chunks retrieved |
| `SIMILARITY_THRESHOLD` | `0.45` | Min cosine score (tune per embedding model) |
| `MAX_HISTORY_PAIRS` | `5` | Conversation pairs in prompt |

## Example Questions

- How can I reset my password?
- What subscription plans are available?
- How do I enable two-factor authentication?
- What is the refund policy?

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`) or OpenAI
- **Vector store:** In-memory + scikit-learn cosine similarity
- **LLM:** OpenAI (extensible to other providers)
- **Frontend:** HTML, CSS, JavaScript (localStorage sessions)

## License

Educational project for TRUEAILAB assignment.
