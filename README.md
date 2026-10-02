# KraVerse AI

Personal AI-powered portfolio and knowledge assistant for Kartik Katke.

## What it does

KraVerse AI can answer verified questions about Kartik's profile and projects using a grounded knowledge layer plus a configurable LLM provider.

## Architecture

```text
Portfolio chat
    -> FastAPI backend
    -> verified knowledge / RAG retrieval
    -> LLM provider
       |-- Online: OpenRouter-compatible API
       `-- Local: Ollama
```

## Modes

- **Online:** `/chat/online` retrieves approved knowledge from `knowledge/` and sends only that context to the configured API model.
- **Local:** `/chat` uses Ollama embeddings + the local SQLite RAG store + an Ollama chat model.
- **Portfolio:** the static portfolio contains the KraVerse AI chat widget and can be pointed at the deployed backend with `window.KRAVERSE_API_URL` or browser storage.

## Local backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
uvicorn app.main:app --reload
```

For local project indexing, use the existing `scripts/ingest.py` with a local project directory. The selected repository map is in `knowledge/projects.json`.

## Online configuration

Create `backend/.env` locally (never commit it):

```text
KRAVERSE_OPENROUTER_API_KEY=your_key_here
KRAVERSE_OPENROUTER_MODEL=openrouter/free
KRAVERSE_OPENROUTER_URL=https://openrouter.ai/api/v1
```

The API key is read from the environment and is never placed in the portfolio frontend.

## Knowledge

Approved profile and project facts live under `knowledge/`. Keep these factual and update them when verified information changes. Do not add secrets or private repository contents.

## Status

Core backend, local RAG, online provider support, verified personal knowledge, and portfolio chat integration are implemented. Final deployment and end-to-end runtime verification still require a configured LLM API key and a deployed backend.
