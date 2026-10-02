# KraVerse AI

Personal AI-powered portfolio and knowledge assistant for Kartik Katke.

## Architecture

```text
Portfolio chat
    -> FastAPI backend
    -> verified knowledge / RAG retrieval
    -> configurable LLM provider
       |-- Online: OpenRouter-compatible API
       `-- Local: Ollama
```

## API

- `GET /` — service status and links
- `GET /health` — backend health and configured provider
- `GET /docs` — interactive FastAPI documentation
- `POST /chat` — local Ollama + SQLite RAG
- `POST /chat/online` — verified knowledge + online LLM

Example request:

```json
{"message":"What is HelpDesk AI?"}
```

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

For local project indexing, use `scripts/ingest.py` with a local project directory. The selected repository map is in `knowledge/projects.json`.

## Online configuration

Set these environment variables on the backend deployment (never commit the key):

```text
KRAVERSE_LLM_PROVIDER=openrouter
KRAVERSE_LLM_API_KEY=your_key_here
KRAVERSE_LLM_MODEL=openrouter/free
KRAVERSE_LLM_BASE_URL=https://openrouter.ai/api/v1
```

The frontend never receives the LLM API key.

## Knowledge

Approved profile and project facts live under `knowledge/`. Keep these factual and update them when verified information changes. Do not add secrets or private repository contents.

## Deployment

The FastAPI backend is configured for Render with:

```text
Build:  pip install -r backend/requirements.txt
Start: uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

Live backend: `https://kraverse-ai-api.onrender.com`

## Status

Core backend, local RAG, online provider support, verified personal knowledge, and portfolio chat integration are implemented. Live runtime verification and online answering require the Render environment to contain a valid LLM API key and the portfolio to point to the live backend.
