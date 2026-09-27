# KraVerse AI backend

## Local/offline

Uses Ollama for chat and embeddings and SQLite for the local RAG store.

1. Install Python 3.11+ and Ollama.
2. Pull models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

3. Install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4. Start the API:

```bash
uvicorn app.main:app --reload
```

5. Index a project:

```bash
python scripts/ingest.py C:\path\to\project
```

6. POST questions to `/chat`.

## Online fallback

KraVerse AI can use OpenRouter's OpenAI-compatible API as an online fallback. Set `KRAVERSE_OPENROUTER_API_KEY` and optionally change `KRAVERSE_OPENROUTER_MODEL`. The default `openrouter/free` router selects from currently available free models.

POST questions to `/chat/online`.

Never commit API keys. Do not index `.env` files, credentials, secrets, or private repositories you do not want exposed.
