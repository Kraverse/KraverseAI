# KraVerse AI backend

Local FastAPI + Ollama + SQLite RAG backend.

## Run

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install Ollama and pull the chat and embedding models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Start the API:

```bash
uvicorn app.main:app --reload
```

## Index a project

From the `backend` directory:

```bash
python scripts/ingest.py ../path/to/project
```

The indexer reads common source/document files, chunks them, creates local Ollama embeddings, and stores them in SQLite at `data/kraverse.db`.

## API

- `GET /health`
- `GET /health/ollama`
- `POST /chat` with `{ "message": "Explain my HelpDesk AI project" }`

The chat endpoint retrieves the most relevant indexed chunks before asking the local model to answer. Responses include the source chunk paths used by retrieval.

## Privacy

The knowledge database is local by default. Do not commit `data/kraverse.db`, private project files, `.env` files, or secrets.
