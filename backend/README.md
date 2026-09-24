# KraVerse AI backend

Minimal FastAPI service for the local AI phase.

## Run

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Local model

Install Ollama separately and make sure the configured model is available.

```bash
ollama pull llama3.2:3b
```

Then check:

- `GET /health`
- `GET /health/ollama`
- `POST /chat` with `{ "message": "Hello" }`

RAG is intentionally not included yet. The next phase will add a small document ingestion and retrieval layer before the model receives project context.
