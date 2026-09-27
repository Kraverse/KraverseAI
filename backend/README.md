# KraVerse AI backend

## Local setup

1. Install Python 3.11+ and Ollama.
2. Pull a chat model and embedding model:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

3. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4. Start the API:

```bash
uvicorn app.main:app --reload
```

5. Index a local project:

```bash
python scripts/ingest.py C:\path\to\project
```

6. Open `/docs` and POST to `/chat`.

The database is local SQLite. No project files are uploaded by the backend. Do not index secrets, `.env` files, private credentials, or repositories you do not want exposed to the assistant.
