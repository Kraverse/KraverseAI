from os import getenv

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag import RAGStore, embed

OLLAMA_URL = getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = getenv("EMBEDDING_MODEL", "nomic-embed-text")
RAG_TOP_K = int(getenv("RAG_TOP_K", "5"))
RAG_MIN_SCORE = float(getenv("RAG_MIN_SCORE", "0.15"))
RAG_DB_PATH = getenv("RAG_DB_PATH", "data/kraverse.db")

app = FastAPI(title="KraVerse AI API", version="0.2.0")
store = RAGStore(RAG_DB_PATH)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    answer: str
    model: str
    mode: str = "local-rag"
    sources: list[str] = []


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local-rag"}


@app.get("/health/ollama")
async def ollama_health() -> dict[str, str | bool]:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
        return {"available": True, "model": OLLAMA_MODEL, "embedding_model": EMBEDDING_MODEL}
    except (httpx.HTTPError, httpx.RequestError) as exc:
        return {"available": False, "model": OLLAMA_MODEL, "error": str(exc)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        query_embedding = await embed(request.message, OLLAMA_URL, EMBEDDING_MODEL)
        matches = [item for item in store.search(query_embedding, RAG_TOP_K) if float(item["score"]) >= RAG_MIN_SCORE]
    except (httpx.HTTPError, httpx.RequestError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Local retrieval is unavailable: {exc}") from exc

    context = "\n\n".join(
        f"SOURCE: {item['source']}\n{item['content']}" for item in matches
    )
    system_prompt = (
        "You are KraVerse AI, Kartik Katke's personal project-aware assistant. "
        "Answer using the supplied knowledge when relevant. Do not invent facts about Kartik or his projects. "
        "If the supplied knowledge does not contain the answer, say that you do not have verified information. "
        "Keep answers clear and useful.\n\n"
        f"RETRIEVED KNOWLEDGE:\n{context or '(No relevant indexed knowledge found.)'}"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, httpx.RequestError) as exc:
        raise HTTPException(status_code=503, detail=f"Local Ollama is unavailable: {exc}") from exc

    answer = data.get("message", {}).get("content", "").strip()
    if not answer:
        raise HTTPException(status_code=502, detail="Ollama returned an empty response")

    sources = list(dict.fromkeys(str(item["source"]) for item in matches))
    return ChatResponse(answer=answer, model=OLLAMA_MODEL, sources=sources)
