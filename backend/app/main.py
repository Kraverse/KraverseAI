from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .ollama import chat
from .rag import RAGStore, ollama_embed

app = FastAPI(title="KraVerse AI API", version="0.1.0")
store = RAGStore(settings.database_path)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    top_k: int | None = Field(default=None, ge=1, le=10)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local"}


@app.get("/health/ollama")
async def ollama_health() -> dict[str, str]:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{settings.ollama_url.rstrip('/')}/api/tags")
            response.raise_for_status()
        return {"status": "ok", "model": settings.chat_model}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {exc}") from exc


@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> dict:
    try:
        query_embedding = await ollama_embed(request.message, settings.ollama_url, settings.embedding_model)
        results = store.search(query_embedding, request.top_k or settings.top_k)
        context = "\n\n".join(
            f"SOURCE: {item['source']}\n{item['content']}" for item in results
        )
        prompt = f"""You are KraVerse AI, a personal project and portfolio assistant.\nAnswer using the supplied context. Do not invent facts. If the context does not contain the answer, say that you do not have enough verified information.\n\nCONTEXT:\n{context or '(No indexed knowledge yet.)'}\n\nUSER QUESTION:\n{request.message}"""
        answer = await chat(prompt, settings.chat_model, settings.ollama_url)
        return {"answer": answer, "sources": [item["source"] for item in results]}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
