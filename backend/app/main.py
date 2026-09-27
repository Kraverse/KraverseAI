from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .ollama import chat
from .online import chat_online
from .rag import RAGStore, ollama_embed

app = FastAPI(title="KraVerse AI API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
store = RAGStore(settings.database_path)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    top_k: int | None = Field(default=None, ge=1, le=10)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local+online"}


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
        context = "\n\n".join(f"SOURCE: {item['source']}\n{item['content']}" for item in results)
        prompt = f"""You are KraVerse AI, a personal project and portfolio assistant.\nAnswer using the supplied context. Do not invent facts. If the context does not contain the answer, say that you do not have enough verified information.\n\nCONTEXT:\n{context or '(No indexed knowledge yet.)'}\n\nUSER QUESTION:\n{request.message}"""
        answer = await chat(prompt, settings.chat_model, settings.ollama_url)
        return {"answer": answer, "sources": [item["source"] for item in results], "mode": "local"}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/chat/online")
async def online_chat_endpoint(request: ChatRequest) -> dict:
    if not settings.openrouter_api_key:
        raise HTTPException(status_code=503, detail="Online provider is not configured")
    try:
        prompt = f"""You are KraVerse AI, Kartik Katke's portfolio assistant. Answer clearly and honestly. Only state personal/project facts supplied by the user or by the portfolio knowledge context. If information is missing, say so.\n\nUSER QUESTION:\n{request.message}"""
        answer = await chat_online(prompt, settings.openrouter_api_key, settings.openrouter_model, settings.openrouter_url)
        return {"answer": answer, "sources": [], "mode": "online", "model": settings.openrouter_model}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
