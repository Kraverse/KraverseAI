from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .knowledge import retrieve
from .ollama import chat
from .online import chat_online
from .rag import RAGStore, ollama_embed

app = FastAPI(title="KraVerse AI API", version="0.4.0")
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


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "KraVerse AI API",
        "status": "ok",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local+online", "provider": settings.llm_provider}


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
        raise HTTPException(status_code=502, detail="Local AI request failed") from exc


@app.post("/chat/online")
async def online_chat_endpoint(request: ChatRequest) -> dict:
    if settings.llm_provider.lower() != "openrouter":
        raise HTTPException(status_code=503, detail=f"Unsupported online provider: {settings.llm_provider}")
    if not settings.llm_api_key:
        raise HTTPException(status_code=503, detail="Online provider is not configured")
    try:
        results = retrieve(request.message, request.top_k or settings.top_k)
        context = "\n\n".join(f"SOURCE: {item['source']}\n{item['content']}" for item in results)
        prompt = f"""You are KraVerse AI, Kartik Katke's portfolio assistant. Answer clearly and honestly. Use ONLY the verified context below for personal and project facts. Do not invent missing details. If the context is insufficient, say that you do not have enough verified information.\n\nVERIFIED CONTEXT:\n{context or '(No matching verified knowledge found.)'}\n\nUSER QUESTION:\n{request.message}"""
        answer = await chat_online(prompt, settings.llm_api_key, settings.llm_model, settings.llm_base_url)
        return {"answer": answer, "sources": [item["source"] for item in results], "mode": "online", "model": settings.llm_model}
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Online AI request failed") from exc
