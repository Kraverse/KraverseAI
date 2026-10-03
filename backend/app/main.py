from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .knowledge import retrieve
from .ollama import chat
from .online import chat_online
from .rag import RAGStore, ollama_embed

app = FastAPI(title="KraVerse AI API", version="0.5.0")
origins = [item.strip() for item in settings.cors_origins.split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
store = RAGStore(settings.database_path)
_requests: dict[str, deque[float]] = defaultdict(deque)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    interview_mode: bool = False


def _check_rate_limit(request: Request) -> None:
    limit = max(1, settings.rate_limit_per_minute)
    now = time.monotonic()
    bucket = _requests[request.client.host if request.client else "unknown"]
    while bucket and now - bucket[0] >= 60:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded; try again shortly")
    bucket.append(now)


def _prompt(question: str, context: str, interview_mode: bool) -> str:
    style = (
        "Keep the answer concise and interview-ready. Prefer a short direct answer followed by "
        "2-4 technical points when useful."
        if interview_mode
        else "Answer clearly and concisely."
    )
    return f"""You are KraVerse AI, Kartik Katke's portfolio assistant. {style}
Use ONLY the verified context below for personal and project facts. Do not invent missing details.
If the context is insufficient, say that you do not have enough verified information.

VERIFIED CONTEXT:
{context or '(No matching verified knowledge found.)'}

USER QUESTION:
{question}"""


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
async def chat_endpoint(request: Request, payload: ChatRequest) -> dict:
    _check_rate_limit(request)
    try:
        query_embedding = await ollama_embed(payload.message, settings.ollama_url, settings.embedding_model)
        results = store.search(query_embedding, payload.top_k or settings.top_k)
        context = "\n\n".join(f"SOURCE: {item['source']}\n{item['content']}" for item in results)
        prompt = _prompt(payload.message, context, payload.interview_mode)
        answer = await chat(prompt, settings.chat_model, settings.ollama_url)
        return {"answer": answer, "sources": [item["source"] for item in results], "mode": "local"}
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Local AI request failed") from exc


@app.post("/chat/online")
async def online_chat_endpoint(request: Request, payload: ChatRequest) -> dict:
    _check_rate_limit(request)
    if settings.llm_provider.lower() != "openrouter":
        raise HTTPException(status_code=503, detail=f"Unsupported online provider: {settings.llm_provider}")
    if not settings.llm_api_key:
        raise HTTPException(status_code=503, detail="Online provider is not configured")
    try:
        results = retrieve(payload.message, payload.top_k or settings.top_k)
        context = "\n\n".join(f"SOURCE: {item['source']}\n{item['content']}" for item in results)
        prompt = _prompt(payload.message, context, payload.interview_mode)
        answer = await chat_online(prompt, settings.llm_api_key, settings.llm_model, settings.llm_base_url)
        return {"answer": answer, "sources": [item["source"] for item in results], "mode": "online", "model": settings.llm_model}
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Online AI request failed") from exc
