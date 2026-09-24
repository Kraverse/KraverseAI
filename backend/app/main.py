from os import getenv

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OLLAMA_URL = getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = getenv("OLLAMA_MODEL", "llama3.2:3b")

app = FastAPI(title="KraVerse AI API", version="0.1.0")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    answer: str
    model: str
    mode: str = "local"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local"}


@app.get("/health/ollama")
async def ollama_health() -> dict[str, str | bool]:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
        return {"available": True, "model": OLLAMA_MODEL}
    except (httpx.HTTPError, httpx.RequestError) as exc:
        return {"available": False, "model": OLLAMA_MODEL, "error": str(exc)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are KraVerse AI, a personal AI assistant. "
                    "Answer clearly and never invent personal or project facts. "
                    "Knowledge retrieval will be added in the next phase."
                ),
            },
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

    return ChatResponse(answer=answer, model=OLLAMA_MODEL)
