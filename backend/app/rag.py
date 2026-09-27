from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

import httpx


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


async def ollama_embed(text: str, url: str, model: str) -> list[float]:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(f"{url.rstrip('/')}/api/embed", json={"model": model, "input": text})
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError("Ollama returned no embeddings")
        return embeddings[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class RAGStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, source TEXT NOT NULL, content TEXT NOT NULL, embedding TEXT NOT NULL)")

    def add(self, chunk_id: str, source: str, content: str, embedding: list[float]) -> None:
        import json
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)", (chunk_id, source, content, json.dumps(embedding)))

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        import json
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT id, source, content, embedding FROM chunks").fetchall()
        ranked = []
        for chunk_id, source, content, raw in rows:
            score = cosine_similarity(query_embedding, json.loads(raw))
            ranked.append({"id": chunk_id, "source": source, "content": content, "score": score})
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]
