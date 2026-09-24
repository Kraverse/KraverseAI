from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Iterable

import httpx


class RAGStore:
    def __init__(self, db_path: str = "data/kraverse.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, source TEXT NOT NULL, content TEXT NOT NULL, embedding TEXT NOT NULL)"
            )

    def add(self, source: str, content: str, embedding: list[float]) -> None:
        encoded = ",".join(str(value) for value in embedding)
        with sqlite3.connect(self.db_path) as db:
            db.execute("DELETE FROM documents WHERE source = ? AND content = ?", (source, content))
            db.execute("INSERT INTO documents(source, content, embedding) VALUES (?, ?, ?)", (source, content, encoded))

    def search(self, query: list[float], limit: int = 5) -> list[dict[str, str | float]]:
        results: list[dict[str, str | float]] = []
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute("SELECT source, content, embedding FROM documents").fetchall()
        for source, content, encoded in rows:
            embedding = [float(value) for value in encoded.split(",") if value]
            score = cosine_similarity(query, embedding)
            results.append({"source": source, "content": content, "score": score})
        return sorted(results, key=lambda item: float(item["score"]), reverse=True)[:limit]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> Iterable[str]:
    text = text.strip()
    if not text:
        return
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        yield text[start:end]
        if end == len(text):
            break
        start = end - overlap


async def embed(text: str, ollama_url: str, model: str) -> list[float]:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(f"{ollama_url}/api/embed", json={"model": model, "input": text})
        response.raise_for_status()
        data = response.json()
    embeddings = data.get("embeddings") or []
    if not embeddings:
        raise RuntimeError("Ollama returned no embedding")
    return embeddings[0]
