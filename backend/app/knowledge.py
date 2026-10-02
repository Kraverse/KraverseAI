from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = ROOT / "knowledge"


def _chunks(text: str, size: int = 900) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= size:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def retrieve(query: str, limit: int = 5) -> list[dict[str, str]]:
    terms = set(re.findall(r"[a-z0-9][a-z0-9+#.-]*", query.lower()))
    results: list[dict[str, str | int]] = []
    for path in KNOWLEDGE_DIR.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for chunk in _chunks(text):
            chunk_terms = set(re.findall(r"[a-z0-9][a-z0-9+#.-]*", chunk.lower()))
            score = len(terms & chunk_terms)
            if score:
                results.append({"source": str(path.relative_to(ROOT)), "content": chunk, "score": score})
    results.sort(key=lambda item: int(item["score"]), reverse=True)
    return [{"source": str(item["source"]), "content": str(item["content"])} for item in results[:limit]]
