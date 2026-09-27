from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.rag import RAGStore, chunk_text, ollama_embed

SUPPORTED = {".md", ".txt", ".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".yml", ".yaml", ".html", ".css"}
IGNORED = {".git", "node_modules", ".next", "venv", ".venv", "__pycache__", "dist", "build"}


async def ingest(root: Path) -> None:
    store = RAGStore(settings.database_path)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED and not any(part in IGNORED for part in p.parts)]
    chunks = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for index, chunk in enumerate(chunk_text(text)):
            vector = await ollama_embed(chunk, settings.ollama_url, settings.embedding_model)
            store.add(f"{path.as_posix()}#chunk-{index}", str(path.relative_to(root)), chunk, vector)
            chunks += 1
    print(f"Indexed {len(files)} files / {chunks} chunks into {settings.database_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/ingest.py <project-directory>")
    asyncio.run(ingest(Path(sys.argv[1]).resolve()))
