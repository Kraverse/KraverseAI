from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import RAGStore, chunk_text, embed  # noqa: E402

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
SUPPORTED = {".md", ".txt", ".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".yml", ".yaml"}


async def ingest(root: Path) -> None:
    store = RAGStore()
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED]
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for index, chunk in enumerate(chunk_text(text)):
            vector = await embed(chunk, OLLAMA_URL, EMBEDDING_MODEL)
            store.add(f"{path.as_posix()}#chunk-{index}", chunk, vector)
    print(f"Indexed {len(files)} files into {store.db_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/ingest.py <project-directory>")
    asyncio.run(ingest(Path(sys.argv[1]).resolve()))
