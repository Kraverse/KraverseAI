from __future__ import annotations

import httpx


async def chat(prompt: str, model: str, url: str) -> str:
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{url.rstrip('/')}/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
        )
        response.raise_for_status()
        message = response.json().get("message", {})
        return message.get("content", "").strip()
