"""HTTP client for the vLLM inference server running on BigRed200."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


class BigRed200Client:
    """Thin wrapper around the OpenAI-compatible vLLM endpoint on BigRed200."""

    def __init__(self) -> None:
        self.base_url = os.environ["BIGRED200_VLLM_URL"]
        self.model = os.environ.get("BIGRED200_EMBED_MODEL", "BAAI/bge-m3")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=120.0,
            headers={"Content-Type": "application/json"},
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Request embeddings from the vLLM server."""
        payload = {"model": self.model, "input": texts}
        logger.debug("BigRed200 embed request: %d texts", len(texts))
        response = await self._client.post("/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def health(self) -> bool:
        """Return True if the vLLM server responds to /health."""
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
