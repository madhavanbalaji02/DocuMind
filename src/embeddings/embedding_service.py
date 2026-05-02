"""Unified embedding service supporting local sentence-transformers and BigRed200 vLLM."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embed texts via local sentence-transformers or remote vLLM endpoint."""

    def __init__(self, batch_size: int = 32) -> None:
        self.mode = os.environ.get("EMBEDDING_MODE", "local")
        self.batch_size = batch_size
        self._local_model: Any = None
        self._http_client: httpx.AsyncClient | None = None

        if self.mode == "local":
            local_model_name = os.environ.get(
                "LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5"
            )
            logger.info("Initializing local embedding model: %s", local_model_name)
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(local_model_name)
            logger.info(
                "Local model loaded (dim=%d)", self._local_model.get_sentence_embedding_dimension()
            )
        else:
            self._vllm_url = os.environ["BIGRED200_VLLM_URL"]
            self._embed_model = os.environ.get("BIGRED200_EMBED_MODEL", "BAAI/bge-m3")
            self._http_client = httpx.AsyncClient(timeout=120.0)
            logger.info(
                "BigRed200 vLLM embedding configured: %s / %s",
                self._vllm_url,
                self._embed_model,
            )

    @property
    def embedding_dim(self) -> int:
        """Return dimensionality of the embedding vectors."""
        if self.mode == "local":
            return self._local_model.get_sentence_embedding_dimension()
        return 1024  # bge-m3 default

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, processing in batches."""
        if not texts:
            return []

        t0 = time.perf_counter()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            if self.mode == "local":
                embeddings = await self._embed_local(batch)
            else:
                embeddings = await self._embed_remote(batch)
            all_embeddings.extend(embeddings)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Embedded %d texts in %.1f ms (mode=%s)",
            len(texts),
            elapsed,
            self.mode,
        )
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed_texts([query])
        return results[0]

    async def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Run sentence-transformers in a thread to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        embeddings: np.ndarray = await loop.run_in_executor(
            None, self._local_model.encode, texts
        )
        return embeddings.tolist()

    async def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        """POST to vLLM OpenAI-compatible /embeddings endpoint."""
        assert self._http_client is not None
        payload = {"model": self._embed_model, "input": texts}
        response = await self._http_client.post(
            f"{self._vllm_url}/embeddings", json=payload
        )
        response.raise_for_status()
        data = response.json()
        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def close(self) -> None:
        """Release HTTP client resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
