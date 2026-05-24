"""Unified embedding service: local sentence-transformers (free) or Bedrock Titan v2."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)

# Module-level cache: SentenceTransformer is thread-safe once loaded.
_MODEL_CACHE: dict[str, Any] = {}


class EmbeddingService:
    """Embed texts via local sentence-transformers (primary) or Bedrock Titan v2 (fallback).

    EMBEDDING_MODE=local  → bge-small-en-v1.5, 384-dim, free, CPU-only
    EMBEDDING_MODE=bedrock → Titan Embed Text v2, 1536-dim, ~$0.00002/1K tokens
    """

    def __init__(self, batch_size: int = 32) -> None:
        self.mode = os.environ.get("EMBEDDING_MODE", "local")
        self.batch_size = batch_size
        self._local_model: Any = None
        self._bedrock_client: Any = None

        if self.mode == "local":
            local_model_name = os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
            if local_model_name not in _MODEL_CACHE:
                logger.info("Loading embedding model: %s", local_model_name)
                from sentence_transformers import SentenceTransformer
                _MODEL_CACHE[local_model_name] = SentenceTransformer(local_model_name)
                dim = (
                    _MODEL_CACHE[local_model_name].get_embedding_dimension()
                    if hasattr(_MODEL_CACHE[local_model_name], "get_embedding_dimension")
                    else _MODEL_CACHE[local_model_name].get_sentence_embedding_dimension()
                )
                logger.info("Embedding model cached (dim=%d)", dim)
            self._local_model = _MODEL_CACHE[local_model_name]
        elif self.mode == "bedrock":
            import boto3
            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
            self._bedrock_client = boto3.client("bedrock-runtime", region_name=region)
            logger.info("Bedrock Titan Embed v2 configured (region=%s)", region)
        else:
            raise ValueError(f"Unknown EMBEDDING_MODE: {self.mode!r}. Use 'local' or 'bedrock'.")

    @property
    def embedding_dim(self) -> int:
        if self.mode == "local":
            if hasattr(self._local_model, "get_embedding_dimension"):
                return self._local_model.get_embedding_dimension()
            return self._local_model.get_sentence_embedding_dimension()
        return 1024  # Bedrock Titan Embed v2 (valid: 256, 512, 1024)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0 = time.perf_counter()
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            if self.mode == "local":
                embeddings = await self._embed_local(batch)
            else:
                embeddings = await self._embed_bedrock(batch)
            all_embeddings.extend(embeddings)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Embedded %d texts in %.1f ms (mode=%s)", len(texts), elapsed, self.mode)
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        results = await self.embed_texts([query])
        return results[0]

    async def _embed_local(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        embeddings: np.ndarray = await loop.run_in_executor(
            None, self._local_model.encode, texts
        )
        return embeddings.tolist()

    async def _embed_bedrock(self, texts: list[str]) -> list[list[float]]:
        """Call Bedrock Titan Embed v2 — one text per request (Titan limitation)."""
        import json
        from concurrent.futures import ThreadPoolExecutor

        model_id = "amazon.titan-embed-text-v2:0"

        def _call_one(text: str) -> list[float]:
            body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
            response = self._bedrock_client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            return result["embedding"]

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [loop.run_in_executor(executor, _call_one, t) for t in texts]
            return list(await asyncio.gather(*futures))

    async def close(self) -> None:
        pass  # no persistent connections to close
