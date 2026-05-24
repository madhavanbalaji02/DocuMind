"""Bedrock embedding client — wraps EmbeddingService for EMBEDDING_MODE=bedrock.

This module exists as a standalone Bedrock-specific embedder for use in
Lambda functions and evaluation scripts that prefer an explicit Bedrock client.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3

from aws.config import REGION
from aws.bedrock.model_registry import EMBEDDING_MODELS


class BedrockEmbeddings:
    """Embed texts via Bedrock Titan Embed v2 (1536-dim, ~$0.00002/1K tokens)."""

    MODEL_ID = EMBEDDING_MODELS["titan"]
    DIM = 1536

    def __init__(self) -> None:
        self._bedrock = boto3.client("bedrock-runtime", region_name=REGION)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Titan requires one call per text."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self._embed_one, texts))
        return results

    def embed_one(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text, "dimensions": self.DIM, "normalize": True})
        resp = self._bedrock.invoke_model(
            modelId=self.MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(resp["body"].read())["embedding"]


class LocalEmbeddings:
    """Embed texts via local sentence-transformers bge-small-en-v1.5 (384-dim, free)."""

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DIM = 384

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.MODEL_NAME)

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        import numpy as np
        embeddings: np.ndarray = self._model.encode(texts, batch_size=batch_size)
        return embeddings.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def get_embedder(mode: str | None = None):
    """Return the appropriate embedder based on EMBEDDING_MODE."""
    mode = mode or os.environ.get("EMBEDDING_MODE", "local")
    if mode == "bedrock":
        return BedrockEmbeddings()
    return LocalEmbeddings()
