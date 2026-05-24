"""Qdrant client factory — HTTP mode when QDRANT_URL is set, embedded otherwise."""

from __future__ import annotations

import os

from qdrant_client import AsyncQdrantClient


def make_async_qdrant_client() -> AsyncQdrantClient:
    url = os.environ.get("QDRANT_URL", "")
    if url.startswith("http"):
        return AsyncQdrantClient(
            url=url,
            api_key=os.environ.get("QDRANT_API_KEY") or None,
        )
    path = os.environ.get("QDRANT_LOCAL_PATH", "/data/qdrant")
    return AsyncQdrantClient(path=path)
