#!/usr/bin/env python3
"""Load demo data into the knowledge base for offline demos.

Usage:
    python scripts/load_demo_data.py
"""

from __future__ import annotations

import asyncio
import time

from dotenv import load_dotenv

load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEMO_SOURCES = [
    ("https://en.wikipedia.org/wiki/Retrieval-augmented_generation", "web"),
    ("https://en.wikipedia.org/wiki/Large_language_model", "web"),
    ("https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)", "web"),
    ("https://en.wikipedia.org/wiki/Vector_database", "web"),
    ("https://en.wikipedia.org/wiki/Multi-agent_system", "web"),
]


async def main() -> None:
    from src.db import connection as db
    from src.ingest.pipeline import IngestionPipeline
    from qdrant_client import AsyncQdrantClient
    import os

    await db.run_migrations()
    pipeline = IngestionPipeline()

    print(f"\nLoading {len(DEMO_SOURCES)} demo sources into knowledge base…\n")
    t_start = time.perf_counter()
    total_chunks = 0
    results = []

    for i, (source, doc_type) in enumerate(DEMO_SOURCES, start=1):
        name = source.split("/wiki/")[-1].replace("_", " ")
        print(f"[{i}/{len(DEMO_SOURCES)}] {name}")
        t0 = time.perf_counter()
        try:
            result = await pipeline.ingest(source, doc_type)
            elapsed = int((time.perf_counter() - t0) * 1000)
            total_chunks += result.chunk_count
            status = "SKIPPED (already ingested)" if result.total_time_ms == 0 else f"{result.chunk_count} chunks in {elapsed}ms"
            print(f"         {status}")
            results.append((name, result.chunk_count, elapsed))
        except Exception as exc:
            print(f"         FAILED: {exc}")

    total_elapsed = time.perf_counter() - t_start

    # Get Qdrant collection size
    qdrant = AsyncQdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    collection_name = os.environ.get("QDRANT_COLLECTION", "documind_chunks")
    try:
        info = await qdrant.get_collection(collection_name)
        vector_count = info.vectors_count
    except Exception:
        vector_count = "unknown"
    finally:
        await qdrant.close()

    print(f"\n{'─'*50}")
    print(f"  Total chunks ingested : {total_chunks}")
    print(f"  Total time            : {total_elapsed:.1f}s")
    print(f"  Qdrant collection size: {vector_count} vectors")
    print(f"{'─'*50}")
    print("\nKnowledge base is ready. Try these demo queries:")
    print("  • What is retrieval augmented generation?")
    print("  • How do transformer architectures work?")
    print("  • What are vector databases used for?")
    print("  • Compare RAG with fine-tuning for LLMs")

    await pipeline.close()
    await db.close_pool()


if __name__ == "__main__":
    asyncio.run(main())
