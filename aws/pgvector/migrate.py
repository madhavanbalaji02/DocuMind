"""Migrate existing Qdrant data → pgvector RDS.

Usage: python aws/pgvector/migrate.py

Prerequisites:
  - Local Docker Qdrant running (docker compose up -d)
  - RDS instance ready (aws rds describe-db-instances --db-instance-identifier documind-db)
  - DATABASE_URL env var set to RDS connection string

Migration steps:
  1. Apply pgvector schema to RDS
  2. Scroll all points from local Qdrant
  3. For each point: create synthetic parent (group by page_number) + insert child
  4. Verify counts match
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()


async def migrate() -> None:
    console.print("\n[bold cyan]DocuMind: Qdrant → pgvector Migration[/bold cyan]\n")

    # Apply pgvector schema first
    from aws.pgvector.client import PgVectorClient, DocumentRecord, ParentChunk, ChildChunk
    pg = PgVectorClient()

    console.print("Applying pgvector schema …")
    pg.apply_schema()
    console.print("[green]✓[/green] Schema applied\n")

    # Connect to local Qdrant
    from qdrant_client import AsyncQdrantClient
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    collection = os.environ.get("QDRANT_COLLECTION", "documind_chunks")

    console.print(f"Connecting to Qdrant: {qdrant_url} / {collection}")
    qdrant = AsyncQdrantClient(url=qdrant_url)

    try:
        info = await qdrant.get_collection(collection)
        total_points = info.points_count or 0
    except Exception as e:
        console.print(f"[red]Cannot connect to Qdrant: {e}[/red]")
        console.print("Make sure Docker is running: docker compose up -d")
        return

    console.print(f"[green]✓[/green] Qdrant: {total_points} points in {collection!r}\n")

    if total_points == 0:
        console.print("[yellow]No points to migrate. Qdrant collection is empty.[/yellow]")
        return

    # Group points by source document
    doc_id_map: dict[str, str] = {}  # source → pgvector document UUID
    parent_id_map: dict[str, str] = {}  # (source, page) → parent UUID
    migrated = 0
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Migrating chunks …", total=total_points)

        offset = None
        while True:
            result, next_offset = await qdrant.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not result:
                break

            for point in result:
                payload = point.payload or {}
                text = payload.get("text", "")
                source = payload.get("source", "unknown")
                doc_type = payload.get("doc_type", "text")
                page_number = payload.get("page_number", 1) or 1
                chunk_index = payload.get("chunk_index", 0)
                vector = point.vector

                if not text or not vector:
                    progress.advance(task)
                    continue

                try:
                    # Get or create pgvector document record
                    if source not in doc_id_map:
                        filename = source.split("/")[-1] if "/" in source else source
                        doc = DocumentRecord(
                            s3_key=source,
                            filename=filename,
                            doc_type=doc_type,
                            status="ready",
                        )
                        doc_id = pg.upsert_document(doc)
                        doc_id_map[source] = doc_id

                    doc_id = doc_id_map[source]

                    # Get or create synthetic parent (one per page)
                    parent_key = f"{source}::{page_number}"
                    if parent_key not in parent_id_map:
                        parent = ParentChunk(
                            document_id=doc_id,
                            section_title=f"Page {page_number}",
                            element_type="text",
                            page_start=page_number,
                            page_end=page_number,
                            content=text,  # synthetic parent = chunk text
                            chunk_index=chunk_index,
                        )
                        parent_id = pg.upsert_parent_chunk(parent)
                        parent_id_map[parent_key] = parent_id

                    parent_id = parent_id_map[parent_key]

                    # Insert child chunk with embedding
                    child = ChildChunk(
                        parent_id=parent_id,
                        document_id=doc_id,
                        content=text,
                        element_type="text",
                        page_number=page_number,
                        chunk_index=chunk_index,
                    )
                    pg.upsert_child_chunk(child, list(vector) if vector else [])
                    migrated += 1

                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        console.print(f"  [red]Error migrating {str(point.id)[:8]}: {e}[/red]")

                progress.advance(task)

            if next_offset is None:
                break
            offset = next_offset

    await qdrant.close()

    # Verify
    counts = pg.get_counts()
    console.print(f"\n[bold green]Migration complete.[/bold green]")
    console.print(f"  Migrated: {migrated} chunks ({errors} errors)")
    console.print(f"\n  pgvector counts:")
    console.print(f"    documents:    {counts['documents']}")
    console.print(f"    parent_chunks:{counts['parent_chunks']}")
    console.print(f"    child_chunks: {counts['child_chunks']}")

    if counts["child_chunks"] > 0:
        ratio = migrated / counts["child_chunks"] if counts["child_chunks"] else 0
        console.print(f"\n[green]✓ Verification: {counts['child_chunks']} chunks in pgvector[/green]")
    else:
        console.print("[yellow]Warning: child_chunks count is 0 — check database connection.[/yellow]")


if __name__ == "__main__":
    asyncio.run(migrate())
