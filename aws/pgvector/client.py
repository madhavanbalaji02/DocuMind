"""pgvector client for DocuMind hierarchical RAG storage.

Uses psycopg2 connection pool (3 connections max — free tier safe).
DSN from SSM: /documind/rds-connection-string or DATABASE_URL env var.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json

from aws.config import REGION

RRF_K = 60  # Reciprocal Rank Fusion constant (same as existing retriever)

# Module-level connection pool (lazily initialized)
_pool: pool.ThreadedConnectionPool | None = None


def _get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        dsn = _get_dsn()
        _pool = pool.ThreadedConnectionPool(1, 3, dsn)
    return _pool


def _get_dsn() -> str:
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return dsn
    # Try SSM
    try:
        import boto3
        ssm = boto3.client("ssm", region_name=REGION)
        resp = ssm.get_parameter(Name="/documind/rds-connection-string", WithDecryption=True)
        return resp["Parameter"]["Value"]
    except Exception:
        raise RuntimeError(
            "Cannot find database connection. Set DATABASE_URL env var "
            "or populate /documind/rds-connection-string in SSM."
        )


@dataclass
class DocumentRecord:
    s3_key: str
    filename: str
    doc_type: str = "pdf"
    page_count: int = 0
    textract_job_id: str = ""
    status: str = "processing"
    entities: dict = field(default_factory=dict)
    key_phrases: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ParentChunk:
    document_id: str
    section_title: str
    element_type: str  # text|table|figure|mixed
    page_start: int
    page_end: int
    content: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ChildChunk:
    parent_id: str
    document_id: str
    content: str
    element_type: str  # text|table_row|figure_desc
    page_number: int
    chunk_index: int
    entities: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    document_id: str
    s3_key: str
    filename: str
    element_type: str
    page_number: int
    score: float
    rank: int
    parent_content: str = ""
    entities: dict = field(default_factory=dict)


class PgVectorClient:
    """CRUD and search operations against the pgvector RDS instance."""

    def __init__(self) -> None:
        self._pool = _get_pool()

    def _conn(self):
        return self._pool.getconn()

    def _release(self, conn) -> None:
        self._pool.putconn(conn)

    # ── Schema ────────────────────────────────────────────────────────────────

    def apply_schema(self) -> None:
        """Apply schema.sql to the database."""
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        sql = open(schema_path).read()
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        finally:
            self._release(conn)

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_document(self, doc: DocumentRecord) -> str:
        """Insert or update document record. Returns UUID."""
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (s3_key, filename, doc_type, page_count,
                        textract_job_id, status, entities, key_phrases, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (s3_key) DO UPDATE SET
                        status = EXCLUDED.status,
                        page_count = EXCLUDED.page_count,
                        textract_job_id = EXCLUDED.textract_job_id,
                        entities = EXCLUDED.entities,
                        key_phrases = EXCLUDED.key_phrases
                    RETURNING id::text
                """, (
                    doc.s3_key, doc.filename, doc.doc_type, doc.page_count,
                    doc.textract_job_id, doc.status,
                    Json(doc.entities), doc.key_phrases, Json(doc.metadata),
                ))
                doc_id = cur.fetchone()[0]
            conn.commit()
            return doc_id
        finally:
            self._release(conn)

    def update_document_status(self, s3_key: str, status: str) -> None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE documents SET status=%s WHERE s3_key=%s", (status, s3_key))
            conn.commit()
        finally:
            self._release(conn)

    def upsert_parent_chunk(self, chunk: ParentChunk) -> str:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO parent_chunks
                        (document_id, section_title, element_type, page_start, page_end,
                         content, char_count, chunk_index, metadata)
                    VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id::text
                """, (
                    chunk.document_id, chunk.section_title, chunk.element_type,
                    chunk.page_start, chunk.page_end, chunk.content,
                    len(chunk.content), chunk.chunk_index, Json(chunk.metadata),
                ))
                parent_id = cur.fetchone()[0]
            conn.commit()
            return parent_id
        finally:
            self._release(conn)

    def upsert_child_chunk(self, chunk: ChildChunk, embedding: list[float]) -> str:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                # Format vector as Postgres array string
                vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
                cur.execute("""
                    INSERT INTO child_chunks
                        (parent_id, document_id, content, embedding,
                         element_type, page_number, chunk_index, entities, metadata)
                    VALUES (%s::uuid, %s::uuid, %s, %s::vector,
                            %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING id::text
                """, (
                    chunk.parent_id, chunk.document_id, chunk.content, vec_str,
                    chunk.element_type, chunk.page_number, chunk.chunk_index,
                    Json(chunk.entities), Json(chunk.metadata),
                ))
                row = cur.fetchone()
                child_id = row[0] if row else ""
            conn.commit()
            return child_id
        finally:
            self._release(conn)

    # ── Search ────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        document_id: str | None = None,
        element_type: str | None = None,
    ) -> list[SearchResult]:
        """Dense vector similarity search (cosine distance)."""
        vec_str = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
        filters = []
        params: list[Any] = [vec_str, top_k]
        if document_id:
            filters.append(f"cc.document_id = ${len(params)+1}::uuid")
            params.append(document_id)
        if element_type:
            filters.append(f"cc.element_type = ${len(params)+1}")
            params.append(element_type)
        where = ("WHERE " + " AND ".join(filters)) if filters else ""

        sql = f"""
            SELECT
                cc.id::text        AS chunk_id,
                cc.content,
                cc.document_id::text,
                d.s3_key,
                d.filename,
                cc.element_type,
                cc.page_number,
                1 - (cc.embedding <=> $1::vector) AS score,
                pc.content         AS parent_content,
                cc.entities
            FROM child_chunks cc
            JOIN documents d ON d.id = cc.document_id
            LEFT JOIN parent_chunks pc ON pc.id = cc.parent_id
            {where}
            ORDER BY cc.embedding <=> $1::vector
            LIMIT $2
        """
        return self._execute_search(sql, params)

    def full_text_search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Postgres FTS using gin index (replaces BM25, same recall, no extra dep)."""
        sql = """
            SELECT
                cc.id::text        AS chunk_id,
                cc.content,
                cc.document_id::text,
                d.s3_key,
                d.filename,
                cc.element_type,
                cc.page_number,
                ts_rank(to_tsvector('english', cc.content),
                        plainto_tsquery('english', %s)) AS score,
                pc.content         AS parent_content,
                cc.entities
            FROM child_chunks cc
            JOIN documents d ON d.id = cc.document_id
            LEFT JOIN parent_chunks pc ON pc.id = cc.parent_id
            WHERE to_tsvector('english', cc.content) @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
        """
        return self._execute_search(sql, [query, query, top_k])

    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """RRF fusion of dense + FTS results, returning top_k with parent context."""
        dense = self.similarity_search(query_embedding, top_k=20)
        sparse = self.full_text_search(query, top_k=20)

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        by_id: dict[str, SearchResult] = {}
        for rank, r in enumerate(dense, 1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            by_id[r.chunk_id] = r
        for rank, r in enumerate(sparse, 1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            if r.chunk_id not in by_id:
                by_id[r.chunk_id] = r

        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
        results: list[SearchResult] = []
        for final_rank, cid in enumerate(sorted_ids, 1):
            r = by_id[cid]
            r.score = scores[cid]
            r.rank = final_rank
            results.append(r)
        return results

    def _execute_search(self, sql: str, params: list) -> list[SearchResult]:
        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            results: list[SearchResult] = []
            for i, row in enumerate(rows):
                results.append(SearchResult(
                    chunk_id=row["chunk_id"],
                    content=row["content"],
                    document_id=row["document_id"],
                    s3_key=row.get("s3_key", ""),
                    filename=row.get("filename", ""),
                    element_type=row.get("element_type", "text"),
                    page_number=row.get("page_number", 0),
                    score=float(row.get("score", 0)),
                    rank=i + 1,
                    parent_content=row.get("parent_content", "") or "",
                    entities=row.get("entities") or {},
                ))
            return results
        finally:
            self._release(conn)

    def health_check(self) -> bool:
        try:
            conn = self._conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            self._release(conn)
            return True
        except Exception:
            return False

    def get_counts(self) -> dict[str, int]:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        (SELECT COUNT(*) FROM documents)     AS documents,
                        (SELECT COUNT(*) FROM parent_chunks) AS parent_chunks,
                        (SELECT COUNT(*) FROM child_chunks)  AS child_chunks
                """)
                row = cur.fetchone()
            return {
                "documents": row[0],
                "parent_chunks": row[1],
                "child_chunks": row[2],
            }
        finally:
            self._release(conn)

    def insert_eval_result(self, result: dict) -> None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO eval_results
                        (query, answer, contexts, ground_truth, faithfulness,
                         answer_relevancy, context_precision, context_recall,
                         composite_score, model_id, latency_ms, cost_usd)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    result.get("query"), result.get("answer"),
                    result.get("contexts", []), result.get("ground_truth"),
                    result.get("faithfulness"), result.get("answer_relevancy"),
                    result.get("context_precision"), result.get("context_recall"),
                    result.get("composite_score"), result.get("model_id"),
                    result.get("latency_ms"), result.get("cost_usd"),
                ))
            conn.commit()
        finally:
            self._release(conn)
