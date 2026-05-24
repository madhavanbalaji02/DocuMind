"""Document management API routes — S3 upload, status, analytics."""

from __future__ import annotations

import io
import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from src.core.logging import get_logger
from src.db import connection as db

logger = get_logger(__name__)
router = APIRouter()


# ── S3 Ingest ─────────────────────────────────────────────────────────────────

@router.post("/s3")
async def ingest_to_s3(file: UploadFile = File(...)):
    """Upload file to S3 → triggers Lambda → SQS → Step Functions pipeline.

    Returns: {s3_key, job_id, status: "queued"}
    """
    from aws.s3.document_store import DocumentStore
    from aws.step_functions.executor import PipelineExecutor

    filename = file.filename or "upload.pdf"
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "bin"
    doc_type = {"pdf": "pdf", "txt": "text", "html": "web", "htm": "web"}.get(ext, "unknown")

    if doc_type == "unknown":
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")

    content = await file.read()
    s3_key = f"uploads/{uuid.uuid4().hex}_{filename}"

    try:
        store = DocumentStore()
        store.upload_bytes(content, s3_key, content_type=f"application/{ext}")
        logger.info("Uploaded to S3: %s (%d bytes)", s3_key, len(content))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    # Start Step Functions execution directly (Lambda trigger may also fire — idempotent)
    execution_arn = ""
    try:
        executor = PipelineExecutor()
        execution_arn = executor.start(
            s3_key=s3_key,
            filename=filename,
            doc_type=doc_type,
            bucket=store.bucket,
            size_bytes=len(content),
        )
        logger.info("Step Functions execution started: %s", execution_arn.split(":")[-1])
    except Exception as e:
        logger.warning("Could not start Step Functions (continuing): %s", e)

    return {
        "s3_key": s3_key,
        "filename": filename,
        "doc_type": doc_type,
        "size_bytes": len(content),
        "execution_arn": execution_arn,
        "status": "queued",
        "message": "Document queued for processing. Poll /documents/{s3_key}/status.",
    }


@router.post("/url")
async def ingest_url(url: str = Form(...), doc_type: str = Form(default="web")):
    """Download URL and store in S3, then trigger ingestion pipeline."""
    import httpx
    from aws.s3.document_store import DocumentStore
    from aws.step_functions.executor import PipelineExecutor

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content = resp.content
            content_type = resp.headers.get("content-type", "text/html")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    filename = url.rstrip("/").split("/")[-1] or "webpage.html"
    if "." not in filename:
        filename += ".html"
    s3_key = f"uploads/web/{uuid.uuid4().hex}_{filename}"

    store = DocumentStore()
    store.upload_bytes(content, s3_key, content_type=content_type)

    execution_arn = ""
    try:
        executor = PipelineExecutor()
        execution_arn = executor.start(
            s3_key=s3_key, filename=filename, doc_type=doc_type,
            bucket=store.bucket, size_bytes=len(content),
        )
    except Exception as e:
        logger.warning("Step Functions start failed: %s", e)

    return {
        "s3_key": s3_key,
        "url": url,
        "doc_type": doc_type,
        "size_bytes": len(content),
        "execution_arn": execution_arn,
        "status": "queued",
    }


# ── Document status ────────────────────────────────────────────────────────────

@router.get("")
async def list_documents(status: str | None = None, limit: int = 20):
    """List documents from pgvector documents table."""
    try:
        if status:
            rows = await db.fetch(
                "SELECT id::text, s3_key, filename, doc_type, status, page_count, ingested_at "
                "FROM documents WHERE status = $1 ORDER BY ingested_at DESC LIMIT $2",
                status, limit,
            )
        else:
            rows = await db.fetch(
                "SELECT id::text, s3_key, filename, doc_type, status, page_count, ingested_at "
                "FROM documents ORDER BY ingested_at DESC LIMIT $1",
                limit,
            )
        return {"documents": [dict(r) for r in rows], "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@router.get("/{s3_key:path}/status")
async def document_status(s3_key: str):
    """Get processing status for a document (from DynamoDB or pgvector)."""
    # Try DynamoDB first (real-time pipeline status)
    try:
        import boto3
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
        ddb = boto3.client("dynamodb", region_name=region)
        table = os.environ.get("DYNAMODB_TABLE", "documind-jobs")
        resp = ddb.get_item(
            TableName=table,
            Key={"s3_key": {"S": s3_key}},
        )
        if "Item" in resp:
            item = resp["Item"]
            return {
                "s3_key": s3_key,
                "status": item.get("status", {}).get("S", "unknown"),
                "chunk_count": int(item.get("chunk_count", {}).get("N", 0)),
                "source": "dynamodb",
            }
    except Exception:
        pass

    # Fall back to pgvector documents table
    try:
        row = await db.fetchrow(
            "SELECT status, page_count FROM documents WHERE s3_key = $1",
            s3_key,
        )
        if row:
            return {"s3_key": s3_key, "status": row["status"],
                    "page_count": row["page_count"], "source": "pgvector"}
    except Exception:
        pass

    raise HTTPException(status_code=404, detail=f"Document not found: {s3_key}")


@router.get("/{s3_key:path}/chunks")
async def document_chunks(s3_key: str):
    """Get chunk statistics for a document."""
    try:
        row = await db.fetchrow(
            """
            SELECT d.id::text, d.filename, d.page_count,
                   COUNT(DISTINCT pc.id) AS parent_count,
                   COUNT(DISTINCT cc.id) AS child_count,
                   d.entities
            FROM documents d
            LEFT JOIN parent_chunks pc ON pc.document_id = d.id
            LEFT JOIN child_chunks cc ON cc.document_id = d.id
            WHERE d.s3_key = $1
            GROUP BY d.id
            """,
            s3_key,
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"Document not found: {s3_key}")
        return dict(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ── Analytics ──────────────────────────────────────────────────────────────────

@router.get("/analytics")
async def analytics():
    """System-wide analytics: documents, chunks, costs, eval scores."""
    result: dict = {}

    # Document + chunk counts from pgvector
    try:
        row = await db.fetchrow(
            """
            SELECT
                (SELECT COUNT(*) FROM documents)                                 AS total_documents,
                (SELECT COUNT(*) FROM documents WHERE status='ready')            AS ready_documents,
                (SELECT COUNT(*) FROM parent_chunks)                             AS total_parent_chunks,
                (SELECT COUNT(*) FROM child_chunks)                              AS total_chunks,
                (SELECT COALESCE(SUM(page_count), 0) FROM documents)            AS textract_pages_processed
            """
        )
        result.update(dict(row) if row else {})
    except Exception as e:
        result["db_error"] = str(e)

    # Latest eval results
    try:
        eval_row = await db.fetchrow(
            """
            SELECT faithfulness, answer_relevancy, composite_score, evaluated_at
            FROM eval_results ORDER BY evaluated_at DESC LIMIT 1
            """
        )
        if eval_row:
            result["last_eval"] = dict(eval_row)
    except Exception:
        pass

    # Avg query latency from sessions
    try:
        session_row = await db.fetchrow(
            """
            SELECT
                COUNT(*) AS total_sessions,
                ROUND(AVG(EXTRACT(EPOCH FROM (completed_at - created_at)) * 1000)
                    FILTER (WHERE status='completed' AND completed_at IS NOT NULL))::int AS avg_workflow_ms
            FROM sessions
            """
        )
        if session_row:
            result.update(dict(session_row))
    except Exception:
        pass

    result["embedding_mode"] = os.environ.get("EMBEDDING_MODE", "local")
    result["llm_provider"] = os.environ.get("LLM_PROVIDER", "bedrock")
    result["cost_estimate_monthly_usd"] = 2.50  # ~$2.50/month per budget table

    return result
