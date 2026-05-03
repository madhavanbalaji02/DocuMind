"""Document ingestion API routes."""

from __future__ import annotations

from src.core.logging import get_logger
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

logger = get_logger(__name__)
router = APIRouter()


@router.post("")
async def ingest_document(
    url: str | None = Form(default=None),
    doc_type: str = Form(default="text"),
    file: UploadFile | None = File(default=None),
):
    """Ingest a document from a file upload or URL."""
    from src.ingest.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    try:
        if file is not None:
            # Write upload to a temp file, then ingest
            suffix = Path(file.filename or "upload").suffix or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            # Infer doc_type from extension if not specified
            ext = suffix.lower().lstrip(".")
            resolved_type = doc_type if doc_type != "text" else (
                "pdf" if ext == "pdf" else
                "csv" if ext == "csv" else
                "code" if ext in ("py", "js", "ts", "go", "rs", "java") else
                "text"
            )
            result = await pipeline.ingest(tmp_path, resolved_type)
            os.unlink(tmp_path)
            source = file.filename or tmp_path

        elif url is not None:
            result = await pipeline.ingest(url, doc_type if doc_type != "text" else "web")
            source = url

        else:
            raise HTTPException(status_code=400, detail="Provide either 'file' or 'url'")

        return {
            "document_id": result.document_id,
            "source": source,
            "chunk_count": result.chunk_count,
            "embed_time_ms": round(result.embed_time_ms),
            "total_time_ms": round(result.total_time_ms),
        }
    finally:
        await pipeline.close()
