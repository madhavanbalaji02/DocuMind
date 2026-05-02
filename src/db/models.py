"""Pydantic models mirroring PostgreSQL table rows."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Session(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: str | None = None
    topic: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRun(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID
    agent_name: str
    input: str
    output: str
    tokens_used: int | None = None
    duration_ms: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Retrieval(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID
    query: str
    chunk_id: str
    document_source: str
    relevance_score: float
    content_preview: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Document(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    filename: str | None = None
    source_url: str | None = None
    doc_type: str
    chunk_count: int
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Report(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID
    title: str
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    quality_score: float | None = None
    word_count: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
