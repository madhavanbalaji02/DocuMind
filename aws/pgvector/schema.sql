-- DocuMind pgvector schema
-- Hierarchical parent-child chunking with vector search + FTS

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Documents ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    s3_key          TEXT UNIQUE NOT NULL,
    filename        TEXT NOT NULL,
    doc_type        TEXT,                          -- pdf|web|url
    page_count      INTEGER,
    textract_job_id TEXT,
    status          TEXT DEFAULT 'processing',     -- processing|ready|failed
    entities        JSONB,                         -- Comprehend entities
    key_phrases     TEXT[],                        -- top key phrases
    metadata        JSONB,
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ── Parent chunks (large sections — provide context, NOT retrieved directly) ──
CREATE TABLE IF NOT EXISTS parent_chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID REFERENCES documents(id) ON DELETE CASCADE,
    section_title   TEXT,
    element_type    TEXT,                          -- text|table|figure|mixed
    page_start      INTEGER,
    page_end        INTEGER,
    content         TEXT NOT NULL,                 -- full section text
    char_count      INTEGER,
    chunk_index     INTEGER,
    metadata        JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Child chunks (small, retrieved via similarity search) ─────────────────────
-- NOTE: embedding is vector(384) for bge-small-en-v1.5 (local, free, CPU)
--       If switching to Bedrock Titan Embed v2: change to vector(1536)
CREATE TABLE IF NOT EXISTS child_chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_id       UUID REFERENCES parent_chunks(id) ON DELETE CASCADE,
    document_id     UUID REFERENCES documents(id) ON DELETE CASCADE,
    content         TEXT NOT NULL,                 -- 150-300 token chunk
    embedding       vector(384),                   -- bge-small (384-dim, CPU fast)
    element_type    TEXT,                          -- text|table_row|figure_desc
    page_number     INTEGER,
    chunk_index     INTEGER,
    entities        JSONB,
    metadata        JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Indexes ───────────────────────────────────────────────────────────────────

-- IVFFlat vector index (fast, free, good for <1M chunks)
-- lists=100 is good for up to ~1M vectors
CREATE INDEX IF NOT EXISTS child_chunks_embedding_idx
ON child_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Filter by document
CREATE INDEX IF NOT EXISTS child_chunks_doc_idx
ON child_chunks (document_id);

-- Full-text search (replaces BM25 — built into Postgres, no extra dependency)
CREATE INDEX IF NOT EXISTS child_chunks_fts_idx
ON child_chunks USING gin(to_tsvector('english', content));

-- Document status lookup
CREATE INDEX IF NOT EXISTS documents_status_idx
ON documents (status);

-- ── Existing tables (keep from local Postgres schema) ─────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    topic           TEXT NOT NULL,
    status          TEXT DEFAULT 'pending',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS reports (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID REFERENCES sessions(id) ON DELETE CASCADE,
    title           TEXT,
    content         TEXT,
    citations       JSONB,
    quality_score   FLOAT,
    word_count      INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(session_id)
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID REFERENCES sessions(id) ON DELETE CASCADE,
    agent_name      TEXT,
    input_text      TEXT,
    output_text     TEXT,
    duration_ms     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retrievals (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID REFERENCES sessions(id) ON DELETE CASCADE,
    query           TEXT,
    chunk_id        TEXT,
    document_source TEXT,
    relevance_score FLOAT,
    content_preview TEXT,
    retrieved_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ── Evaluation results ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_results (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query             TEXT NOT NULL,
    answer            TEXT NOT NULL,
    contexts          TEXT[],
    ground_truth      TEXT,
    faithfulness      FLOAT,
    answer_relevancy  FLOAT,
    context_precision FLOAT,
    context_recall    FLOAT,
    composite_score   FLOAT,
    model_id          TEXT,
    latency_ms        INTEGER,
    cost_usd          FLOAT,
    evaluated_at      TIMESTAMPTZ DEFAULT NOW()
);
