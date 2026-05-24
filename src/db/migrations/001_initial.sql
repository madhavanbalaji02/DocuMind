-- Research sessions
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    topic TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);
-- Ensure columns exist even if table was created by pgvector schema migration
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id VARCHAR(255);
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}';

-- pgvector schema adds s3_key NOT NULL but direct-upload pipeline doesn't use S3
ALTER TABLE documents ALTER COLUMN s3_key DROP NOT NULL;
ALTER TABLE documents ALTER COLUMN filename DROP NOT NULL;

-- Individual agent runs within a session
CREATE TABLE IF NOT EXISTS agent_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    agent_name VARCHAR(100),
    input TEXT,
    output TEXT,
    tokens_used INTEGER,
    duration_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Retrieved document chunks used in research
CREATE TABLE IF NOT EXISTS retrievals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    query TEXT,
    chunk_id VARCHAR(255),
    document_source TEXT,
    relevance_score FLOAT,
    content_preview TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Ingested documents
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(500),
    source_url TEXT,
    doc_type VARCHAR(50),
    chunk_count INTEGER,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Final research reports
CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) UNIQUE,
    title TEXT,
    content TEXT,
    citations JSONB DEFAULT '[]',
    quality_score FLOAT,
    word_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_session ON agent_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_retrievals_session ON retrievals(session_id);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_url);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
