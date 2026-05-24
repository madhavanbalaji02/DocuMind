-- SQLite schema — ephemeral per-session, no RDS needed

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT DEFAULT 'anon',
    topic TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    agent_name TEXT,
    input TEXT,
    output TEXT,
    tokens_used INTEGER,
    duration_ms INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS retrievals (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    query TEXT,
    chunk_id TEXT,
    document_source TEXT,
    relevance_score REAL,
    content_preview TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id TEXT DEFAULT 'anon',
    filename TEXT,
    source_url TEXT,
    doc_type TEXT,
    chunk_count INTEGER,
    ingested_at TEXT DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    session_id TEXT UNIQUE REFERENCES sessions(id),
    title TEXT,
    content TEXT,
    citations TEXT DEFAULT '[]',
    quality_score REAL,
    word_count INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_agent_runs_session ON agent_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_retrievals_session ON retrievals(session_id);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_url);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id)
