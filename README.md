# DocuMind

A production-grade multi-agent research and intelligence platform. Users submit a research topic; LangGraph orchestrates a stateful workflow; CrewAI agents (Researcher, Analyst, Writer, Critic) collaborate to produce a cited report; results persist in PostgreSQL; the entire system is accessible via a custom MCP server; embeddings optionally run on BigRed200 GPU via vLLM.

---

## Architecture

```
User / Claude Desktop
       │
       ▼
  MCP Server  ◄──►  FastAPI (REST + WebSocket)
       │                    │
       ▼                    ▼
  LangGraph            Streamlit UI
  Workflow
  ┌──────────────────────────────────┐
  │ plan → retrieve → analyze → crew │
  │       ↕ critic loop (max 2x)     │
  │            → finalize            │
  └──────────────────────────────────┘
       │                    │
       ▼                    ▼
  CrewAI Crew          PostgreSQL
  Researcher           (sessions, reports,
  Analyst               agent_runs, retrievals)
  Writer
  Critic
       │
       ▼
  RAG Pipeline
  ┌────────────────────────┐
  │ HybridRetriever        │
  │  Dense  (Qdrant/vLLM)  │
  │  Sparse (BM25)         │
  │  Fusion (RRF)          │
  │  Rerank (CrossEncoder) │
  └────────────────────────┘
       │
       ▼
  Qdrant Vector DB
  Embeddings: local bge-small / BigRed200 bge-m3
```

---

## Tech Stack

| Technology | Role | Why Chosen |
|---|---|---|
| LangGraph | Workflow orchestration | Stateful, checkpointable DAG with conditional loops |
| CrewAI | Multi-agent collaboration | Declarative agent/task/crew abstraction |
| Qdrant | Vector database | High-performance ANN search, rich payload filtering |
| PostgreSQL | Persistent state | ACID guarantees for sessions, reports, audit trail |
| FastAPI | REST API | Async-native, auto-docs, WebSocket support |
| Streamlit | Dashboard UI | Rapid prototyping, no frontend build step |
| MCP SDK | Claude Desktop integration | First-class tool protocol for LLM clients |
| sentence-transformers | Local embeddings + reranking | Offline, GPU-optional, battle-tested |
| vLLM (BigRed200) | GPU embedding server | High-throughput batched inference on A100 |
| Anthropic Claude | LLM backbone | Best-in-class reasoning and instruction following |
| asyncpg | PostgreSQL client | Native async, fastest Python Postgres driver |
| rank-bm25 | Sparse retrieval | Keyword recall complement to dense search |
| Redis | Future caching layer | Rate limiting and session caching |

---

## Prerequisites

- Docker Desktop (for Qdrant, Postgres, Redis)
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Anthropic API key
- *(Optional)* BigRed200 account for GPU embeddings

---

## Quick Start

```bash
# 1. Clone and enter project
git clone <repo-url> && cd documind

# 2. Start infrastructure
docker compose up -d

# 3. Install dependencies
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY at minimum

# 5. Start the API
uvicorn src.api.main:app --reload
# Streamlit dashboard
streamlit run src/ui/app.py
```

API docs: http://localhost:8000/docs  
Dashboard: http://localhost:8501

---

## BigRed200 Setup

### 1. SSH to BigRed200

```bash
ssh <username>@bigred200.indiana.edu
cd /N/project/<project_name>
git clone <repo-url> documind && cd documind
uv venv && source .venv/bin/activate
uv pip install -e .
```

### 2. Start the vLLM embedding server

```bash
sbatch jobs/vllm_server.slurm
squeue -u $USER          # note the node name, e.g. r003n15
```

### 3. Create SSH tunnel (from your local machine)

```bash
ssh -N -L 8000:r003n15:8000 <username>@bigred200.indiana.edu
```

### 4. Update local `.env`

```
EMBEDDING_MODE=bigred200
BIGRED200_VLLM_URL=http://localhost:8000/v1
```

### 5. Batch ingest documents on BigRed200

```bash
sbatch --export=INPUT_DIR=/N/project/data/papers,COLLECTION_NAME=documind_chunks \
       jobs/embed_batch.slurm
```

---

## API Reference

### Start research

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "Attention mechanisms in transformer models", "user_id": "alice"}'
# → {"session_id": "...", "status": "pending"}
```

### Poll status

```bash
curl http://localhost:8000/research/<session_id>
```

### Get final report

```bash
curl http://localhost:8000/research/<session_id>/report
```

### Ingest a PDF

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@paper.pdf"
```

### Ingest a URL

```bash
curl -X POST http://localhost:8000/ingest \
  -F "url=https://arxiv.org/abs/1706.03762" \
  -F "doc_type=web"
```

### List sessions

```bash
curl "http://localhost:8000/sessions?limit=10"
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## MCP Setup (Claude Desktop)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "documind": {
      "command": "/path/to/documind/.venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/path/to/documind",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "QDRANT_URL": "http://localhost:6333",
        "DATABASE_URL": "postgresql://documind:documind_secret@localhost:5432/documind",
        "DATABASE_URL_ASYNC": "postgresql+asyncpg://documind:documind_secret@localhost:5432/documind",
        "EMBEDDING_MODE": "local"
      }
    }
  }
}
```

Available MCP tools: `search_knowledge_base`, `ingest_document`, `run_research`, `get_research_status`, `get_report`, `list_documents`

---

## Example Research Queries

```
"Explain the evolution of attention mechanisms from Bahdanau to Flash Attention"
"What are the key differences between RLHF, RLAIF, and DPO for LLM alignment?"
"Summarise recent advances in protein structure prediction after AlphaFold2"
"Compare RAG architectures: naive RAG vs advanced RAG vs modular RAG"
"What are the main techniques for reducing hallucination in large language models?"
```

---

## Performance Benchmarks

| Metric | Local (bge-small) | BigRed200 (bge-m3) |
|---|---|---|
| Embedding throughput | TBD chunks/s | TBD chunks/s |
| RAG latency (p50) | TBD ms | TBD ms |
| End-to-end research | TBD min | TBD min |
| Report quality score | TBD | TBD |

*Fill in after running benchmarks with `jobs/embed_batch.slurm` and timing `POST /research`.*

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/test_rag.py -v       # RAG pipeline
pytest tests/test_workflow.py -v  # LangGraph nodes
pytest tests/test_agents.py -v    # CrewAI tools
```

---

## Project Structure

```
documind/
├── src/
│   ├── ingest/          # document loading, chunking, pipeline
│   ├── embeddings/      # local + BigRed200 vLLM embedding service
│   ├── rag/             # hybrid retriever, reranker, RAG chain
│   ├── agents/          # CrewAI researcher/analyst/writer/critic + tools
│   ├── workflow/        # LangGraph state, nodes, graph, checkpointer
│   ├── mcp_server/      # MCP server + tool registrations
│   ├── api/             # FastAPI app, routes, WebSocket
│   ├── db/              # asyncpg pool, Pydantic models, migrations
│   └── ui/              # Streamlit dashboard
├── jobs/
│   ├── vllm_server.slurm
│   └── embed_batch.slurm
├── tests/
│   ├── test_rag.py
│   ├── test_workflow.py
│   └── test_agents.py
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```
