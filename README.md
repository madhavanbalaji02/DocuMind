# DocuMind

![CI](https://github.com/madhavanbalaji02/DocuMind/actions/workflows/ci.yml/badge.svg)
![RAG Eval](https://github.com/madhavanbalaji02/DocuMind/actions/workflows/rag_eval.yml/badge.svg)

Enterprise-grade Layout-Aware Multi-Modal RAG research platform. Submit a topic; LangGraph orchestrates a stateful workflow; CrewAI agents (Researcher, Analyst, Writer, Critic) collaborate to produce a cited report; results persist in AWS RDS (pgvector); documents are ingested via S3 → SQS → Step Functions with Amazon Textract layout-aware parsing.

---

## AWS Architecture

```
User / Claude Desktop
       │
       ▼
  MCP Server  ◄──►  FastAPI + Streamlit (EC2 t2.micro)
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
        LangGraph Workflow            S3 → SQS → Step Functions
        ┌──────────────────┐          ┌──────────────────────────┐
        │ plan_research     │          │ Stage1: Textract parse    │
        │ retrieve_context  │          │ Stage2: bge-small embed   │
        │ analyze_data      │          │ Stage3: pgvector store    │
        │ run_crew          │          └──────────────────────────┘
        │ critic_review     │
        │ finalize_report   │
        └──────────────────┘
              │
              ▼
     CrewAI (Bedrock Haiku)
     Researcher → Analyst
         Writer → Critic
              │
              ▼
       RAG Pipeline
     ┌────────────────────────────┐
     │ PgVectorHybridRetriever    │
     │  Dense  (pgvector IVFFlat) │
     │  Sparse (Postgres FTS)     │
     │  Fusion (RRF k=60)         │
     │  Rerank (CrossEncoder)     │
     │  Parent context injection  │
     └────────────────────────────┘
              │
              ▼
     AWS RDS PostgreSQL 15 + pgvector
     (documind-db, db.t3.micro, us-east-2)
```

---

## AWS Deployment

| Service | Resource | Notes |
|---|---|---|
| EC2 | t2.micro (us-east-2) | FREE TIER — FastAPI + Streamlit + nginx |
| RDS | db.t3.micro PostgreSQL 15 | pgvector extension, FREE TIER (stop when not demoing) |
| S3 | documind-documents-574772738370 | Document store + eval reports |
| SQS | documind-ingestion-queue | Ingestion event bus (DLQ after 3 failures) |
| Step Functions | documind-ingestion-pipeline | 3-stage STANDARD state machine |
| DynamoDB | documind-jobs | PAY_PER_REQUEST, job status tracking |
| Lambda | stage1-parse, stage2-embed, stage3-store | Triggered by S3 ObjectCreated |
| Textract | Per-document | async AnalyzeDocument / DetectDocumentText |
| Comprehend | Per-document | entity + key phrase extraction |
| Bedrock | Claude Haiku 4.5 | LLM backbone for all agents and RAG |
| CloudWatch | Billing alarm at $50 | Cost guard |
| Glue | documind_catalog | Batch ETL for >10 doc bulk loads |

---

## Cost Budget

| Service | Estimated Monthly | Notes |
|---|---|---|
| EC2 t2.micro | $0 | AWS Free Tier (12 months) |
| RDS db.t3.micro | $0 | AWS Free Tier — STOP when not demoing |
| S3 | ~$0.02 | 50 documents, minimal storage |
| Bedrock Haiku | ~$2.00 | 500 research runs × $0.004 avg |
| Textract | ~$0.15 | 10 docs/month × $0.015/page |
| Comprehend | ~$0.05 | Entity extraction |
| SQS + Lambda | ~$0.00 | Well within free tier |
| **Total** | **~$2.22/month** | Budget: $200 credit |

> Stop RDS between demos: `aws rds stop-db-instance --db-instance-identifier documind-db`

---

## Tech Stack

| Technology | Role | Why Chosen |
|---|---|---|
| LangGraph | Workflow orchestration | Stateful, checkpointable DAG with conditional loops |
| CrewAI + LiteLLM | Multi-agent collaboration | Declarative agents; LiteLLM routes to Bedrock |
| Amazon Bedrock | LLM backbone | Claude Haiku — best cost/quality, no API key mgmt |
| AWS RDS + pgvector | Vector + relational DB | Unified store: vectors, sessions, reports |
| Amazon Textract | Layout-aware PDF parsing | Tables, figures, reading order; async polling |
| Amazon Comprehend | Entity extraction | NER + key phrases stored as JSONB metadata |
| S3 + SQS + Step Functions | Event-driven ingestion | Serverless, retryable 3-stage pipeline |
| FastAPI | REST API | Async-native, auto-docs, WebSocket support |
| Streamlit | Dashboard UI | Rapid prototyping, no frontend build step |
| MCP SDK | Claude Desktop integration | First-class tool protocol for LLM clients |
| sentence-transformers | Local embeddings + reranking | bge-small-en-v1.5 (384-dim), CPU-only, free |
| asyncpg | PostgreSQL client | Native async, fastest Python Postgres driver |
| Pydantic Guardrails | Answer validation | Hallucination + citation + length checks |
| RAGAS | RAG evaluation | Faithfulness, answer relevancy, context precision |
| EC2 + nginx | Reverse proxy | / → FastAPI:8000, /ui → Streamlit:8501 |

---

## Prerequisites

- Docker Desktop (for local Qdrant + Postgres + Redis dev)
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- AWS credentials with access to Bedrock, S3, SQS, RDS, Step Functions
- Anthropic API key (fallback when Bedrock unavailable)

---

## Local Development

```bash
# 1. Clone and enter project
git clone https://github.com/madhavanbalaji02/DocuMind && cd DocuMind

# 2. Start local infrastructure (Postgres + Qdrant + Redis)
docker compose up -d

# 3. Install dependencies
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and AWS credentials

# 5. Load demo knowledge base
python scripts/load_demo_data.py

# 6. Run smoke test
python scripts/smoke_test.py

# 7. Start API + Dashboard
uvicorn src.api.main:app --reload &
streamlit run src/ui/app.py
```

API docs: http://localhost:8000/docs  
Dashboard: http://localhost:8501

---

## AWS Provisioning

```bash
# Provision all AWS resources (idempotent)
python aws/setup.py

# Migrate existing Qdrant data to pgvector
python aws/pgvector/migrate.py

# Deploy to EC2
python aws/ec2/setup_ec2.py

# Run RAGAS evaluation
python evaluation/eval_runner.py --n-samples 10 --fail-below 0.60

# TEARDOWN when not demoing (preserves RDS free tier hours)
python aws/teardown.py
```

---

## Ingestion Pipeline

Documents flow through a 3-stage serverless pipeline:

```
Upload PDF/URL
      │
      ▼
  S3 ObjectCreated
      │
      ▼  Lambda s3-trigger
  SQS message
      │
      ▼  Step Functions
  Stage1: Textract parse (layout-aware, async)
      │    ├── SECTION_HEADER, PARAGRAPH, TABLE, FIGURE
      │    └── BoundingBox, reading_order, confidence
      ▼
  Stage2: Hierarchical chunking + bge-small embed
      │    ├── Parent chunks: 600–1200 tokens (context for retrieval)
      │    └── Child chunks: 100–300 tokens (vector(384), 20% overlap)
      ▼
  Stage3: Comprehend entities + pgvector store
           └── CloudWatch metrics: DocumentsProcessed, ChunksIndexed
```

Cost guard: max 50 pages per document (`TEXTRACT_MAX_PAGES=50`).

---

## API Reference

### Start research

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "Attention mechanisms in transformer models", "user_id": "alice"}'
```

### Upload document (S3 pipeline)

```bash
curl -X POST http://localhost:8000/documents/s3 \
  -F "file=@paper.pdf"
```

### Ingest from URL

```bash
curl -X POST http://localhost:8000/documents/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/abs/1706.03762", "doc_type": "web"}'
```

### Document status

```bash
curl http://localhost:8000/documents/{s3_key}/status
```

### Analytics

```bash
curl http://localhost:8000/documents/analytics
```

### Health check

```bash
curl http://localhost:8000/health
# Returns: {"status":"healthy","pgvector_ok":true,"llm_provider":"bedrock"}
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
        "AWS_DEFAULT_REGION": "us-east-2",
        "DATABASE_URL_ASYNC": "postgresql://documind:...@documind-db.XXXXX.us-east-2.rds.amazonaws.com:5432/documind",
        "EMBEDDING_MODE": "local",
        "LLM_PROVIDER": "bedrock"
      }
    }
  }
}
```

Available MCP tools: `search_knowledge_base`, `ingest_document`, `run_research`, `get_research_status`, `get_report`, `list_documents`

---

## Evaluation

The CI pipeline runs RAGAS-style evaluation on every change to `src/rag/`, `src/agents/`, or `src/workflow/nodes.py`:

```bash
python evaluation/eval_runner.py --n-samples 30 --fail-below 0.60
```

Metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`  
Report saved to S3: `s3://documind-documents-574772738370/evaluations/{timestamp}/report.json`

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/test_rag.py -v       # RAG pipeline
pytest tests/test_workflow.py -v  # LangGraph nodes
pytest tests/test_agents.py -v    # CrewAI tools
```

---

## Performance

| Metric | Value | Notes |
|---|---|---|
| Embedding throughput | ~167 texts/sec | bge-small-en-v1.5, M-series Mac CPU |
| Embed latency p50 | ~9.4 ms | Local sentence-transformers |
| RAG retrieve p50 | ~15 ms | pgvector IVFFlat + FTS hybrid |
| Bedrock Haiku latency | ~1–3 sec | Claude Haiku 4.5 |
| End-to-end research | ~10–20 min | 4-agent CrewAI crew |
| RAGAS composite (target) | ≥0.60 | CI fail threshold |

---

## Project Structure

```
documind/
├── src/
│   ├── ingest/          # document loading, hierarchical chunking, pipeline
│   ├── embeddings/      # local bge-small + Bedrock Titan embedding service
│   ├── rag/             # hybrid retriever, reranker, RAG chain + guardrails
│   ├── agents/          # CrewAI researcher/analyst/writer/critic (Bedrock LLM)
│   ├── workflow/        # LangGraph state, nodes, graph
│   ├── mcp_server/      # MCP server + tool registrations
│   ├── api/             # FastAPI app, routes (research + documents)
│   ├── db/              # asyncpg pool, migrations
│   ├── workers/         # SQS ingestion worker daemon
│   └── ui/              # Streamlit dashboard (Upload, Status, Evaluation)
├── aws/
│   ├── setup.py         # Idempotent AWS resource provisioner
│   ├── teardown.py      # Destroy all resources
│   ├── config.py        # Central constants + boto3 helpers
│   ├── bedrock/         # BedrockLLM client + embeddings + model registry
│   ├── comprehend/      # Entity + key phrase extraction
│   ├── ec2/             # EC2 provisioner + bootstrap.sh
│   ├── glue/            # Batch ETL for large document sets
│   ├── lambda_functions/# s3-trigger, stage1-parse, stage2-embed, stage3-store
│   ├── pgvector/        # PgVectorClient, HybridRetriever, schema.sql, migrate.py
│   ├── s3/              # DocumentStore
│   ├── sqs/             # IngestionQueue
│   ├── ssm/             # SecretsManager
│   ├── step_functions/  # pipeline.json + PipelineExecutor
│   └── textract/        # TextractLayoutParser + TableToMarkdown + MultiModalExtractor
├── evaluation/
│   ├── eval_runner.py   # RAGAS-style evaluation CLI
│   ├── guardrails.py    # Pydantic answer validation
│   └── ground_truth.jsonl  # 30 Q&A pairs
├── tests/
├── .github/workflows/   # CI + RAG eval
├── docker-compose.yml   # LOCAL DEV ONLY
├── Dockerfile           # EC2 container image
├── pyproject.toml
└── .env.example
```
