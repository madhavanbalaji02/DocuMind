# DocuMind

![CI](https://github.com/madhavanbalaji02/DocuMind/actions/workflows/ci.yml/badge.svg)
![RAG Eval](https://github.com/madhavanbalaji02/DocuMind/actions/workflows/rag_eval.yml/badge.svg)

**Enterprise-grade multi-agent RAG research platform** built on AWS. Submit a research topic; a LangGraph workflow orchestrates CrewAI agents (Researcher → Analyst → Writer → Critic) to produce a fully cited report. Documents are ingested via direct upload or URL, chunked, embedded with Bedrock Titan, and stored in pgvector on RDS — all behind a polished dark Streamlit dashboard served over CloudFront HTTPS.

**Live:** `https://d2ivfev3xe6utf.cloudfront.net`

---

## System Design Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                    │
│                                                                          │
│   Browser ──► CloudFront (HTTPS) ──► nginx (EC2 t3.micro)               │
│   Claude Desktop ──► MCP Server (stdio)                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      EC2 t3.micro        │
                    │  ┌───────────────────┐  │
                    │  │  Streamlit :8501  │  │
                    │  │  (Dark UI / HTTPS)│  │
                    │  └────────┬──────────┘  │
                    │           │ httpx        │
                    │  ┌────────▼──────────┐  │
                    │  │  FastAPI  :8888   │  │
                    │  │  + WebSocket /ws  │  │
                    │  └────────┬──────────┘  │
                    │           │              │
                    │  ┌────────▼──────────┐  │
                    │  │  Qdrant  :6333    │  │
                    │  │  (local vector DB)│  │
                    │  └───────────────────┘  │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                       │
          ▼                      ▼                       ▼
┌─────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│  INGESTION PATH │   │  RESEARCH PATH   │   │   STORAGE LAYER      │
│                 │   │                  │   │                      │
│ Upload File/URL │   │ POST /research   │   │ RDS PostgreSQL 15    │
│       │         │   │       │          │   │ + pgvector extension │
│       ▼         │   │       ▼          │   │                      │
│ DocumentLoader  │   │  LangGraph DAG   │   │ Tables:              │
│  ├─ pymupdf     │   │  ┌────────────┐  │   │  documents           │
│  ├─ BeautifulSoup│  │  │plan_research│ │   │  sessions            │
│  └─ PDF URL     │   │  │retrieve_ctx│  │   │  agent_runs          │
│  auto-detect    │   │  │analyze_data│  │   │  retrievals          │
│       │         │   │  │run_crew    │  │   │  reports             │
│       ▼         │   │  │critic_loop │  │   │                      │
│ SemanticChunker │   │  │finalize    │  │   │ AWS S3               │
│  512 tok/64 ovlp│   │  └─────┬──────┘  │   │  document store      │
│       │         │   │        │          │   │  eval reports        │
│       ▼         │   │        ▼          │   │                      │
│ EmbeddingService│   │   CrewAI Crew     │   │ DynamoDB             │
│  Bedrock Titan  │   │  ┌─────────────┐ │   │  job status tracking │
│  Embed v2       │   │  │ Researcher  │ │   │                      │
│  1024-dim       │   │  │ Analyst     │ │   │ SQS                  │
│       │         │   │  │ Writer      │ │   │  ingestion event bus │
│       ▼         │   │  │ Critic      │ │   └──────────────────────┘
│ Qdrant upsert   │   │  └──────┬──────┘ │
│ + Postgres write│   │         │         │
└─────────────────┘   │         ▼         │
                       │  HybridRetriever │
                       │  ┌─────────────┐ │
                       │  │ Dense search│ │
                       │  │ (pgvector)  │ │
                       │  │ BM25 sparse │ │
                       │  │ RRF fusion  │ │
                       │  │ CrossEncoder│ │
                       │  │ reranker    │ │
                       │  └──────┬──────┘ │
                       │         │         │
                       │         ▼         │
                       │  Bedrock Claude   │
                       │  Haiku (LLM)      │
                       │  + Guardrails     │
                       └──────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE AWS PIPELINE (Step Functions)               │
│                                                                          │
│  S3 upload ──► Lambda s3-trigger ──► SQS ──► Step Functions             │
│                                              │                           │
│                              ┌───────────────┼────────────────────┐     │
│                              ▼               ▼                    ▼     │
│                         Stage 1:        Stage 2:            Stage 3:    │
│                         Textract        Bedrock Titan       pgvector    │
│                         parse PDF       embed chunks        store +     │
│                         (layout-aware)  1024-dim            Comprehend  │
│                         AnalyzeDocument                     entities    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What's Built

### Core Platform
- **LangGraph** stateful research workflow with critic→revise loop (max 2 iterations), human-in-the-loop approval via `POST /research/{id}/approve`
- **CrewAI** 4-agent crew: Researcher → Analyst → Writer → Critic running on Bedrock Claude Haiku
- **Hybrid RAG**: dense pgvector search + BM25 sparse search fused via Reciprocal Rank Fusion (RRF k=60), reranked by CrossEncoder `ms-marco-MiniLM-L-6-v2`
- **MCP Server** with 6 tools for Claude Desktop integration

### Document Ingestion
- **Direct upload** (file or URL) through `/ingest` — synchronous, ~400ms for typical docs
- **Auto PDF detection** from URL Content-Type headers (arXiv, research papers)
- **pymupdf** PDF parsing (10x faster than pypdf), capped at 50 pages
- **Bedrock Titan Embed v2** at 1024-dim — no local GPU needed
- **Idempotent** ingestion: SHA-256 chunk IDs, source deduplication in Postgres

### Infrastructure & Hosting
- **EC2 t3.micro** running FastAPI + Streamlit + nginx + Qdrant (Docker), with 2GB swap
- **CloudFront HTTPS** CDN in front of EC2 — `AllViewerExceptHostHeader` origin policy
- **nginx** with gzip compression, 250MB upload limit, static asset caching (24h TTL), WebSocket proxying
- **RDS PostgreSQL 15** with pgvector on db.t3.micro (free tier)
- **systemd** services for API and UI with auto-restart
- **uvicorn** `--limit-max-requests 500` to auto-recycle worker and prevent memory bloat

### UI Dashboard
- Dark Streamlit theme (`#0d1117` background, `#238636` primary)
- `@st.fragment(run_every=N)` non-blocking live agent feed
- Plotly gauge charts for evaluation metrics, pie chart for pipeline status
- Report download, tabbed session viewer, sortable session table
- Sidebar health auto-refresh every 30s

### Enterprise AWS Pipeline
- **S3** document store with versioning and lifecycle rules
- **SQS** ingestion queue with DLQ (3 retries)
- **Step Functions** 3-stage state machine: Textract parse → Bedrock embed → pgvector store
- **Lambda** functions: s3-trigger, stage1-parse, stage2-embed, stage3-store
- **Amazon Textract** layout-aware PDF parsing (tables, figures, reading order)
- **Amazon Comprehend** entity + key phrase extraction stored as JSONB metadata
- **AWS Glue** catalog for batch ETL on large document sets
- **SSM Parameter Store** for secrets management
- **CloudWatch** billing alarm at $50

---

## AWS Resources

| Service | Resource | Notes |
|---|---|---|
| EC2 | t3.micro (us-east-2) | FastAPI + Streamlit + nginx + Qdrant |
| RDS | db.t3.micro PostgreSQL 15 | pgvector, free tier — stop when not demoing |
| S3 | documind-documents-574772738370 | Document store + eval reports |
| SQS | documind-ingestion-queue | Event bus, DLQ after 3 failures |
| Step Functions | documind-ingestion-pipeline | 3-stage STANDARD state machine |
| DynamoDB | documind-jobs | PAY_PER_REQUEST, job status tracking |
| Lambda | stage1/2/3 | Triggered by S3 ObjectCreated |
| Textract | Per-document | AnalyzeDocument for tables, DetectDocumentText for text |
| Comprehend | Per-document | Entity + key phrase extraction |
| Bedrock | Claude Haiku + Titan Embed v2 | LLM + embeddings backbone |
| CloudFront | E1UNZSUNNCCIFV | HTTPS CDN, static asset caching |
| CloudWatch | Billing alarm | Cost guard at $50 |
| Glue | documind_catalog | Batch ETL |

---

## Cost Budget

| Service | Estimated Monthly | Notes |
|---|---|---|
| EC2 t3.micro | $0 | AWS Free Tier (12 months) |
| RDS db.t3.micro | $0 | AWS Free Tier — stop when not demoing |
| S3 | ~$0.02 | Minimal storage |
| Bedrock Haiku | ~$2.00 | 500 research runs × $0.004 avg |
| Bedrock Titan Embed v2 | ~$0.10 | ~5M tokens/month |
| Textract | ~$0.15 | 10 docs/month × $0.015/page |
| Comprehend | ~$0.05 | Entity extraction |
| CloudFront | ~$0.00 | 1TB free/month |
| SQS + Lambda | ~$0.00 | Within free tier |
| **Total** | **~$2.32/month** | |

> Stop RDS when not demoing: `aws rds stop-db-instance --db-instance-identifier documind-db`

---

## Tech Stack

| Technology | Role |
|---|---|
| LangGraph | Stateful research workflow with conditional critic→revise loop |
| CrewAI + LiteLLM | 4-agent collaboration (Researcher, Analyst, Writer, Critic) |
| Amazon Bedrock | Claude Haiku (LLM) + Titan Embed v2 (1024-dim embeddings) |
| FastAPI | Async REST API + WebSocket streaming |
| Streamlit | Dark-themed dashboard with live fragments |
| asyncpg | Native async PostgreSQL driver |
| pgvector | Vector similarity search on RDS PostgreSQL 15 |
| Qdrant | Local vector DB for dev/fallback |
| pymupdf | Fast PDF text extraction (10x faster than pypdf) |
| sentence-transformers | CrossEncoder reranker (ms-marco-MiniLM-L-6-v2) |
| BM25 | Sparse retrieval fused with dense via RRF |
| Amazon Textract | Layout-aware PDF parsing (tables, figures, reading order) |
| Amazon Comprehend | Entity + key phrase extraction |
| S3 + SQS + Step Functions | Event-driven 3-stage enterprise ingestion pipeline |
| CloudFront + nginx | HTTPS CDN, gzip, static caching, WebSocket proxying |
| MCP SDK | Claude Desktop integration (6 tools) |
| RAGAS | RAG evaluation framework |
| Pydantic Guardrails | Hallucination + citation + length validation |

---

## Local Development

```bash
# 1. Clone
git clone https://github.com/madhavanbalaji02/DocuMind && cd DocuMind

# 2. Start local infrastructure
docker compose up -d        # Qdrant :6333, Postgres :5432, Redis :6379

# 3. Install dependencies
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Set ANTHROPIC_API_KEY, AWS credentials, DATABASE_URL

# 5. Load demo knowledge base
make demo

# 6. Run smoke test
make smoke

# 7. Start API + Dashboard
make api   # FastAPI on :8888
make ui    # Streamlit on :8501
```

API docs: `http://localhost:8888/docs`  
Dashboard: `http://localhost:8501`

---

## AWS Provisioning & Deploy

```bash
# Provision all AWS resources (idempotent)
python aws/setup.py

# Provision EC2 instance
make provision          # python scripts/provision_ec2.py

# Deploy to EC2
make deploy             # rsync + systemd restart

# Run RAGAS evaluation
python evaluation/eval_runner.py --n-samples 10 --fail-below 0.60

# Teardown all resources (irreversible, prompts first)
python aws/teardown.py
```

---

## API Reference

### Ingest a document

```bash
# File upload
curl -X POST http://localhost:8888/ingest \
  -F "file=@paper.pdf"

# URL (auto-detects PDF vs HTML)
curl -X POST http://localhost:8888/ingest \
  -F "url=https://arxiv.org/pdf/1706.03762" \
  -F "doc_type=web"
```

### Start research

```bash
curl -X POST http://localhost:8888/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "Attention mechanisms in transformers", "user_id": "alice"}'
```

### Health check

```bash
curl http://localhost:8888/health
# {"status":"ok","qdrant_ok":true,"postgres_ok":true,"pgvector_ok":true,"embedding_mode":"bedrock","llm_provider":"bedrock"}
```

### Analytics

```bash
curl http://localhost:8888/documents/analytics
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
        "DATABASE_URL_ASYNC": "postgresql+asyncpg://documind:...@documind-db.XXXXX.us-east-2.rds.amazonaws.com:5432/documind?ssl=true",
        "EMBEDDING_MODE": "bedrock",
        "LLM_PROVIDER": "bedrock"
      }
    }
  }
}
```

Available tools: `search_knowledge_base`, `ingest_document`, `run_research`, `get_research_status`, `get_report`, `list_documents`

---

## Performance

| Metric | Value | Notes |
|---|---|---|
| PDF parse (pymupdf) | ~176ms | 15-page paper on EC2 t3.micro |
| URL fetch + parse | ~500ms | Including HTTP download |
| Bedrock Titan embed | ~100ms/batch | 4 parallel threads, 1024-dim |
| Full ingest (typical) | ~400–800ms | Chunk + embed + Qdrant + Postgres |
| Full ingest (arxiv PDF) | ~3.5s | 15 pages, 92 chunks |
| RAG retrieve p50 | ~15ms | pgvector IVFFlat + BM25 hybrid |
| Bedrock Haiku latency | ~1–3s | Per LLM call |
| End-to-end research | ~10–20 min | 4-agent CrewAI crew |
| RAGAS target | ≥0.60 | CI fail threshold |

---

## Project Structure

```
documind/
├── src/
│   ├── api/             # FastAPI app, routes (ingest, research, documents, sessions)
│   │   └── websocket.py # Real-time job status streaming
│   ├── ingest/          # DocumentLoader (pymupdf/BS4), SemanticChunker, IngestionPipeline
│   ├── embeddings/      # EmbeddingService (Bedrock Titan v2 / local bge-small)
│   ├── rag/             # HybridRetriever (dense+BM25+RRF), CrossEncoderReranker, RAGChain
│   ├── agents/          # CrewAI researcher/analyst/writer/critic factories
│   ├── workflow/        # LangGraph ResearchState, nodes, graph builder
│   ├── mcp_server/      # MCP stdio server + 6 tool registrations
│   ├── db/              # asyncpg pool, migrations (001_initial.sql)
│   ├── workers/         # SQS ingestion worker daemon thread
│   ├── core/            # Logging (JSON), Qdrant client factory
│   └── ui/              # Streamlit dark dashboard
├── aws/
│   ├── setup.py         # Idempotent AWS resource provisioner
│   ├── teardown.py      # Destroy all resources
│   ├── config.py        # Central constants + cost documentation
│   ├── bedrock/         # BedrockLLM, Titan embeddings, model registry
│   ├── comprehend/      # Entity + key phrase extraction
│   ├── ec2/             # EC2 provisioner + bootstrap.sh
│   ├── glue/            # Batch ETL for large document sets
│   ├── lambda_functions/# s3-trigger, stage1-parse, stage2-embed, stage3-store
│   ├── pgvector/        # PgVectorClient, HybridRetriever, schema.sql
│   ├── s3/              # DocumentStore
│   ├── sqs/             # IngestionQueue
│   ├── ssm/             # SecretsManager wrapper
│   ├── step_functions/  # pipeline.json + PipelineExecutor
│   └── textract/        # Layout parser, table extractor, multimodal extractor
├── scripts/
│   ├── provision_ec2.py # Launch + configure EC2 instance
│   ├── deploy_to_ec2.sh # rsync + remote init + service restart
│   └── ec2_init.sh      # Remote: venv, Docker, nginx, systemd setup
├── evaluation/
│   ├── eval_runner.py   # RAGAS-style evaluation CLI
│   ├── guardrails.py    # Pydantic answer validation
│   └── ground_truth.jsonl
├── tests/
├── .github/workflows/   # CI (pytest) + RAG eval on RAG/workflow changes
├── .streamlit/config.toml # Dark theme, fast reruns, 200MB upload
├── docker-compose.yml   # Local dev: Qdrant + Postgres + Redis
├── Dockerfile
├── Makefile             # api, ui, mcp, test, smoke, bench, demo, provision, deploy
└── pyproject.toml
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `API error (/health): timed out` | EC2 memory pressure after large PDF | SSH in, `sudo systemctl restart documind-api` |
| `ModuleNotFoundError: plotly` | Not installed in venv | `pip install plotly` in EC2 venv |
| PDF upload fails silently | nginx `client_max_body_size` too small | Already set to 250m |
| Qdrant health red | Docker container stopped to save RAM | `docker start tmp-qdrant-1` on EC2 |
| `Vector dimension error` | Qdrant collection dim mismatch | `curl -X DELETE localhost:6333/collections/documind_chunks` then restart API |
| `ssl` error on asyncpg | SSL params in both DSN and kwarg | Strip `?ssl=...` from DSN before passing to asyncpg |
