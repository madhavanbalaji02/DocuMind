---
title: DocuMind
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# DocuMind

A production-grade multi-agent research and intelligence platform.

Submit a research topic → LangGraph orchestrates a workflow → CrewAI agents (Researcher, Analyst, Writer, Critic) collaborate → cited report delivered.

## Required Secret

Set `ANTHROPIC_API_KEY` in the Space **Settings → Variables and secrets** tab before starting.

## Architecture

- **Streamlit UI** — port 7860 (public)
- **FastAPI backend** — port 8888 (internal)
- **Qdrant** vector DB — port 6333 (internal)
- **PostgreSQL 15** — port 5432 (internal)
- **Embeddings** — `BAAI/bge-small-en-v1.5` (local, cached at build time)
