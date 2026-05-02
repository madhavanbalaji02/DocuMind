"""DocuMind Streamlit dashboard."""

from __future__ import annotations

import asyncio
import json
import os
import time

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def api_get(path: str, **params) -> dict | list | None:
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{API_BASE}{path}", params=params)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def api_post(path: str, json_body: dict) -> dict | None:
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{API_BASE}{path}", json=json_body)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_research():
    st.header("Research")
    st.write("Submit a topic and track the multi-agent research workflow in real time.")

    topic = st.text_input("Research Topic", placeholder="e.g., Transformer architectures in NLP")
    user_id = st.text_input("User ID (optional)", value="demo_user")

    if st.button("Start Research", type="primary", disabled=not topic.strip()):
        with st.spinner("Submitting research request…"):
            result = api_post("/research", {"topic": topic.strip(), "user_id": user_id or None})
        if result:
            st.session_state["session_id"] = result["session_id"]
            st.success(f"Session started: `{result['session_id']}`")

    if "session_id" in st.session_state:
        session_id = st.session_state["session_id"]
        st.divider()
        st.subheader(f"Session: `{session_id}`")

        status_placeholder = st.empty()
        report_placeholder = st.empty()

        # Poll until terminal state
        with st.status("Running research workflow…", expanded=True) as status_box:
            terminal_statuses = {"completed", "failed"}
            poll_attempts = 0
            max_attempts = 240  # 2-minute timeout at 0.5s intervals

            while poll_attempts < max_attempts:
                data = api_get(f"/research/{session_id}")
                if data:
                    current_status = data.get("status", "unknown")
                    agent_count = data.get("agent_run_count", 0)
                    status_box.write(
                        f"**Status:** {current_status} | **Agent runs:** {agent_count}"
                    )
                    if current_status in terminal_statuses:
                        if current_status == "completed":
                            status_box.update(label="Research complete!", state="complete")
                        else:
                            status_box.update(label="Research failed", state="error")
                        break
                time.sleep(2)
                poll_attempts += 1

        # Show final report
        report_data = api_get(f"/research/{session_id}/report")
        if report_data:
            st.subheader(report_data.get("title", "Research Report"))
            st.markdown(report_data.get("content", ""))

            citations = report_data.get("citations")
            if citations:
                if isinstance(citations, str):
                    try:
                        citations = json.loads(citations)
                    except Exception:
                        citations = []
                if citations:
                    with st.expander(f"Citations ({len(citations)})", expanded=False):
                        for i, cit in enumerate(citations, start=1):
                            st.markdown(
                                f"**[{i}]** `{cit.get('source', 'unknown')}` — {cit.get('excerpt', '')[:200]}"
                            )

        # Agent run breakdown
        agents_data = api_get(f"/sessions/{session_id}/agents")
        if agents_data:
            with st.expander("Agent Run Details", expanded=False):
                for run in agents_data:
                    with st.expander(
                        f"{run['agent_name']} — {run.get('duration_ms', 0)} ms", expanded=False
                    ):
                        st.code(run.get("output", "")[:2000], language="markdown")


def page_ingest():
    st.header("Ingest Documents")
    st.write("Add PDFs, CSVs, web pages, or text files to the knowledge base.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload File")
        uploaded_file = st.file_uploader(
            "Choose a file", type=["pdf", "csv", "txt", "py", "js", "ts", "md"]
        )
        if uploaded_file and st.button("Ingest File"):
            with st.spinner(f"Ingesting {uploaded_file.name}…"):
                try:
                    with httpx.Client(timeout=120.0) as client:
                        resp = client.post(
                            f"{API_BASE}/ingest",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                        )
                        resp.raise_for_status()
                        result = resp.json()
                    st.success(
                        f"Ingested **{uploaded_file.name}**\n"
                        f"- Chunks: {result['chunk_count']}\n"
                        f"- Embed time: {result['embed_time_ms']} ms\n"
                        f"- Total time: {result['total_time_ms']} ms"
                    )
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    with col2:
        st.subheader("Ingest URL")
        url_input = st.text_input("URL", placeholder="https://example.com/paper.pdf")
        doc_type = st.selectbox("Document type", ["web", "pdf", "text"])
        if url_input and st.button("Ingest URL"):
            with st.spinner(f"Fetching and ingesting {url_input}…"):
                try:
                    with httpx.Client(timeout=120.0) as client:
                        resp = client.post(
                            f"{API_BASE}/ingest",
                            data={"url": url_input, "doc_type": doc_type},
                        )
                        resp.raise_for_status()
                        result = resp.json()
                    st.success(
                        f"Ingested **{url_input}**\n"
                        f"- Chunks: {result['chunk_count']}\n"
                        f"- Embed time: {result['embed_time_ms']} ms"
                    )
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")


def page_sessions():
    st.header("Past Research Sessions")

    data = api_get("/sessions", limit=50, offset=0)
    if not data:
        return

    sessions = data.get("sessions", [])
    total = data.get("total", 0)
    st.write(f"Total sessions: **{total}**")

    if not sessions:
        st.info("No research sessions yet. Go to the Research page to start one.")
        return

    rows = [
        {
            "ID": s["id"][:8] + "…",
            "Topic": s["topic"][:60],
            "Status": s["status"],
            "Agent Runs": s["agent_run_count"],
            "Created": str(s["created_at"])[:16],
        }
        for s in sessions
    ]
    st.dataframe(rows, use_container_width=True)

    st.divider()
    selected_id = st.selectbox(
        "View session details",
        options=[s["id"] for s in sessions],
        format_func=lambda x: x[:8] + "…",
    )
    if selected_id and st.button("Load Session"):
        report_data = api_get(f"/research/{selected_id}/report")
        if report_data:
            st.subheader(report_data.get("title", "Report"))
            st.markdown(report_data.get("content", ""))
        else:
            session_data = api_get(f"/research/{selected_id}")
            st.json(session_data)

        agents_data = api_get(f"/sessions/{selected_id}/agents")
        if agents_data:
            with st.expander("Agent Runs", expanded=False):
                for run in agents_data:
                    st.markdown(f"**{run['agent_name']}** — {run.get('duration_ms', 'N/A')} ms")
                    st.code(str(run.get("output", ""))[:1000])


def page_knowledge_base():
    st.header("Knowledge Base")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Qdrant Stats")
        health = api_get("/health")
        if health:
            st.metric("Qdrant", "Online" if health["qdrant_ok"] else "Offline")
            st.metric("Postgres", "Online" if health["postgres_ok"] else "Offline")
            st.metric("Embedding Mode", health.get("embedding_mode", "N/A"))

    with col2:
        st.subheader("Search Knowledge Base")
        query = st.text_input("Search query", placeholder="What is attention mechanism?")
        if query and st.button("Search"):
            with st.spinner("Searching…"):
                try:
                    with httpx.Client(timeout=60.0) as client:
                        resp = client.post(
                            f"{API_BASE}/research",
                            json={"topic": query, "user_id": "kb_search"},
                        )
                        resp.raise_for_status()
                        result = resp.json()
                    st.info(
                        f"Research session started: `{result['session_id']}`\n\n"
                        "Check the Research page to view results."
                    )
                except Exception as exc:
                    st.error(f"Search failed: {exc}")

    st.divider()
    st.subheader("Ingested Documents")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{API_BASE}/ingest")
            if resp.status_code == 200:
                docs = resp.json()
                if docs:
                    st.dataframe(docs, use_container_width=True)
                else:
                    st.info("No documents ingested yet.")
    except Exception:
        # Fall back to querying via direct session list
        data = api_get("/sessions", limit=1, offset=0)
        st.info("Document listing unavailable — use the API directly at GET /ingest.")


# ── Sidebar navigation ────────────────────────────────────────────────────────

PAGES = {
    "Research": page_research,
    "Ingest": page_ingest,
    "Sessions": page_sessions,
    "Knowledge Base": page_knowledge_base,
}

with st.sidebar:
    st.title("DocuMind")
    st.caption("Multi-agent research platform")
    st.divider()
    selected_page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption(f"API: `{API_BASE}`")

PAGES[selected_page]()
