"""DocuMind Streamlit dashboard."""

from __future__ import annotations

import json
import os
import time

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8888")

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

_AGENT_COLORS = {
    "researcher": "#1E88E5",
    "analyst":    "#F59E0B",
    "writer":     "#10B981",
    "critic":     "#EF4444",
}

_AGENT_ICONS = {
    "researcher": "🔍",
    "analyst":    "📊",
    "writer":     "✍️",
    "critic":     "🔎",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str, **params):
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, timeout=15.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error ({path}): {exc}")
        return None


def api_post(path: str, body: dict):
    try:
        r = httpx.post(f"{API_BASE}{path}", json=body, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error ({path}): {exc}")
        return None


def _agent_badge(name: str) -> str:
    color = _AGENT_COLORS.get(name, "#6B7280")
    icon  = _AGENT_ICONS.get(name, "🤖")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:12px;font-size:0.75rem;font-weight:600;">'
        f'{icon} {name.title()}</span>'
    )


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_research():
    st.header("Research")
    st.caption("Submit a topic — the multi-agent workflow runs and streams output in real time.")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Research topic", placeholder="e.g. How do transformer attention mechanisms work?")
    with col2:
        user_id = st.text_input("User ID", value="demo")

    related_session_id = st.text_input(
        "Prior session ID (optional — injects previous report as context)",
        placeholder="paste a session UUID to build on prior work",
    )

    if st.button("▶ Start Research", type="primary", disabled=not topic.strip()):
        body: dict = {"topic": topic.strip(), "user_id": user_id or "demo"}
        if related_session_id.strip():
            body["related_session_id"] = related_session_id.strip()

        with st.spinner("Submitting…"):
            result = api_post("/research", body)
        if result:
            st.session_state["session_id"] = result["session_id"]
            st.session_state["agent_messages"] = []
            st.success(f"Session started: `{result['session_id']}`")

    if "session_id" not in st.session_state:
        return

    session_id = st.session_state["session_id"]
    st.divider()
    st.subheader(f"Session `{session_id[:8]}…`")

    # Live agent feed container
    feed_container = st.container()
    status_placeholder = st.empty()

    with st.status("Running workflow…", expanded=True) as status_box:
        last_agent_count = 0
        for _ in range(300):                        # ~10 min timeout at 2s poll
            data = api_get(f"/research/{session_id}")
            if not data:
                break

            current_status = data.get("status", "unknown")
            agent_count    = int(data.get("agent_run_count", 0))
            status_box.write(f"**Status:** `{current_status}` | **Agent runs:** {agent_count}")

            # Fetch new agent runs
            if agent_count > last_agent_count:
                runs = api_get(f"/sessions/{session_id}/agents") or []
                for run in runs[last_agent_count:]:
                    name = run["agent_name"]
                    with feed_container:
                        with st.expander(
                            f"{_AGENT_ICONS.get(name,'🤖')} **{name.title()}** "
                            f"— {run.get('duration_ms', '?')} ms",
                            expanded=True,
                        ):
                            st.markdown(
                                f'<div style="border-left:4px solid {_AGENT_COLORS.get(name,"#6B7280")};'
                                f'padding-left:12px;">{(run.get("output") or "")[:2000]}</div>',
                                unsafe_allow_html=True,
                            )
                last_agent_count = agent_count

            if current_status in ("completed", "failed"):
                label = "✅ Research complete!" if current_status == "completed" else "❌ Research failed"
                state = "complete" if current_status == "completed" else "error"
                status_box.update(label=label, state=state, expanded=False)
                break

            time.sleep(2)

    # Final report
    report_data = api_get(f"/research/{session_id}/report")
    if report_data:
        st.subheader(report_data.get("title", "Research Report"))
        word_count = report_data.get("word_count", 0)
        st.caption(f"{word_count:,} words")
        st.markdown(report_data.get("content", ""))

        raw_citations = report_data.get("citations")
        if raw_citations:
            if isinstance(raw_citations, str):
                try:
                    raw_citations = json.loads(raw_citations)
                except Exception:
                    raw_citations = []
            if raw_citations:
                with st.expander(f"📚 Citations ({len(raw_citations)})", expanded=False):
                    for i, cit in enumerate(raw_citations, 1):
                        st.markdown(
                            f"**[{i}]** `{cit.get('source','?')}` — {cit.get('excerpt','')[:200]}"
                        )


def page_ingest():
    st.header("Ingest Documents")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload File")
        f = st.file_uploader("PDF, CSV, TXT, code", type=["pdf","csv","txt","py","js","ts","md"])
        if f and st.button("Ingest File"):
            with st.spinner(f"Ingesting {f.name}…"):
                try:
                    r = httpx.post(
                        f"{API_BASE}/ingest",
                        files={"file": (f.name, f.getvalue())},
                        timeout=120.0,
                    )
                    r.raise_for_status()
                    res = r.json()
                    st.success(
                        f"**{f.name}** ingested\n\n"
                        f"- Chunks: **{res['chunk_count']}**\n"
                        f"- Embed: {res['embed_time_ms']} ms\n"
                        f"- Total: {res['total_time_ms']} ms"
                    )
                except Exception as exc:
                    st.error(f"Failed: {exc}")

    with col2:
        st.subheader("Ingest URL")
        url = st.text_input("URL", placeholder="https://example.com/paper")
        dtype = st.selectbox("Type", ["web","pdf","text"])
        if url and st.button("Ingest URL"):
            with st.spinner(f"Fetching {url}…"):
                try:
                    r = httpx.post(
                        f"{API_BASE}/ingest",
                        data={"url": url, "doc_type": dtype},
                        timeout=120.0,
                    )
                    r.raise_for_status()
                    res = r.json()
                    st.success(f"Ingested **{url}** — {res['chunk_count']} chunks in {res['total_time_ms']} ms")
                except Exception as exc:
                    st.error(f"Failed: {exc}")


def page_sessions():
    st.header("Past Research Sessions")
    data = api_get("/sessions", limit=50)
    if not data:
        return
    sessions = data.get("sessions", [])
    st.caption(f"Total: **{data.get('total', 0)}** sessions")

    if not sessions:
        st.info("No sessions yet.")
        return

    rows = [
        {
            "ID": s["id"][:8] + "…",
            "Topic": s["topic"][:70],
            "Status": s["status"],
            "Agents": s["agent_run_count"],
            "Created": str(s["created_at"])[:16],
        }
        for s in sessions
    ]
    st.dataframe(rows, use_container_width=True)

    st.divider()
    selected = st.selectbox("View session", [s["id"] for s in sessions],
                            format_func=lambda x: x[:8] + "…")
    if selected and st.button("Load"):
        report = api_get(f"/research/{selected}/report")
        if report:
            st.subheader(report.get("title", "Report"))
            st.markdown(report.get("content", ""))
        else:
            st.json(api_get(f"/research/{selected}"))

        agents = api_get(f"/sessions/{selected}/agents") or []
        if agents:
            with st.expander("Agent runs"):
                for run in agents:
                    name = run["agent_name"]
                    st.markdown(
                        _agent_badge(name) +
                        f" &nbsp; {run.get('duration_ms','?')} ms",
                        unsafe_allow_html=True,
                    )
                    st.code((run.get("output") or "")[:800], language="markdown")


def page_knowledge_base():
    st.header("Knowledge Base")

    # Metrics panel
    metrics = api_get("/metrics")
    if metrics:
        q = metrics.get("qdrant", {})
        p = metrics.get("postgres", {})
        e = metrics.get("embeddings", {})

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Vectors", f"{q.get('vector_count', 0):,}")
        c2.metric("Documents", p.get("documents_ingested", 0))
        c3.metric("Sessions", p.get("sessions_total", 0))
        c4.metric("Reports", p.get("reports_generated", 0))
        c5.metric("Agent Runs", p.get("agent_runs_total", 0))

        with st.expander("Full metrics JSON"):
            st.json(metrics)

    st.divider()
    st.subheader("Search Knowledge Base")
    query = st.text_input("Query", placeholder="What is attention mechanism?")
    if query and st.button("Search"):
        with st.spinner("Running RAG query…"):
            result = api_post("/research", {"topic": query, "user_id": "kb_search"})
        if result:
            st.info(f"Research started: `{result['session_id']}`\nCheck the Research page for results.")

    st.divider()
    st.subheader("Ingested Documents")
    doc_data = api_get("/sessions", limit=1)
    if doc_data is not None:
        health = api_get("/health")
        if health:
            col1, col2, col3 = st.columns(3)
            col1.metric("Qdrant", "🟢 Online" if health.get("qdrant_ok") else "🔴 Offline")
            col2.metric("Postgres", "🟢 Online" if health.get("postgres_ok") else "🔴 Offline")
            col3.metric("Embed mode", health.get("embedding_mode", "?"))


# ── Navigation ────────────────────────────────────────────────────────────────

PAGES = {
    "🔬 Research":        page_research,
    "📥 Ingest":          page_ingest,
    "📋 Sessions":        page_sessions,
    "🗄️ Knowledge Base": page_knowledge_base,
}

with st.sidebar:
    st.title("DocuMind 🧠")
    st.caption("Multi-agent research platform")
    st.divider()
    page = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption(f"API: `{API_BASE}`")

PAGES[page]()
