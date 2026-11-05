"""DocuMind Streamlit dashboard."""

from __future__ import annotations

import json
import os
import uuid

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8888")

# Unique ID for this browser session — cleared on page refresh
if "sid" not in st.session_state:
    st.session_state.sid = str(uuid.uuid4())
SID = st.session_state.sid

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
#MainMenu, footer, .stDeployButton {visibility: hidden;}
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {visibility: hidden;}

/* Always keep sidebar visible */
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 18rem !important;
    width: 18rem !important;
    transform: none !important;
}

[data-testid="stSidebar"] {background:#0d1117;border-right:1px solid #21262d;}
[data-testid="stSidebar"] * {color:#e6edf3 !important;}

.stApp {background:#0d1117;color:#e6edf3;}
.main .block-container {padding-top:1.5rem;max-width:1200px;}

[data-testid="stMetric"] {
    background:#161b22;border:1px solid #30363d;
    border-radius:8px;padding:12px 16px;
}
[data-testid="stMetricValue"] {color:#e6edf3 !important;}
[data-testid="stMetricLabel"] {color:#8b949e !important;}

.stButton > button[kind="primary"] {
    background:#238636;border:1px solid #2ea043;
    color:white;border-radius:6px;font-weight:600;
}
.stButton > button[kind="primary"]:hover {background:#2ea043;}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background:#161b22 !important;border:1px solid #30363d !important;
    color:#e6edf3 !important;border-radius:6px !important;
}

[data-testid="stExpander"] > div:first-child {
    background:#161b22 !important;border:1px solid #30363d !important;border-radius:6px !important;
}

.stDataFrame {background:#161b22;}
hr {border-color:#21262d !important;}
</style>
""", unsafe_allow_html=True)

_COLORS = {
    "researcher": "#1E88E5",
    "analyst":    "#F59E0B",
    "writer":     "#10B981",
    "critic":     "#EF4444",
}
_ICONS = {
    "researcher": "🔍",
    "analyst":    "📊",
    "writer":     "✍️",
    "critic":     "🔎",
}


# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(path: str, **params):
    try:
        r = httpx.get(
            f"{API_BASE}{path}",
            params={k: v for k, v in params.items() if v is not None},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("Cannot reach API — is it running?")
        return None
    except Exception as exc:
        st.error(f"API error ({path}): {exc}")
        return None


def api_post(path: str, body: dict | None = None, files=None, data=None):
    try:
        if files:
            r = httpx.post(f"{API_BASE}{path}", files=files, data=data, timeout=120.0)
        else:
            r = httpx.post(f"{API_BASE}{path}", json=body, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error ({path}): {exc}")
        return None


# ── UI components ─────────────────────────────────────────────────────────────

def agent_card(run: dict) -> None:
    name = run.get("agent_name", "agent")
    color = _COLORS.get(name, "#6B7280")
    icon = _ICONS.get(name, "🤖")
    output = (run.get("output") or "")[:3000]
    st.markdown(
        f"""<div style="border-left:4px solid {color};background:#161b22;
        border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;">
        <span style="background:{color};color:white;padding:2px 8px;border-radius:12px;
        font-size:0.75rem;font-weight:600;">{icon} {name.title()}</span>
        <span style="color:#8b949e;font-size:0.75rem;margin-left:8px;">
        {run.get('duration_ms','?')} ms</span>
        <div style="margin-top:10px;color:#e6edf3;font-size:0.88rem;
        white-space:pre-wrap;line-height:1.5;">{output}</div></div>""",
        unsafe_allow_html=True,
    )


def gauge(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#e6edf3", "size": 13}},
        number={"font": {"color": "#e6edf3"}, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#8b949e"},
            "bar": {"color": "#238636"},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0.0, 0.5], "color": "#21262d"},
                {"range": [0.5, 0.8], "color": "#1c2e28"},
                {"range": [0.8, 1.0], "color": "#163320"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        margin=dict(t=40, b=10, l=20, r=20), height=200,
    )
    return fig


def report_section(session_id: str) -> None:
    report_data = api_get(f"/research/{session_id}/report")
    if not report_data:
        return
    st.divider()
    st.markdown(f"## {report_data.get('title', 'Research Report')}")
    st.caption(f"{report_data.get('word_count', 0):,} words")
    content = report_data.get("content", "")
    st.markdown(content)

    raw = report_data.get("citations")
    if raw:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = []
        if raw:
            with st.expander(f"📚 Citations ({len(raw)})"):
                for i, c in enumerate(raw, 1):
                    st.markdown(f"**[{i}]** `{c.get('source','?')}` — {c.get('excerpt','')[:200]}")

    st.download_button(
        "⬇ Download Report (Markdown)",
        data=content,
        file_name=f"report_{session_id[:8]}.md",
        mime="text/markdown",
    )


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_research() -> None:
    st.markdown("## Research")
    st.caption("Submit a topic — the multi-agent pipeline runs and streams results in real time.")

    topic = st.text_input(
        "Research topic",
        placeholder="e.g. How do transformer attention mechanisms work?",
    )

    related = st.text_input(
        "Prior session ID (optional — injects prior report as context)",
        placeholder="paste a UUID",
    )

    if st.button("▶ Start Research", type="primary", disabled=not topic.strip()):
        body: dict = {"topic": topic.strip(), "user_id": SID}
        if related.strip():
            body["related_session_id"] = related.strip()
        with st.spinner("Queuing…"):
            result = api_post("/research", body)
        if result:
            st.session_state.update({
                "session_id": result["session_id"],
                "seen_agents": 0,
            })
            st.success(f"Started — session `{result['session_id']}`")

    if "session_id" not in st.session_state:
        return

    session_id = st.session_state["session_id"]
    st.divider()
    st.markdown(f"**Session** `{session_id[:8]}…`")

    # Agent feed placeholder (rendered outside fragment so cards persist)
    feed = st.container()

    @st.fragment(run_every=5)
    def live_status() -> None:
        data = api_get(f"/research/{session_id}")
        if not data:
            return

        status = data.get("status", "unknown")
        agent_count = int(data.get("agent_run_count", 0))
        seen = st.session_state.get("seen_agents", 0)

        if agent_count > seen:
            runs = api_get(f"/sessions/{session_id}/agents") or []
            with feed:
                for run in runs[seen:]:
                    agent_card(run)
            st.session_state["seen_agents"] = agent_count

        _badge_color = {"completed": "#238636", "failed": "#da3633"}.get(status, "#1f6feb")
        st.markdown(
            f'<span style="background:{_badge_color};color:white;padding:2px 10px;'
            f'border-radius:12px;font-size:0.8rem;">{status}</span>',
            unsafe_allow_html=True,
        )

        if status == "completed":
            report_section(session_id)
        elif status == "failed":
            st.error("Research job failed — check API logs.")

    live_status()


def page_upload() -> None:
    st.markdown("## Upload Document")
    st.caption("Ingest files or URLs directly into the vector store.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Upload File")
        st.caption("PDF, TXT, CSV, code files up to 200MB")
        f = st.file_uploader(
            "Drop file here",
            type=["pdf", "txt", "csv", "html", "htm", "py", "js", "ts", "go", "java"],
        )
        if f and st.button("Upload & Ingest", type="primary"):
            with st.spinner(f"Ingesting {f.name}…"):
                try:
                    r = httpx.post(
                        f"{API_BASE}/ingest",
                        files={"file": (f.name, f.getvalue(), "application/octet-stream")},
                        data={"user_id": SID},
                        timeout=120.0,
                    )
                    r.raise_for_status()
                    res = r.json()
                    st.success(
                        f"✅ **{f.name}** ingested  \n"
                        f"Chunks: **{res['chunk_count']}** | "
                        f"Embed: **{res['embed_time_ms']}ms** | "
                        f"Total: **{res['total_time_ms']}ms**"
                    )
                    st.session_state["last_ingest"] = res
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    with col2:
        st.markdown("### Ingest URL")
        st.caption("Web pages, arXiv papers, documentation")
        url = st.text_input("URL", placeholder="https://arxiv.org/abs/1706.03762")
        if url and st.button("Fetch & Ingest", type="primary"):
            with st.spinner(f"Fetching and ingesting {url}…"):
                try:
                    r = httpx.post(
                        f"{API_BASE}/ingest",
                        data={"url": url, "doc_type": "web", "user_id": SID},
                        timeout=120.0,
                    )
                    r.raise_for_status()
                    res = r.json()
                    st.success(
                        f"✅ **{res['source']}** ingested  \n"
                        f"Chunks: **{res['chunk_count']}** | "
                        f"Embed: **{res['embed_time_ms']}ms** | "
                        f"Total: **{res['total_time_ms']}ms**"
                    )
                    st.session_state["last_ingest"] = res
                except Exception as exc:
                    st.error(f"Failed: {exc}")

    if "last_ingest" in st.session_state:
        res = st.session_state["last_ingest"]
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", "🟢 Ready")
        c2.metric("Chunks Indexed", res.get("chunk_count", 0))
        c3.metric("Embed Time", f"{res.get('embed_time_ms', 0)}ms")
        c4.metric("Total Time", f"{res.get('total_time_ms', 0)}ms")
        st.caption(f"Document ID: `{res.get('document_id', '')}`")


def page_pipeline_status() -> None:
    st.markdown("## Pipeline Status")
    st.caption("S3 → Textract → pgvector ingestion overview.")

    analytics = api_get("/documents/analytics")
    if analytics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Documents", analytics.get("total_documents", 0))
        c2.metric("Ready", analytics.get("ready_documents", 0))
        c3.metric("Chunks Indexed", f"{analytics.get('total_chunks', 0):,}")
        c4.metric("Textract Pages", analytics.get("textract_pages_processed", 0))

        total = analytics.get("total_documents", 0)
        if total > 0:
            ready = analytics.get("ready_documents", 0)
            failed = analytics.get("failed_documents", 0)
            processing = max(total - ready - failed, 0)
            fig = px.pie(
                names=["Ready", "Processing", "Failed"],
                values=[ready, processing, failed],
                color_discrete_map={
                    "Ready": "#238636", "Processing": "#1f6feb", "Failed": "#da3633"
                },
                hole=0.5,
            )
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_color="#e6edf3",
                legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
                margin=dict(t=20, b=20), height=260,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Embedding mode", analytics.get("embedding_mode", "local"))
        c1.metric("LLM provider", analytics.get("llm_provider", "bedrock"))
        c2.metric("Est. monthly cost", f"${analytics.get('cost_estimate_monthly_usd', 2.50):.2f}")

    st.divider()
    st.markdown("### Indexed Documents")
    status_filter = st.selectbox("Filter", ["all", "ready", "processing", "failed"])
    docs_data = api_get(
        "/documents",
        status=status_filter if status_filter != "all" else None,
        limit=50,
        user_id=SID,
    )
    if docs_data:
        docs = docs_data.get("documents", [])
        if docs:
            df = pd.DataFrame([{
                "Filename": d.get("filename", ""),
                "Type": d.get("doc_type", ""),
                "Status": d.get("status", ""),
                "Pages": d.get("page_count") or 0,
                "Ingested": str(d.get("ingested_at", ""))[:16],
            } for d in docs])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No documents found.")

    st.divider()
    health = api_get("/health")
    if health:
        st.markdown("### System Health")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Postgres", "🟢" if health.get("postgres_ok") else "🔴")
        c2.metric("pgvector (RDS)", "🟢" if health.get("pgvector_ok") else "⚪")
        c3.metric("Qdrant", "🟢" if health.get("qdrant_ok") else "🔴")
        c4.metric("Embed mode", health.get("embedding_mode", "?"))


def page_evaluation() -> None:
    st.markdown("## Evaluation")
    st.caption("RAGAS faithfulness, answer relevancy, and composite score.")

    analytics = api_get("/documents/analytics")
    if analytics and analytics.get("last_eval"):
        ev = analytics["last_eval"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(gauge(ev.get("faithfulness", 0), "Faithfulness"), use_container_width=True)
        with c2:
            st.plotly_chart(gauge(ev.get("answer_relevancy", 0), "Answer Relevancy"), use_container_width=True)
        with c3:
            st.plotly_chart(gauge(ev.get("composite_score", 0), "Composite"), use_container_width=True)
        st.caption(f"Last evaluated: {str(ev.get('evaluated_at', ''))[:16]}")
    else:
        st.info(
            "No results yet.  \n"
            "Run: `python evaluation/eval_runner.py --n-samples 10`"
        )

    st.divider()
    st.markdown("### Monthly Cost Estimate")
    df = pd.DataFrame({
        "Service": [
            "EC2 t3.micro", "RDS db.t3.micro", "S3 (<5 GB)",
            "SQS + Lambda + Step Fn", "Textract (~200 pages/mo)",
            "Comprehend (~500 docs/mo)", "Bedrock Haiku (~500 runs/mo)", "CloudWatch + SSM",
        ],
        "Cost / month": [
            "$0 (free tier)", "$0 (free tier)", "$0 (free tier)", "$0 (free tier)",
            "~$0.30", "~$0.05", "~$2.00", "$0 (free tier)",
        ],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.metric("Total estimated", "~$2.50 / month")


def page_sessions() -> None:
    st.markdown("## Research Sessions")
    data = api_get("/sessions", limit=100, user_id=SID)
    if not data:
        return
    sessions = data.get("sessions", [])
    st.caption(f"**{data.get('total', 0)}** sessions")

    if not sessions:
        st.info("No sessions yet.")
        return

    df = pd.DataFrame([{
        "ID": s["id"][:8],
        "Topic": s["topic"][:72],
        "Status": s["status"],
        "Agents": s["agent_run_count"],
        "Created": str(s["created_at"])[:16],
    } for s in sessions])
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Agents": st.column_config.NumberColumn(format="%d"),
        },
    )

    st.divider()
    selected = st.selectbox(
        "Inspect session",
        [s["id"] for s in sessions],
        format_func=lambda x: x[:8] + "…",
    )
    if selected and st.button("Load"):
        tab_report, tab_agents = st.tabs(["Report", "Agent Runs"])
        with tab_report:
            report = api_get(f"/research/{selected}/report")
            if report:
                content = report.get("content", "")
                st.markdown(f"## {report.get('title', 'Report')}")
                st.caption(f"{report.get('word_count', 0):,} words")
                st.markdown(content)
                st.download_button(
                    "⬇ Download (Markdown)",
                    data=content,
                    file_name=f"report_{selected[:8]}.md",
                    mime="text/markdown",
                )
            else:
                st.info("No report for this session.")
        with tab_agents:
            runs = api_get(f"/sessions/{selected}/agents") or []
            if runs:
                for run in runs:
                    agent_card(run)
            else:
                st.info("No agent runs recorded.")


def page_knowledge_base() -> None:
    st.markdown("## Knowledge Base")

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
        c5.metric("Embed mode", e.get("mode", "local"))

    st.divider()
    st.markdown("### Ask the Knowledge Base")
    query = st.text_area(
        "Question",
        placeholder="What does the literature say about attention mechanisms?",
        height=80,
    )
    top_k = st.slider("Top chunks", 3, 10, 5)

    if query.strip() and st.button("Search", type="primary"):
        with st.spinner("Querying…"):
            result = api_post("/research/query", {"query": query.strip(), "top_k": top_k})
        if result:
            answer = result.get("answer") or result.get("response", "")
            chunks = result.get("chunks", [])
            if answer:
                st.markdown("**Answer**")
                st.markdown(answer)
            if chunks:
                with st.expander(f"Retrieved chunks ({len(chunks)})"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(
                            f"**[{i}]** `{chunk.get('source','?')}` "
                            f"— score {chunk.get('score', 0):.3f}"
                        )
                        st.code(chunk.get("text", "")[:400], language="text")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="font-size:1.4rem;font-weight:700;color:#58a6ff;">🧠 DocuMind</div>'
        '<div style="font-size:0.75rem;color:#8b949e;margin-bottom:1rem;">'
        'Enterprise RAG on AWS</div>',
        unsafe_allow_html=True,
    )

    try:
        _h = httpx.get(f"{API_BASE}/health", timeout=3.0)
        _h.raise_for_status()
        _hd = _h.json()
        _pg = "🟢" if _hd.get("postgres_ok") else "🔴"
        _qd = "🟢" if _hd.get("qdrant_ok") else "🔴"
        _em = _hd.get("embedding_mode", "local")
        _lm = _hd.get("llm_provider", "bedrock")
        st.markdown(
            f"Postgres {_pg} &nbsp; Qdrant {_qd}  \n"
            f"Embed: `{_em}` | LLM: `{_lm}`",
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown("⚠️ API unreachable", unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🔬 Research",
            "📤 Upload",
            "📊 Pipeline Status",
            "📈 Evaluation",
            "📋 Sessions",
            "🗄️ Knowledge Base",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"API: `{API_BASE}`")


# ── Router ────────────────────────────────────────────────────────────────────

_pages = {
    "🔬 Research":        page_research,
    "📤 Upload":          page_upload,
    "📊 Pipeline Status": page_pipeline_status,
    "📈 Evaluation":      page_evaluation,
    "📋 Sessions":        page_sessions,
    "🗄️ Knowledge Base":  page_knowledge_base,
}
_pages[page]()
