"""DocuMind Streamlit dashboard."""

from __future__ import annotations

import html
import json
import os
import uuid
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8888")
ASSETS = Path(__file__).parent / "assets"

st.set_page_config(
    page_title="DocuMind · Research workspace",
    page_icon=str(ASSETS / "logo_icon.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# Unique ID for this browser session — cleared on page refresh
if "sid" not in st.session_state:
    st.session_state.sid = str(uuid.uuid4())
SID = st.session_state.sid

# Agent identity colours — validated as an ordered categorical set on the dark
# surface (adjacent-pair CVD ΔE ≥ 8.4); always shown with a name + icon.
AGENTS = {
    "researcher": {"color": "#3987e5", "icon": "travel_explore", "label": "Researcher"},
    "analyst":    {"color": "#d95926", "icon": "insights",       "label": "Analyst"},
    "writer":     {"color": "#199e70", "icon": "edit_note",      "label": "Writer"},
    "critic":     {"color": "#c98500", "icon": "fact_check",     "label": "Critic"},
}
STATUS = {
    "completed": ("good", "Completed"),
    "failed":    ("bad", "Failed"),
    "running":   ("info", "Running"),
    "pending":   ("muted", "Queued"),
}

# ── Design system ─────────────────────────────────────────────────────────────

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root {
  --bg: #0B0F17; --surface: #121826; --surface-2: #182033; --border: #232C3F;
  --border-soft: #1C2435; --text: #E8ECF4; --text-2: #A3ADC2; --muted: #8792A9;
  --accent: #7C8CFF; --accent-soft: rgba(124,140,255,.12);
  --good: #3DD68C; --warn: #F5B94A; --bad: #F2555A; --info: #7C8CFF;
}

/* chrome */
/* Hide menu/deploy chrome only — stToolbar also hosts the expand-sidebar control */
#MainMenu, footer, .stDeployButton, [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stMainMenu"],
[data-testid="stToolbarActions"] {display: none !important;}
header[data-testid="stHeader"] {background: transparent;}

/* Desktop: sidebar is the primary navigation — keep it permanently open */
@media (min-width: 768px) {
  [data-testid="stSidebar"], [data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 300px !important; max-width: 300px !important;
    transform: none !important; margin-left: 0 !important; visibility: visible !important;}
  [data-testid="stSidebarCollapseButton"] {display: none !important;}
}
.block-container {padding-top: 2.25rem; padding-bottom: 4rem; max-width: 1120px;}
.stApp {background:
  radial-gradient(1200px 500px at 15% -10%, rgba(124,140,255,.08), transparent 60%), var(--bg);}

/* sidebar */
[data-testid="stSidebarNav"] a {border-radius: 8px; margin: 1px 0;}
[data-testid="stSidebarNav"] a[aria-current="page"] {background: var(--accent-soft);}
[data-testid="stSidebarNav"] a[aria-current="page"] span {color: #fff;}
[data-testid="stSidebarNavSeparator"] {display: none;}
[data-testid="stSidebarHeader"] {padding-bottom: .75rem;}

/* buttons: dark ink on the light accent for AA contrast */
button[kind="primary"], button[kind="primaryFormSubmit"] {
  color: #0B0F17 !important; font-weight: 600; border: none;
  box-shadow: 0 1px 0 rgba(255,255,255,.15) inset, 0 6px 20px -8px rgba(124,140,255,.6);}
button[kind="primary"] p, button[kind="primaryFormSubmit"] p {color: #0B0F17 !important;}
button[kind="primary"]:disabled, button[kind="primaryFormSubmit"]:disabled {
  background: var(--surface-2) !important; box-shadow: none; border: 1px solid var(--border) !important;}
button[kind="primary"]:disabled p, button[kind="primaryFormSubmit"]:disabled p,
button[kind="primary"]:disabled span {color: var(--muted) !important;}
button[kind="secondary"] {font-weight: 500;}

/* page header */
.dm-eyebrow {font-size: .75rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: var(--accent); margin-bottom: .35rem;}
.dm-title {font-size: 1.85rem; font-weight: 650; letter-spacing: -.02em; color: var(--text);
  line-height: 1.2; margin: 0;}
.dm-sub {color: var(--text-2); font-size: .98rem; margin: .45rem 0 .6rem; max-width: 680px;
  line-height: 1.55;}

/* stat tiles */
.dm-stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px;
  margin: .25rem 0 1.25rem;}
.dm-stat {background: var(--surface); border: 1px solid var(--border-soft); border-radius: 12px;
  padding: 14px 16px;}
.dm-stat-label {color: var(--muted); font-size: .78rem; font-weight: 500; display: flex;
  align-items: center; gap: 6px;}
.dm-stat-value {color: var(--text); font-size: 1.6rem; font-weight: 650; letter-spacing: -.02em;
  margin-top: 4px; font-variant-numeric: tabular-nums;}
.dm-stat-hint {color: var(--muted); font-size: .76rem; margin-top: 2px;}

/* pills */
.dm-pill {display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-size: .78rem; font-weight: 600; border: 1px solid transparent; white-space: nowrap;}
.dm-pill::before {content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor;}
.dm-pill.good {color: var(--good); background: rgba(61,214,140,.10); border-color: rgba(61,214,140,.25);}
.dm-pill.bad {color: var(--bad); background: rgba(242,85,90,.10); border-color: rgba(242,85,90,.25);}
.dm-pill.warn {color: var(--warn); background: rgba(245,185,74,.10); border-color: rgba(245,185,74,.25);}
.dm-pill.info {color: var(--info); background: var(--accent-soft); border-color: rgba(124,140,255,.3);}
.dm-pill.muted {color: var(--text-2); background: var(--surface-2); border-color: var(--border);}
.dm-pill.info::before {animation: dm-pulse 1.4s ease-in-out infinite;}
@keyframes dm-pulse {50% {opacity: .3;}}

/* stepper */
.dm-steps {display: flex; gap: 6px; margin: .5rem 0 1rem; flex-wrap: wrap;}
.dm-step {flex: 1 1 120px; padding: 10px 12px; border-radius: 10px; background: var(--surface);
  border: 1px solid var(--border-soft); font-size: .8rem; color: var(--muted);}
.dm-step b {display: block; font-size: .7rem; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; margin-bottom: 2px; color: inherit; opacity: .8;}
.dm-step.done {color: var(--text-2); border-color: rgba(61,214,140,.25);}
.dm-step.done b {color: var(--good); opacity: 1;}
.dm-step.active {color: var(--text); border-color: var(--accent); background: var(--accent-soft);}
.dm-step.active b {color: var(--accent); opacity: 1;}
.dm-step.failed {color: var(--text); border-color: var(--bad);}

/* agent cards */
.dm-agent {background: var(--surface); border: 1px solid var(--border-soft); border-radius: 12px;
  padding: 14px 16px; margin: 10px 0; position: relative; overflow: hidden;}
.dm-agent::before {content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--agent);}
.dm-agent-head {display: flex; align-items: center; gap: 10px;}
.dm-agent-icon {width: 28px; height: 28px; border-radius: 8px; display: grid; place-items: center;
  background: color-mix(in srgb, var(--agent) 18%, transparent); color: var(--agent);
  font-family: 'Material Symbols Rounded'; font-size: 18px;}
.dm-agent-name {font-weight: 600; color: var(--text); font-size: .92rem;}
.dm-agent-meta {color: var(--muted); font-size: .76rem; margin-left: auto;}
.dm-agent-body {color: var(--text-2); font-size: .86rem; line-height: 1.6; margin-top: 10px;
  white-space: pre-wrap; max-height: 9.6em; overflow: hidden;
  mask-image: linear-gradient(#000 60%, transparent);}
.dm-agent details summary {cursor: pointer; color: var(--accent); font-size: .8rem; margin-top: 6px;
  list-style: none;}
.dm-agent details[open] .dm-agent-body {max-height: none; mask-image: none;}
.dm-agent details[open] summary {display: none;}

/* report paper */
.st-key-report {background: var(--surface); border: 1px solid var(--border-soft); border-radius: 16px;
  padding: 2rem 2.4rem !important;}
.st-key-report p, .st-key-report li {font-family: 'Source Serif 4', Georgia, serif; font-size: 1.04rem;
  line-height: 1.75; color: #D9DEE9;}
.st-key-report h1 {font-size: 1.6rem;} .st-key-report h2 {font-size: 1.25rem; margin-top: 1.4rem;}
.st-key-report h3 {font-size: 1.05rem;}
.st-key-answer h1 {font-size: 1.3rem;} .st-key-answer h2 {font-size: 1.12rem;}
.st-key-answer h3 {font-size: 1rem;} .st-key-answer hr {margin: 1rem 0;}
.st-key-answer {padding: 1.4rem 1.6rem !important;}

/* source cards */
.dm-sources {display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px;}
.dm-source {background: var(--surface); border: 1px solid var(--border-soft); border-radius: 10px;
  padding: 12px 14px;}
.dm-source-top {display: flex; align-items: center; gap: 8px; margin-bottom: 6px;}
.dm-source-n {font: 600 .72rem 'JetBrains Mono', monospace; color: var(--accent);
  background: var(--accent-soft); border-radius: 6px; padding: 1px 6px;}
.dm-source-name {font-size: .82rem; font-weight: 600; color: var(--text); overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap;}
.dm-source-score {margin-left: auto; font: .72rem 'JetBrains Mono', monospace; color: var(--muted);}
.dm-source-text {font-size: .8rem; color: var(--text-2); line-height: 1.5; display: -webkit-box;
  -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden;}

/* empty state + misc */
.dm-empty {border: 1px dashed var(--border); border-radius: 14px; padding: 2.2rem; text-align: center;
  color: var(--muted); background: rgba(18,24,38,.5);}
.dm-empty-icon {font-family: 'Material Symbols Rounded'; font-size: 32px; color: var(--accent);
  display: block; margin-bottom: .4rem;}
.dm-empty {font-size: .88rem;}
.dm-empty b {display: block; color: var(--text); font-size: .98rem; margin-bottom: .25rem;}
.dm-section {font-size: .78rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase;
  color: var(--muted); margin: 1.1rem 0 .1rem;}
.dm-kv {margin: 0; padding: 0; display: grid; grid-template-columns: max-content 1fr; gap: 8px 18px; font-size: .88rem;}
.dm-kv dt {color: var(--muted);} .dm-kv dd {margin: 0; color: var(--text);
  font-family: 'JetBrains Mono', monospace; font-size: .82rem;}
.dm-side-status {font-size: .8rem; color: var(--muted); line-height: 1.7;}
.dm-side-status code {font-size: .75rem;}
[data-testid="stExpander"] details {border-color: var(--border-soft);}
</style>
""")


# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(path: str, quiet: bool = False, **params):
    try:
        r = httpx.get(
            f"{API_BASE}{path}",
            params={k: v for k, v in params.items() if v is not None},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        if not quiet:
            st.error("The DocuMind API is unreachable. Please try again shortly.",
                     icon=":material/cloud_off:")
    except httpx.HTTPStatusError as exc:
        if not quiet:
            st.error(f"Request failed ({exc.response.status_code}) — {path}",
                     icon=":material/error:")
    except Exception as exc:
        if not quiet:
            st.error(f"Something went wrong: {exc}", icon=":material/error:")
    return None


def api_post(path: str, body: dict | None = None, files=None, data=None, timeout: float = 30.0):
    try:
        if files or data:
            r = httpx.post(f"{API_BASE}{path}", files=files, data=data, timeout=timeout)
        else:
            r = httpx.post(f"{API_BASE}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        st.error(f"Request failed: {detail}", icon=":material/error:")
    except Exception as exc:
        st.error(f"Something went wrong: {exc}", icon=":material/error:")
    return None


@st.cache_data(ttl=30, show_spinner=False)
def cached_health() -> dict | None:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=4.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Components ────────────────────────────────────────────────────────────────

def esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.html(
        f'<div class="dm-eyebrow">{esc(eyebrow)}</div>'
        f'<h1 class="dm-title">{esc(title)}</h1>'
        f'<p class="dm-sub">{esc(subtitle)}</p>'
    )


def stat_tiles(items: list[tuple[str, str, str]]) -> None:
    """items: (label, value, hint)."""
    tiles = "".join(
        f'<div class="dm-stat"><div class="dm-stat-label">{esc(label)}</div>'
        f'<div class="dm-stat-value">{esc(value)}</div>'
        f'<div class="dm-stat-hint">{esc(hint)}</div></div>'
        for label, value, hint in items
    )
    st.html(f'<div class="dm-stats">{tiles}</div>')


def pill_html(kind: str, text: str) -> str:
    return f'<span class="dm-pill {kind}">{esc(text)}</span>'


def status_pill(status: str) -> str:
    kind, label = STATUS.get(status, ("muted", status.replace("_", " ").title()))
    return pill_html(kind, label)


def section(title: str) -> None:
    st.html(f'<div class="dm-section">{esc(title)}</div>')


def empty_state(icon: str, title: str, body: str) -> None:
    st.html(
        f'<div class="dm-empty"><span class="dm-empty-icon">{esc(icon)}</span>'
        f'<b>{esc(title)}</b>{esc(body)}</div>'
    )


def fmt_ms(ms) -> str:
    if not ms:
        return "—"
    ms = float(ms)
    return f"{ms / 1000:.1f}s" if ms < 60_000 else f"{ms / 60_000:.1f} min"


def parse_citations(raw) -> list[dict]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    return raw if isinstance(raw, list) else []


def source_cards(items: list[dict], score_key: str | None = None) -> None:
    cards = []
    for i, c in enumerate(items, 1):
        name = Path(str(c.get("source") or "unknown")).name or str(c.get("source"))
        score = (
            f'<span class="dm-source-score">{float(c.get(score_key) or 0):.3f}</span>'
            if score_key else ""
        )
        text = c.get("excerpt") or c.get("text") or ""
        cards.append(
            f'<div class="dm-source"><div class="dm-source-top">'
            f'<span class="dm-source-n">{i}</span>'
            f'<span class="dm-source-name" title="{esc(c.get("source"))}">{esc(name)}</span>'
            f'{score}</div><div class="dm-source-text">{esc(text[:400])}</div></div>'
        )
    st.html(f'<div class="dm-sources">{"".join(cards)}</div>')


def agent_card(run: dict) -> None:
    name = run.get("agent_name", "agent")
    meta = AGENTS.get(name, {"color": "#8792A9", "icon": "smart_toy", "label": name.title()})
    output = (run.get("output") or "").strip()
    duration = fmt_ms(run.get("duration_ms"))
    body = f'<div class="dm-agent-body">{esc(output[:6000])}</div>'
    if len(output) > 420:
        body = f"<details>{body}<summary>Show full output</summary></details>"
    st.html(
        f'<div class="dm-agent" style="--agent:{meta["color"]}">'
        f'<div class="dm-agent-head"><span class="dm-agent-icon">{meta["icon"]}</span>'
        f'<span class="dm-agent-name">{esc(meta["label"])}</span>'
        f'<span class="dm-agent-meta">{esc(duration)}</span></div>{body}</div>'
    )


STEPS = ["Plan & retrieve", "Researcher", "Analyst", "Writer", "Critic", "Final report"]


def stepper(status: str, agent_count: int) -> None:
    if status == "completed":
        current = len(STEPS)
    elif agent_count == 0:
        current = 0
    else:
        current = min(agent_count + 1, len(STEPS) - 1)
    cells = []
    for i, label in enumerate(STEPS):
        if i < current:
            cls, tag = "done", "Done"
        elif i == current:
            cls, tag = ("failed", "Failed") if status == "failed" else ("active", "In progress")
        else:
            cls, tag = "", f"Step {i + 1}"
        cells.append(f'<div class="dm-step {cls}"><b>{tag}</b>{esc(label)}</div>')
    st.html(f'<div class="dm-steps">{"".join(cells)}</div>')


def render_report(session_id: str, report: dict) -> None:
    content = report.get("content", "")
    citations = parse_citations(report.get("citations"))
    words = report.get("word_count") or len(content.split())

    head, actions = st.columns([3, 1], vertical_alignment="bottom")
    with head:
        st.html(
            f'<div class="dm-eyebrow">Research report</div>'
            f'<div style="color:var(--muted);font-size:.85rem">{words:,} words · '
            f'{len(citations)} source{"s" if len(citations) != 1 else ""} · '
            f'session <code>{esc(session_id[:8])}</code></div>'
        )
    with actions:
        st.download_button(
            "Download .md",
            data=content,
            file_name=f"documind_report_{session_id[:8]}.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
            key=f"dl_{session_id}",
        )
    with st.container(key="report"):
        st.markdown(content)
    if citations:
        section(f"Sources cited ({len(citations)})")
        source_cards(citations)


# ── Pages ─────────────────────────────────────────────────────────────────────

EXAMPLES = [
    "How does retrieval-augmented generation reduce hallucinations?",
    "Compare dense and sparse retrieval for enterprise search",
    "What are the key ideas behind transformer attention?",
]


def _use_example() -> None:
    choice = st.session_state.get("example_pick")
    if choice:
        st.session_state["topic_input"] = choice


def page_research() -> None:
    page_header(
        "Multi-agent research",
        "What do you want to research?",
        "A team of AI agents plans the question, retrieves evidence from your knowledge base, "
        "analyses it, drafts a report and has it reviewed by a critic — with citations.",
    )

    with st.form("research_form", border=True):
        topic = st.text_area(
            "Research topic",
            key="topic_input",
            placeholder="e.g. What are the trade-offs between hybrid search and pure vector search?",
            height=96,
            label_visibility="collapsed",
        )
        with st.expander("Advanced options"):
            related = st.text_input(
                "Build on a previous session",
                placeholder="Paste a session ID to include its report as context",
            )
        submitted = st.form_submit_button(
            "Start research", type="primary", icon=":material/arrow_forward:"
        )

    st.pills(
        "Try an example",
        EXAMPLES,
        key="example_pick",
        on_change=_use_example,
        label_visibility="visible",
    )

    if submitted:
        if not topic.strip():
            st.warning("Enter a topic to start.", icon=":material/edit:")
        else:
            body: dict = {"topic": topic.strip(), "user_id": SID}
            if related.strip():
                body["related_session_id"] = related.strip()
            result = api_post("/research", body)
            if result:
                st.session_state["session_id"] = result["session_id"]
                st.session_state["job_topic"] = topic.strip()
                st.session_state["job_final"] = None
                st.query_params["session"] = result["session_id"]

    linked = st.query_params.get("session")
    if linked and linked != st.session_state.get("session_id"):
        try:
            uuid.UUID(linked)
            st.session_state["session_id"] = linked
            st.session_state["job_final"] = None
        except ValueError:
            st.query_params.pop("session", None)

    session_id = st.session_state.get("session_id")
    if not session_id:
        return

    st.divider()
    final = st.session_state.get("job_final")
    if final:
        _render_job(session_id, final)
    else:
        _live_job(session_id)


@st.fragment(run_every=4)
def _live_job(session_id: str) -> None:
    data = api_get(f"/research/{session_id}", quiet=True)
    if not data:
        st.caption("Waiting for the research service…")
        return
    status = data.get("status", "pending")
    if status in ("completed", "failed"):
        st.session_state["job_final"] = status
        st.rerun(scope="app")
    _render_job(session_id, status, data)


def _render_job(session_id: str, status: str, data: dict | None = None) -> None:
    data = data or api_get(f"/research/{session_id}", quiet=True) or {}
    topic = data.get("topic") or st.session_state.get("job_topic", "")
    runs = api_get(f"/sessions/{session_id}/agents", quiet=True) or []

    left, right = st.columns([4, 1], vertical_alignment="center")
    left.markdown(f"#### {topic}")
    right.html(f'<div style="text-align:right">{status_pill(status)}</div>')
    stepper(status, len(runs))

    if status == "completed":
        report = api_get(f"/research/{session_id}/report", quiet=True)
        if report:
            render_report(session_id, report)
        if runs:
            with st.expander(f"Agent activity ({len(runs)} steps)", icon=":material/groups:"):
                for run in runs:
                    agent_card(run)
    elif status == "failed":
        st.error("This research run failed. Try again, or rephrase the topic.",
                 icon=":material/error:")
    else:
        if runs:
            for run in runs:
                agent_card(run)
        else:
            st.caption("Planning the research and retrieving relevant passages…")


ASK_EXAMPLES = [
    "What is Reciprocal Rank Fusion and why is it used?",
    "How do vector databases index embeddings?",
    "What is a large language model?",
]


def _use_ask_example() -> None:
    if st.session_state.get("ask_pick"):
        st.session_state["ask_input"] = st.session_state["ask_pick"]


def page_ask() -> None:
    page_header(
        "Knowledge base",
        "Ask your documents",
        "Get a direct, cited answer from everything indexed in DocuMind — no full research run needed.",
    )
    with st.form("ask_form", border=True):
        query = st.text_area(
            "Question",
            key="ask_input",
            placeholder="e.g. What is Reciprocal Rank Fusion and why is it used?",
            height=88,
            label_visibility="collapsed",
        )
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        top_k = c1.slider("Passages to retrieve", 3, 10, 5)
        asked = c2.form_submit_button(
            "Ask", type="primary", icon=":material/search:", width="stretch"
        )

    if not st.session_state.get("ask_result"):
        st.pills("Try asking", ASK_EXAMPLES, key="ask_pick", on_change=_use_ask_example)

    if asked and query.strip():
        with st.spinner("Searching the knowledge base…"):
            result = api_post(
                "/research/query", {"query": query.strip(), "top_k": top_k}, timeout=120.0
            )
        if result:
            st.session_state["ask_result"] = result
    elif asked:
        st.warning("Type a question first.", icon=":material/edit:")

    result = st.session_state.get("ask_result")
    if not result:
        return

    conf = float(result.get("confidence") or 0)
    kind = "good" if conf >= 0.6 else "warn" if conf >= 0.35 else "bad"
    pills = pill_html(kind, f"Confidence {conf:.0%}")
    if result.get("is_grounded") is False:
        pills += " " + pill_html("warn", "Low grounding")
    section("Answer")
    with st.container(key="answer", border=True):
        st.html(f"<div>{pills}</div>")
        st.markdown(result.get("answer", ""))
    if result.get("guardrail_warning"):
        st.caption(result["guardrail_warning"])
    chunks = result.get("chunks") or []
    if chunks:
        section(f"Retrieved passages ({len(chunks)})")
        source_cards(chunks, score_key="score")


def page_library() -> None:
    page_header(
        "Library",
        "Your documents",
        "Add PDFs, text, CSV, code or web pages. Content is chunked, embedded and made searchable "
        "for research and Q&A.",
    )
    analytics = api_get("/documents/analytics", quiet=True) or {}
    stat_tiles([
        ("Documents indexed", f"{analytics.get('total_documents', 0):,}", "across the knowledge base"),
        ("Chunks", f"{analytics.get('total_chunks', 0):,}", "searchable passages"),
        ("Embedding", str(analytics.get("embedding_mode", "—")).title(), "vector backend"),
    ])

    with st.container(border=True):
        tab_file, tab_url = st.tabs([":material/upload_file: Upload a file", ":material/link: Add a URL"])
        with tab_file:
            f = st.file_uploader(
                "Upload a file",
                type=["pdf", "txt", "md", "csv", "html", "htm", "py", "js", "ts", "go", "java"],
                label_visibility="collapsed",
            )
            if st.button("Add to library", type="primary", disabled=f is None,
                         icon=":material/add:", key="add_file"):
                with st.spinner(f"Processing {f.name}…"):
                    res = api_post(
                        "/ingest",
                        files={"file": (f.name, f.getvalue(), "application/octet-stream")},
                        data={"user_id": SID},
                        timeout=180.0,
                    )
                if res:
                    _ingest_success(res)
        with tab_url:
            url = st.text_input("URL", placeholder="https://arxiv.org/abs/1706.03762",
                                label_visibility="collapsed")
            if st.button("Fetch & add", type="primary", disabled=not url.strip(),
                         icon=":material/add_link:", key="add_url"):
                with st.spinner("Fetching and processing the page…"):
                    res = api_post(
                        "/ingest",
                        data={"url": url.strip(), "doc_type": "web", "user_id": SID},
                        timeout=180.0,
                    )
                if res:
                    _ingest_success(res)

    section("Added in this session")
    docs = (api_get("/documents", quiet=True, limit=50, user_id=SID) or {}).get("documents", [])
    if not docs:
        empty_state("folder_open", "No documents yet",
                    "Files and URLs you add appear here. Existing knowledge-base content is "
                    "already searchable.")
        return
    df = pd.DataFrame([{
        "Name": Path(d.get("filename") or "").name or d.get("source_url") or "—",
        "Type": (d.get("doc_type") or "").upper(),
        "Chunks": d.get("chunk_count") or 0,
        "Added": pd.to_datetime(d.get("ingested_at"), errors="coerce"),
    } for d in docs])
    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={
            "Chunks": st.column_config.NumberColumn(format="%d"),
            "Added": st.column_config.DatetimeColumn(format="MMM D, h:mm a"),
        },
    )


def _ingest_success(res: dict) -> None:
    st.success(
        f"**{res.get('source', 'Document')}** added — {res.get('chunk_count', 0)} chunks "
        f"in {fmt_ms(res.get('total_time_ms'))}.",
        icon=":material/check_circle:",
    )


def page_history() -> None:
    page_header(
        "History",
        "Research sessions",
        "Every research run from this browser session. Select one to read its report and "
        "agent activity.",
    )
    data = api_get("/sessions", quiet=True, limit=100, user_id=SID) or {}
    sessions = data.get("sessions", [])
    if not sessions:
        empty_state("history", "No research yet",
                    "Runs you start on the Research page will appear here.")
        return

    df = pd.DataFrame([{
        "Topic": s.get("topic", ""),
        "Status": STATUS.get(s.get("status"), ("", s.get("status", "")))[1],
        "Agents": s.get("agent_run_count", 0),
        "Started": pd.to_datetime(s.get("created_at"), errors="coerce"),
    } for s in sessions])
    event = st.dataframe(
        df, width="stretch", hide_index=True,
        on_select="rerun", selection_mode="single-row", key="history_table",
        column_config={
            "Topic": st.column_config.TextColumn(width="large"),
            "Agents": st.column_config.NumberColumn(format="%d", width="small"),
            "Started": st.column_config.DatetimeColumn(format="MMM D, h:mm a"),
        },
    )
    rows = event.selection.rows if event and event.selection else []
    if not rows:
        st.caption("Select a row to open its report.")
        return

    s = sessions[rows[0]]
    sid = s["id"]
    st.divider()
    st.markdown(f"#### {s.get('topic', '')}")
    st.html(status_pill(s.get("status", "")))
    tab_report, tab_agents = st.tabs([":material/description: Report", ":material/groups: Agents"])
    with tab_report:
        report = api_get(f"/research/{sid}/report", quiet=True)
        if report:
            render_report(sid, report)
        else:
            empty_state("hourglass_empty", "No report", "This session has not produced a report.")
    with tab_agents:
        runs = api_get(f"/sessions/{sid}/agents", quiet=True) or []
        if runs:
            for run in runs:
                agent_card(run)
        else:
            empty_state("smart_toy", "No agent activity", "No agent steps were recorded.")


def page_system() -> None:
    page_header(
        "System",
        "Platform status",
        "Live health, usage and evaluation metrics for the DocuMind stack.",
    )
    health = api_get("/health", quiet=True) or {}
    metrics = api_get("/metrics", quiet=True) or {}
    analytics = api_get("/documents/analytics", quiet=True) or {}
    q, p, e = metrics.get("qdrant", {}), metrics.get("postgres", {}), metrics.get("embeddings", {})

    section("Usage")
    stat_tiles([
        ("Vectors", f"{q.get('vector_count', 0):,}", f"collection {q.get('collection', '—')}"),
        ("Documents", f"{p.get('documents_ingested', 0):,}", "ingested"),
        ("Research runs", f"{p.get('sessions_total', 0):,}",
         f"{p.get('sessions_completed', 0)} completed · {p.get('sessions_failed', 0)} failed"),
        ("Reports", f"{p.get('reports_generated', 0):,}", "generated"),
        ("Avg. run time", fmt_ms(p.get("avg_workflow_ms")), "end-to-end workflow"),
    ])

    left, right = st.columns(2, gap="large")
    with left:
        section("Services")
        rows = [
            ("API", health.get("status") == "ok", "FastAPI"),
            ("Vector store", health.get("qdrant_ok"), "Qdrant"),
            ("Database", health.get("postgres_ok"), "SQLite"),
        ]
        items = "".join(
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'padding:10px 0;border-bottom:1px solid var(--border-soft)">'
            f'<span><b style="color:var(--text);font-weight:600">{esc(name)}</b>'
            f'<span style="color:var(--muted);font-size:.8rem;margin-left:8px">{esc(tech)}</span>'
            f'</span>{pill_html("good", "Operational") if ok else pill_html("bad", "Down")}</div>'
            for name, ok, tech in rows
        )
        st.html(f"<div>{items}</div>")
    with right:
        section("Configuration")
        st.html(
            '<dl class="dm-kv">'
            f'<dt>LLM provider</dt><dd>{esc(health.get("llm_provider", "—"))}</dd>'
            f'<dt>Embedding mode</dt><dd>{esc(e.get("mode", "—"))}</dd>'
            f'<dt>Embedding model</dt><dd>{esc(e.get("model", "—"))}</dd>'
            f'<dt>Vector dimension</dt><dd>{esc(e.get("dimension", "—"))}</dd>'
            f'<dt>Index status</dt><dd>{esc(q.get("status", "—"))}</dd>'
            "</dl>"
        )

    section("Answer quality (RAGAS)")
    ev = analytics.get("last_eval")
    if ev:
        stat_tiles([
            ("Faithfulness", f"{float(ev.get('faithfulness') or 0):.2f}", "grounded in sources · 0–1"),
            ("Answer relevancy", f"{float(ev.get('answer_relevancy') or 0):.2f}", "on-topic · 0–1"),
            ("Composite", f"{float(ev.get('composite_score') or 0):.2f}",
             f"evaluated {str(ev.get('evaluated_at', ''))[:10]}"),
        ])
    else:
        empty_state("analytics", "No evaluation results yet",
                    "Run python evaluation/eval_runner.py --n-samples 10 to score answer quality.")

    section("Estimated monthly cost")
    cost = pd.DataFrame({
        "Service": ["EC2 t3.micro", "S3 (<5 GB)", "SQS + Lambda + Step Functions",
                    "Textract (~200 pages)", "Comprehend (~500 docs)",
                    "Claude Haiku (~500 runs)", "CloudWatch + SSM"],
        "Monthly": ["Free tier", "Free tier", "Free tier", "~$0.30", "~$0.05", "~$2.00",
                    "Free tier"],
    })
    st.dataframe(cost, width="stretch", hide_index=True)
    st.caption(f"Estimated total ≈ ${analytics.get('cost_estimate_monthly_usd', 2.5):.2f} / month")


# ── Navigation & sidebar ──────────────────────────────────────────────────────

st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "logo_icon.svg"), size="large")

nav = st.navigation(
    {
        "Workspace": [
            st.Page(page_research, title="Research", icon=":material/science:", default=True),
            st.Page(page_ask, title="Ask", icon=":material/forum:", url_path="ask"),
            st.Page(page_library, title="Library", icon=":material/library_books:",
                    url_path="library"),
            st.Page(page_history, title="History", icon=":material/history:",
                    url_path="history"),
        ],
        "Platform": [
            st.Page(page_system, title="System", icon=":material/monitor_heart:",
                    url_path="system"),
        ],
    }
)

with st.sidebar:
    h = cached_health()
    if h is None:
        status_html = pill_html("bad", "API unreachable")
    elif h.get("status") == "ok" and h.get("qdrant_ok"):
        status_html = pill_html("good", "All systems operational")
    else:
        status_html = pill_html("warn", "Degraded performance")
    st.html(
        f'<div class="dm-side-status">{status_html}<br>'
        f'Session <code>{esc(SID[:8])}</code> · resets on refresh</div>'
    )

nav.run()
