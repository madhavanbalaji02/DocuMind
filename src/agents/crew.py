"""Assembles and runs the full DocuMind CrewAI research crew."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from crewai import Crew, Process, Task

from src.agents.analyst import build_analyst
from src.agents.critic import build_critic
from src.agents.researcher import build_researcher
from src.agents.writer import build_writer
from src.core.logging import get_logger
from src.db import connection as db

logger = get_logger(__name__)


@dataclass
class Citation:
    source: str
    chunk_id: str
    excerpt: str


@dataclass
class CrewResult:
    draft_report: str
    citations: list[Citation] = field(default_factory=list)
    agent_run_ids: list[str] = field(default_factory=list)


class ResearchCrew:

    def __init__(self) -> None:
        self.researcher = build_researcher()
        self.analyst = build_analyst()
        self.writer = build_writer()
        self.critic = build_critic()

    def _build_tasks(self, topic: str, context: str) -> list[Task]:
        research_task = Task(
            description=(
                f"Research the following topic comprehensively:\n\n**Topic:** {topic}\n\n"
                f"**Knowledge base context:**\n{context[:3000]}\n\n"
                f"Use search_knowledge_base to find additional relevant information. "
                f"Supplement with web searches for recent developments. "
                f"Cover history, mechanisms, current state, and open questions."
            ),
            expected_output=(
                "A structured research summary with:\n"
                "(1) Direct answers to each aspect of the research topic\n"
                "(2) Key facts with source citations in [source: URL] format\n"
                "(3) Conflicting viewpoints or open debates, if any\n"
                "(4) Quantitative data and statistics where available\n"
                "Minimum 400 words. Every claim must cite a source."
            ),
            agent=self.researcher,
        )

        analysis_task = Task(
            description=(
                f"Analyse the research findings for: {topic}\n\n"
                f"Query the database for related past reports using query_database. "
                f"Identify patterns, contradictions, and evidence gaps."
            ),
            expected_output=(
                "A data analysis section with:\n"
                "(1) Quantitative findings (numbers, percentages, dates)\n"
                "(2) Comparison tables in markdown where applicable\n"
                "(3) Statistical context and trend analysis\n"
                "(4) Confidence assessment for major claims: High/Medium/Low\n"
                "Format as markdown with ## headers."
            ),
            agent=self.analyst,
            context=[research_task],
        )

        writing_task = Task(
            description=(
                f"Write a comprehensive research report on: {topic}\n\n"
                f"Synthesise the researcher's and analyst's output. "
                f"Every factual claim must carry an inline [N] citation. "
                f"The report must be self-contained and publication-ready."
            ),
            expected_output=(
                "A complete markdown research report:\n\n"
                "# [Descriptive Title]\n\n"
                "## Executive Summary\n(100-150 words)\n\n"
                "## Key Findings\n(bullets with [N] citations)\n\n"
                "## Detailed Analysis\n(4+ paragraphs, citations, data)\n\n"
                "## Conclusion\n(synthesis and implications)\n\n"
                "## References\n([N] full source URL or title)\n\n"
                "Minimum 800 words total."
            ),
            agent=self.writer,
            context=[research_task, analysis_task],
        )

        critique_task = Task(
            description=(
                f"Review the draft report on '{topic}' for accuracy and completeness.\n\n"
                f"Verify every claim against source material. "
                f"Flag unsupported assertions, logical gaps, and missing citations."
            ),
            expected_output=(
                "VERDICT: PASS or FAIL\n"
                "SCORE: X/10\n\n"
                "ISSUES:\n1. [specific problem]\n\n"
                "SUGGESTIONS:\n1. [actionable fix]\n\n"
                "CITATION CHECK: confirm all claims cite sources, or list missing ones.\n\n"
                "If PASS: state that all major claims are supported by provided sources."
            ),
            agent=self.critic,
            context=[writing_task],
        )

        return [research_task, analysis_task, writing_task, critique_task]

    async def run(self, topic: str, context: str, session_id: str) -> CrewResult:
        logger.info("ResearchCrew starting session=%s topic=%r", session_id, topic)
        tasks = self._build_tasks(topic, context)

        crew = Crew(
            agents=[self.researcher, self.analyst, self.writer, self.critic],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        t_start = asyncio.get_event_loop().time()
        loop = asyncio.get_event_loop()
        crew_output = await loop.run_in_executor(None, crew.kickoff)
        elapsed_ms = int((asyncio.get_event_loop().time() - t_start) * 1000)

        draft_report = str(tasks[2].output) if tasks[2].output else str(crew_output)

        agent_run_ids: list[str] = []
        agent_names = ["researcher", "analyst", "writer", "critic"]
        for task, name in zip(tasks, agent_names):
            run_id = str(uuid.uuid4())
            agent_run_ids.append(run_id)
            output_text = str(task.output) if task.output else ""
            try:
                await db.execute(
                    """
                    INSERT INTO agent_runs
                        (id, session_id, agent_name, input, output, duration_ms, created_at)
                    VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7)
                    """,
                    run_id, session_id, name, topic,
                    output_text[:10000], elapsed_ms // 4,
                    datetime.now(timezone.utc),
                )
            except Exception as exc:
                logger.warning("Failed to save agent_run %s: %s", name, exc)

        logger.info(
            "ResearchCrew complete session=%s elapsed_ms=%d draft_len=%d",
            session_id, elapsed_ms, len(draft_report),
        )
        return CrewResult(draft_report=draft_report, agent_run_ids=agent_run_ids)
