"""Assembles and runs the full DocuMind CrewAI research crew."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from crewai import Crew, Process, Task

from src.agents.analyst import build_analyst
from src.agents.critic import build_critic
from src.agents.researcher import build_researcher
from src.agents.writer import build_writer
from src.db import connection as db

logger = logging.getLogger(__name__)


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
    """Orchestrates the four-agent research crew."""

    def __init__(self) -> None:
        self.researcher = build_researcher()
        self.analyst = build_analyst()
        self.writer = build_writer()
        self.critic = build_critic()

    def _build_tasks(self, topic: str, context: str) -> list[Task]:
        research_task = Task(
            description=(
                f"Research the following topic comprehensively:\n\n**Topic:** {topic}\n\n"
                f"**Provided context from knowledge base:**\n{context[:3000]}\n\n"
                f"Use search_knowledge_base to find additional relevant information. "
                f"Supplement with web searches for recent developments. "
                f"Produce a detailed findings summary (500-800 words) with source citations."
            ),
            expected_output=(
                "A structured research findings summary with:\n"
                "1. Key facts and data points\n"
                "2. Main arguments and perspectives\n"
                "3. Important statistics or evidence\n"
                "4. All sources cited in [Author/URL] format"
            ),
            agent=self.researcher,
        )

        analysis_task = Task(
            description=(
                f"Analyse the research findings for the topic: {topic}\n\n"
                f"Query the database for relevant past reports using query_database. "
                f"Identify patterns, contradictions, and gaps in the evidence. "
                f"Provide quantitative analysis where possible."
            ),
            expected_output=(
                "An analytical report (300-500 words) containing:\n"
                "1. Key patterns and trends identified\n"
                "2. Data gaps or inconsistencies\n"
                "3. Quantitative insights from database queries\n"
                "4. Confidence assessment for major claims"
            ),
            agent=self.analyst,
            context=[research_task],
        )

        writing_task = Task(
            description=(
                f"Write a comprehensive research report on: {topic}\n\n"
                f"Use the researcher's findings and analyst's insights to produce "
                f"a well-structured markdown report (800-1200 words). "
                f"Include: Executive Summary, Background, Key Findings, Analysis, "
                f"Conclusions, and a Sources section. Cite every claim."
            ),
            expected_output=(
                "A complete markdown research report with:\n"
                "- # Title\n"
                "- ## Executive Summary\n"
                "- ## Background\n"
                "- ## Key Findings\n"
                "- ## Analysis\n"
                "- ## Conclusions\n"
                "- ## Sources\n"
                "All claims cited with inline references like [1], [2], etc."
            ),
            agent=self.writer,
            context=[research_task, analysis_task],
        )

        critique_task = Task(
            description=(
                f"Review the draft research report on '{topic}' for quality.\n\n"
                f"Check every factual claim against the source material. "
                f"Verify that all citations are present and accurate. "
                f"Identify logical gaps, unsupported assertions, or missing context. "
                f"If the report passes, confirm it. If not, provide specific corrections."
            ),
            expected_output=(
                "A structured critique with:\n"
                "1. Overall quality verdict (APPROVED / NEEDS REVISION)\n"
                "2. List of issues found (if any)\n"
                "3. Specific corrections or additions needed\n"
                "4. Final note on citation completeness"
            ),
            agent=self.critic,
            context=[writing_task],
        )

        return [research_task, analysis_task, writing_task, critique_task]

    async def run(self, topic: str, context: str, session_id: str) -> CrewResult:
        """Execute the full crew pipeline and return the draft report."""
        logger.info("[ResearchCrew] starting session=%s topic=%r", session_id, topic)
        tasks = self._build_tasks(topic, context)

        crew = Crew(
            agents=[self.researcher, self.analyst, self.writer, self.critic],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        t_start = asyncio.get_event_loop().time()

        # CrewAI kickoff is synchronous — run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        crew_output = await loop.run_in_executor(None, crew.kickoff)
        elapsed_ms = int((asyncio.get_event_loop().time() - t_start) * 1000)

        # The writing task output is the third task result
        draft_report = str(tasks[2].output) if tasks[2].output else str(crew_output)

        # Persist agent runs to postgres
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
                    run_id,
                    session_id,
                    name,
                    topic,
                    output_text[:10000],
                    elapsed_ms // 4,
                    datetime.now(timezone.utc),
                )
            except Exception as exc:
                logger.warning("Failed to save agent_run for %s: %s", name, exc)

        logger.info(
            "[ResearchCrew] completed session=%s elapsed=%dms draft_len=%d",
            session_id,
            elapsed_ms,
            len(draft_report),
        )
        return CrewResult(
            draft_report=draft_report,
            agent_run_ids=agent_run_ids,
        )
