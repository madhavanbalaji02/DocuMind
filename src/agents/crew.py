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

CREW_TIMEOUT_SECONDS = 240  # 4-minute hard cap on the full crew run


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
    error: str | None = None


class ResearchCrew:

    def __init__(self) -> None:
        self.researcher = build_researcher()
        self.analyst = build_analyst()
        self.writer = build_writer()
        self.critic = build_critic()

    def _build_tasks(self, topic: str, context: str) -> list[Task]:
        research_task = Task(
            description=(
                f"Research this topic: {topic}\n\n"
                f"Context from knowledge base:\n{context[:2000]}\n\n"
                f"Use search_knowledge_base once or twice for the most relevant angles. "
                f"Be concise and focused — do not make more than 3 tool calls total."
            ),
            expected_output=(
                "Research summary (300-500 words) with:\n"
                "1. Key facts with [source: URL] citations\n"
                "2. Main mechanisms or concepts explained\n"
                "3. Practical applications or significance"
            ),
            agent=self.researcher,
        )

        analysis_task = Task(
            description=(
                f"Analyse findings for: {topic}\n\n"
                f"Identify 3-5 key insights. Be analytical and concise."
            ),
            expected_output=(
                "Analysis section (200-300 words) with:\n"
                "1. Key patterns and trends\n"
                "2. Comparative context\n"
                "3. Confidence assessment (High/Medium/Low) for major claims"
            ),
            agent=self.analyst,
            context=[research_task],
        )

        writing_task = Task(
            description=(
                f"Write a complete research report on: {topic}\n\n"
                f"Use researcher and analyst findings. Every claim needs a citation."
            ),
            expected_output=(
                "Complete markdown report:\n"
                "# Title\n## Executive Summary\n## Key Findings\n"
                "## Analysis\n## Conclusion\n## References\n"
                "Minimum 600 words. All claims cited [N]."
            ),
            agent=self.writer,
            context=[research_task, analysis_task],
        )

        critique_task = Task(
            description=(
                f"Review the draft report on '{topic}' for accuracy.\n"
                f"Check claims against sources. Be specific and brief."
            ),
            expected_output=(
                "VERDICT: PASS or FAIL\nSCORE: X/10\n"
                "ISSUES: (numbered, or 'None')\n"
                "SUGGESTIONS: (numbered, or 'None')"
            ),
            agent=self.critic,
            context=[writing_task],
        )

        return [research_task, analysis_task, writing_task, critique_task]

    def _kickoff_sync(self, topic: str, context: str) -> str:
        """Run crew.kickoff() synchronously — called from run_in_executor."""
        tasks = self._build_tasks(topic, context)
        crew = Crew(
            agents=[self.researcher, self.analyst, self.writer, self.critic],
            tasks=tasks,
            process=Process.sequential,
            verbose=False,  # reduce noise
        )
        crew.kickoff()
        return str(tasks[2].output) if tasks[2].output else "Report generation incomplete."

    async def run(self, topic: str, context: str, session_id: str) -> CrewResult:
        logger.info("ResearchCrew starting session=%s", session_id)
        loop = asyncio.get_event_loop()

        try:
            draft_report = await asyncio.wait_for(
                loop.run_in_executor(None, self._kickoff_sync, topic, context),
                timeout=CREW_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "ResearchCrew TIMED OUT after %ds session=%s",
                CREW_TIMEOUT_SECONDS, session_id,
            )
            return CrewResult(
                draft_report=(
                    f"# Research Report: {topic}\n\n"
                    "## Note\n\nThe multi-agent research workflow reached its time limit. "
                    "The retrieved context below represents the best available information.\n\n"
                    f"*Topic: {topic}*\n\n*Context collected:* {context[:1000]}"
                ),
                error="crew_timeout",
            )
        except Exception as exc:
            logger.error("ResearchCrew failed session=%s: %s", session_id, exc)
            return CrewResult(
                draft_report=f"# Research: {topic}\n\nAgent execution encountered an error: {exc}",
                error=str(exc),
            )

        # Persist agent runs
        agent_run_ids: list[str] = []
        for name in ["researcher", "analyst", "writer", "critic"]:
            run_id = str(uuid.uuid4())
            agent_run_ids.append(run_id)
            try:
                await db.execute(
                    """
                    INSERT INTO agent_runs
                        (id, session_id, agent_name, input, output, created_at)
                    VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6)
                    """,
                    run_id, session_id, name, topic,
                    draft_report[:5000] if name == "writer" else "",
                    datetime.now(timezone.utc),
                )
            except Exception as exc:
                logger.warning("agent_run save failed %s: %s", name, exc)

        logger.info("ResearchCrew complete session=%s", session_id)
        return CrewResult(draft_report=draft_report, agent_run_ids=agent_run_ids)
