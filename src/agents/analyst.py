"""CrewAI Analyst agent definition."""

from __future__ import annotations

import os

from crewai import Agent, LLM

from src.agents.tools import query_database, search_knowledge_base


def build_analyst() -> Agent:
    llm = LLM(
        model="anthropic/claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    return Agent(
        role="Data Analyst",
        goal=(
            "Analyse structured data from the database and extracted knowledge to identify "
            "patterns, trends, and quantitative insights. Translate raw data into "
            "actionable analytical findings."
        ),
        backstory=(
            "You are a precision-focused data analyst who spent a decade turning messy "
            "datasets into clear narratives. You write SQL fluently, spot anomalies "
            "immediately, and always quantify your claims. You trust data over opinion "
            "and will push back when assertions lack numerical backing."
        ),
        tools=[query_database, search_knowledge_base],
        llm=llm,
        verbose=True,
        max_iter=4,
    )
