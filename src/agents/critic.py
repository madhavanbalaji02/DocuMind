"""CrewAI Critic agent — uses Bedrock Claude Haiku (fallback: direct Anthropic)."""

from __future__ import annotations

import os

from crewai import LLM, Agent


def _build_llm() -> LLM:
    provider = os.environ.get("LLM_PROVIDER", "bedrock")
    if provider == "bedrock":
        return LLM(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            aws_region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
        )
    return LLM(
        model="anthropic/claude-haiku-4-5",
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )


def build_critic() -> Agent:
    return Agent(
        role="Quality Critic",
        goal="Review the draft report and give a PASS/FAIL verdict with score.",
        backstory=(
            "You review reports efficiently. No tool calls — you judge "
            "the draft against the context already provided."
        ),
        tools=[],
        llm=_build_llm(),
        verbose=False,
        max_iter=2,
        max_rpm=10,
    )
