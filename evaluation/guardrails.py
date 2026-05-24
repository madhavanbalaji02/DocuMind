"""Pydantic guardrails for RAG answer validation.

Wraps every LLM generation call with:
1. Citation check
2. Hallucination check (keyword overlap)
3. Refusal check (topic relevance)
4. Length check
5. Structured output via RAGAnswer Pydantic model

GUARDRAILS_ENABLED=true|false (default: true)
"""

from __future__ import annotations

import os
import re
from typing import Optional

from pydantic import BaseModel, field_validator


class RAGAnswer(BaseModel):
    answer: str
    confidence: float         # 0.0-1.0
    sources: list[str]        # cited document names
    is_grounded: bool         # passed hallucination check
    fallback_used: bool       # True if guardrail triggered fallback
    warning: Optional[str]    # any warning message

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


_CITATION_RE = re.compile(r'\[(\d+)\]|Source:|source:|cited|according to', re.IGNORECASE)
_NOT_FOUND_TEMPLATE = (
    "Based on the retrieved documents, I cannot find specific information about '{topic}'. "
    "The documents do contain information about: {related}."
)
_OFF_TOPIC_TEMPLATE = (
    "The indexed documents don't contain information about this topic. "
    "Please upload relevant documents or refine your question."
)


class RAGGuardrails:
    """Validate and guard LLM answers before returning to user."""

    MAX_ANSWER_WORDS = 2000
    MIN_KEYWORD_OVERLAP = 0.3    # hallucination threshold
    MIN_CONTEXT_SIMILARITY = 0.4  # refusal threshold

    def __init__(self) -> None:
        self._enabled = os.environ.get("GUARDRAILS_ENABLED", "true").lower() == "true"

    def validate_answer(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        sources: list[str] | None = None,
        confidence: float = 0.5,
    ) -> RAGAnswer:
        if not self._enabled:
            return RAGAnswer(
                answer=answer,
                confidence=confidence,
                sources=sources or [],
                is_grounded=True,
                fallback_used=False,
                warning=None,
            )

        warning: str | None = None
        fallback_used = False
        final_answer = answer

        # 1. REFUSAL CHECK — if question has no relation to context
        if contexts and not self._is_on_topic(question, contexts):
            return RAGAnswer(
                answer=_OFF_TOPIC_TEMPLATE,
                confidence=0.0,
                sources=[],
                is_grounded=False,
                fallback_used=True,
                warning="off_topic: query not related to indexed documents",
            )

        # 2. HALLUCINATION CHECK — answer claims not in context
        is_grounded = True
        if contexts and answer:
            overlap = self._keyword_overlap(answer, contexts)
            if overlap < self.MIN_KEYWORD_OVERLAP:
                is_grounded = False
                related = self._extract_topics(contexts)
                topic = self._extract_topic(question)
                final_answer = _NOT_FOUND_TEMPLATE.format(topic=topic, related=related)
                fallback_used = True
                warning = f"hallucination_risk: keyword overlap {overlap:.2f} < {self.MIN_KEYWORD_OVERLAP}"

        # 3. CITATION CHECK — add note if no citations
        if not _CITATION_RE.search(final_answer):
            if not warning:
                warning = "no_citations: answer lacks explicit source references"
            final_answer += "\n\n*Note: answer based on retrieved context only.*"

        # 4. LENGTH CHECK — summarize if too long
        word_count = len(final_answer.split())
        if word_count > self.MAX_ANSWER_WORDS:
            final_answer = " ".join(final_answer.split()[: self.MAX_ANSWER_WORDS]) + "…"
            if not warning:
                warning = f"truncated: answer was {word_count} words"

        return RAGAnswer(
            answer=final_answer,
            confidence=confidence if is_grounded else confidence * 0.5,
            sources=sources or self._extract_sources(final_answer),
            is_grounded=is_grounded,
            fallback_used=fallback_used,
            warning=warning,
        )

    def _is_on_topic(self, question: str, contexts: list[str]) -> bool:
        """Check if the question has at least some keyword overlap with contexts."""
        q_words = set(self._tokenize(question))
        for ctx in contexts:
            ctx_words = set(self._tokenize(ctx))
            if q_words & ctx_words:
                return True
        return False

    def _keyword_overlap(self, answer: str, contexts: list[str]) -> float:
        """Fraction of answer content words that appear in contexts."""
        answer_words = set(self._tokenize(answer))
        if not answer_words:
            return 1.0
        ctx_words: set[str] = set()
        for ctx in contexts:
            ctx_words.update(self._tokenize(ctx))
        overlap = len(answer_words & ctx_words) / len(answer_words)
        return overlap

    def _tokenize(self, text: str) -> list[str]:
        """Simple content-word tokenizer (removes stopwords)."""
        STOPWORDS = {
            "the", "a", "an", "is", "it", "in", "of", "to", "and", "or",
            "that", "this", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "with", "for",
            "on", "at", "by", "from", "as", "into", "about", "than",
        }
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w not in STOPWORDS]

    def _extract_topics(self, contexts: list[str]) -> str:
        """Extract a brief description of what the contexts cover."""
        combined = " ".join(contexts[:2])[:500]
        words = self._tokenize(combined)
        # Return top 5 most common non-stopwords
        from collections import Counter
        common = Counter(words).most_common(5)
        return ", ".join(w for w, _ in common)

    def _extract_topic(self, question: str) -> str:
        """Extract the main topic from a question."""
        # Remove question words
        cleaned = re.sub(r'^(what|how|why|when|where|who|which|is|are|does|do)\s+', '',
                         question.lower().strip(), flags=re.IGNORECASE)
        return cleaned.rstrip("?").strip()[:60]

    def _extract_sources(self, answer: str) -> list[str]:
        """Extract source citations from the answer text."""
        sources: list[str] = []
        # Look for [N] Source: filename patterns
        for match in re.finditer(r'\[(\d+)\]\s+([^\n—]+)', answer):
            sources.append(match.group(2).strip())
        return sources[:5]


# Singleton
_guardrails: RAGGuardrails | None = None


def get_guardrails() -> RAGGuardrails:
    global _guardrails
    if _guardrails is None:
        _guardrails = RAGGuardrails()
    return _guardrails
