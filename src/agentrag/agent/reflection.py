"""Self-reflection: grade the agent's answer against the evidence it gathered.

This is a lightweight guardrail against the most damaging RAG failure mode — a
confident answer that isn't actually supported by the retrieved evidence. After
the agent produces an answer, a grader LLM call classifies it GROUNDED or
UNGROUNDED. The grading is pure text-in/text-out, so the parsing logic is
unit-tested without any model.
"""
from __future__ import annotations

from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage

from agentrag.agent.prompts import REFLECTION_PROMPT
from agentrag.utils.logging import get_logger

logger = get_logger("reflection")


def collect_evidence(messages: list[AnyMessage]) -> str:
    """Concatenate all tool outputs — the evidence the agent had to work with."""
    parts = [m.content for m in messages if isinstance(m, ToolMessage)]
    return "\n\n".join(str(p) for p in parts).strip()


def first_question(messages: list[AnyMessage]) -> str:
    """The user's most recent question (last human turn)."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def parse_grade(text: str) -> bool:
    """Return True if the grader said GROUNDED.

    Robust to extra prose: we look at the first non-empty line and default to
    GROUNDED (fail-open) if the grader is unclear, so reflection never blocks a
    legitimate answer on a parsing quirk.
    """
    for line in text.splitlines():
        token = line.strip().upper()
        if not token:
            continue
        if token.startswith("UNGROUNDED"):
            return False
        if token.startswith("GROUNDED"):
            return True
        break
    return True


def build_reflection_prompt(question: str, evidence: str, answer: str) -> str:
    return REFLECTION_PROMPT.format(
        question=question or "(unknown)",
        evidence=evidence or "(no tool evidence was gathered)",
        answer=answer,
    )
