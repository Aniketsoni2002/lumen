"""Tests for the self-reflection grading helpers (pure text logic)."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentrag.agent import reflection


def test_parse_grade_grounded():
    assert reflection.parse_grade("GROUNDED") is True
    assert reflection.parse_grade("grounded\nsome reasoning") is True


def test_parse_grade_ungrounded():
    assert reflection.parse_grade("UNGROUNDED") is False
    assert reflection.parse_grade("  ungrounded - claim X unsupported") is False


def test_parse_grade_fails_open_on_garbage():
    # Unclear grader output must not block a legitimate answer.
    assert reflection.parse_grade("I'm not sure honestly") is True
    assert reflection.parse_grade("") is True


def test_collect_evidence_only_tool_messages():
    msgs = [
        HumanMessage(content="q"),
        AIMessage(content="thinking"),
        ToolMessage(content="fact one", tool_call_id="1"),
        ToolMessage(content="fact two", tool_call_id="2"),
    ]
    evidence = reflection.collect_evidence(msgs)
    assert "fact one" in evidence and "fact two" in evidence
    assert "thinking" not in evidence


def test_first_question_returns_latest_human():
    msgs = [
        HumanMessage(content="old question"),
        AIMessage(content="answer"),
        HumanMessage(content="new question"),
    ]
    assert reflection.first_question(msgs) == "new question"


def test_build_reflection_prompt_includes_all_parts():
    prompt = reflection.build_reflection_prompt("Q?", "E", "A")
    assert "Q?" in prompt and "E" in prompt and "A" in prompt
    assert "GROUNDED" in prompt  # instructs the grader


def test_build_reflection_prompt_handles_empty_evidence():
    prompt = reflection.build_reflection_prompt("Q?", "", "A")
    assert "no tool evidence" in prompt
