"""Tests for the LangGraph agent loop with a scripted fake LLM.

We replace the bound LLM with a fake that first asks for the calculator tool,
then (after seeing the tool result) returns a final answer. This exercises the
full graph: agent -> tools -> agent -> END, the step counter, and the
answer/trace extraction — all without Ollama.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from agentrag.agent import graph


class _ScriptedLLM:
    """Returns a tool-calling message first, then a plain answer."""

    def __init__(self):
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "6 * 7"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="The answer is 42.")


def test_agent_uses_tool_then_answers(monkeypatch):
    scripted = _ScriptedLLM()
    # _build_llm returns the object the agent node calls .invoke() on.
    monkeypatch.setattr(graph, "_build_llm", lambda: scripted)

    result = graph.run_agent("What is 6 times 7?")

    assert "42" in result.answer
    assert "calculator" in result.tool_calls
    assert result.steps >= 2  # at least one tool turn + final answer turn


def test_agent_answers_without_tools(monkeypatch):
    class _DirectLLM:
        def invoke(self, _messages):
            return AIMessage(content="Hello! How can I help?")

    monkeypatch.setattr(graph, "_build_llm", lambda: _DirectLLM())

    result = graph.run_agent("hi")

    assert "help" in result.answer.lower()
    assert result.tool_calls == []
    assert result.steps == 1


def test_agent_respects_step_cap(monkeypatch):
    # An LLM that NEVER stops asking for a tool would loop forever without a cap.
    class _LoopingLLM:
        def invoke(self, _messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "1 + 1"},
                        "id": "call_x",
                        "type": "tool_call",
                    }
                ],
            )

    monkeypatch.setattr(graph, "_build_llm", lambda: _LoopingLLM())
    monkeypatch.setenv("AGENTRAG_MAX_AGENT_STEPS", "3")
    from agentrag import config

    config.get_settings.cache_clear()

    result = graph.run_agent("loop forever?")

    # It must terminate (fallback answer) and not exceed the cap by much.
    assert result.answer  # non-empty
    assert result.steps <= 4
    config.get_settings.cache_clear()
