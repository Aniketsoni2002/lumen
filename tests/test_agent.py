"""Tests for the LangGraph agent loop with scripted fake LLMs.

The fakes let us exercise the full graph — agent -> tools -> agent -> reflection
-> END — including the step cap, the self-reflection grader, and multi-turn
memory, all without Ollama.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from lumen import config
from lumen.agent import graph


@pytest.fixture(autouse=True)
def _clear_settings(tmp_path, monkeypatch):
    """Every test starts from clean settings and an isolated memory DB."""
    config.get_settings.cache_clear()
    # Use the persistent SQLite backend at a throwaway path so the memory test
    # exercises real cross-call persistence; reset the cached checkpointer so it
    # is rebuilt against that path.
    monkeypatch.setenv("LUMEN_MEMORY_BACKEND", "sqlite")
    monkeypatch.setenv("LUMEN_MEMORY_DB", str(tmp_path / "mem.sqlite"))
    graph._CHECKPOINTER = None
    yield
    graph._CHECKPOINTER = None
    config.get_settings.cache_clear()


def _calc_call(expr: str, call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "calculator", "args": {"expression": expr},
             "id": call_id, "type": "tool_call"}
        ],
    )


class _ScriptedLLM:
    """Yields a scripted sequence of responses across successive .invoke calls.

    ``bind_tools`` is accepted (and ignored) so it matches the real signature;
    the same object is returned for both the agent and grader roles, and we key
    the grader response off the prompt content.
    """

    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    def invoke(self, messages):
        # Grader calls include the reflection prompt as a single human message.
        text = messages[-1].content if messages else ""
        if isinstance(text, str) and "fact-checking grader" in text:
            return AIMessage(content="GROUNDED")
        resp = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return resp


def _patch_llm(monkeypatch, scripted):
    monkeypatch.setattr(graph, "_build_llm", lambda bind_tools=True: scripted)


def test_agent_uses_tool_then_answers(monkeypatch):
    monkeypatch.setenv("LUMEN_ENABLE_REFLECTION", "false")
    scripted = _ScriptedLLM([_calc_call("6 * 7"), AIMessage(content="It is 42.")])
    _patch_llm(monkeypatch, scripted)

    result = graph.run_agent("What is 6 times 7?")

    assert "42" in result.answer
    assert "calculator" in result.tool_calls
    assert result.steps >= 2


def test_agent_answers_without_tools(monkeypatch):
    monkeypatch.setenv("LUMEN_ENABLE_REFLECTION", "false")
    scripted = _ScriptedLLM([AIMessage(content="Hello! How can I help?")])
    _patch_llm(monkeypatch, scripted)

    result = graph.run_agent("hi")

    assert "help" in result.answer.lower()
    assert result.tool_calls == []
    assert result.steps == 1


def test_agent_respects_step_cap(monkeypatch):
    monkeypatch.setenv("LUMEN_ENABLE_REFLECTION", "false")
    monkeypatch.setenv("LUMEN_MAX_AGENT_STEPS", "3")

    class _LoopingLLM:
        def invoke(self, _messages):
            return _calc_call("1 + 1", call_id="loop")

    monkeypatch.setattr(graph, "_build_llm", lambda bind_tools=True: _LoopingLLM())

    result = graph.run_agent("loop forever?")

    assert result.answer
    assert result.steps <= 4


def test_reflection_grounded_passes_through(monkeypatch):
    # Reflection ON; grader returns GROUNDED, so the answer is returned as-is.
    monkeypatch.setenv("LUMEN_ENABLE_REFLECTION", "true")
    scripted = _ScriptedLLM([AIMessage(content="The capital is Paris.")])
    _patch_llm(monkeypatch, scripted)

    result = graph.run_agent("What is the capital of France?")

    assert "Paris" in result.answer
    assert result.reflections == 0


def test_reflection_ungrounded_triggers_one_retry(monkeypatch):
    monkeypatch.setenv("LUMEN_ENABLE_REFLECTION", "true")

    class _GraderThenFix:
        """First answer is judged UNGROUNDED; the agent then corrects it."""

        def __init__(self):
            self.answer_turns = 0

        def invoke(self, messages):
            text = messages[-1].content if messages else ""
            if isinstance(text, str) and "fact-checking grader" in text:
                # Ungrounded on the first grading, grounded after.
                return AIMessage(
                    content="UNGROUNDED" if self.answer_turns == 1 else "GROUNDED"
                )
            self.answer_turns += 1
            if self.answer_turns == 1:
                return AIMessage(content="A guessed, unsupported answer.")
            return AIMessage(content="A corrected, honest answer.")

    grader = _GraderThenFix()
    monkeypatch.setattr(graph, "_build_llm", lambda bind_tools=True: grader)

    result = graph.run_agent("something tricky")

    assert result.reflections == 1
    assert "corrected" in result.answer


def test_memory_preserves_turns_across_calls(monkeypatch):
    monkeypatch.setenv("LUMEN_ENABLE_REFLECTION", "false")

    class _EchoLLM:
        """Answers, and on the 2nd turn can 'see' the earlier human message."""

        def invoke(self, messages):
            human = [m for m in messages if m.__class__.__name__ == "HumanMessage"]
            return AIMessage(content=f"seen {len(human)} question(s)")

    monkeypatch.setattr(graph, "_build_llm", lambda bind_tools=True: _EchoLLM())

    tid = "test-thread-1"
    r1 = graph.run_agent("first question", thread_id=tid)
    r2 = graph.run_agent("second question", thread_id=tid)

    assert "1 question" in r1.answer
    # Second call must see BOTH questions -> memory persisted across calls.
    assert "2 question" in r2.answer
