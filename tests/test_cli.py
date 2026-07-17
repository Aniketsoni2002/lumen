"""Tests for the CLI entry point (agent/ingest mocked)."""
from __future__ import annotations

from lumen import cli
from lumen.agent.graph import AgentResult


def test_cli_ask_prints_answer_and_trace(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "run_agent",
        lambda q, thread_id=None: AgentResult(
            answer="The answer is 42.",
            tool_calls=["calculator"],
            steps=2,
            reflections=1,
        ),
    )
    rc = cli.main(["ask", "what is 6*7?"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "42" in out
    assert "calculator" in out
    assert "self-corrections: 1" in out


def test_cli_ask_passes_session(monkeypatch, capsys):
    seen = {}

    def _fake(q, thread_id=None):
        seen["thread_id"] = thread_id
        return AgentResult(answer="ok", tool_calls=[], steps=1)

    monkeypatch.setattr(cli, "run_agent", _fake)
    cli.main(["ask", "hi", "--session", "s-1"])
    assert seen["thread_id"] == "s-1"


def test_cli_ingest_reports_count(monkeypatch, capsys):
    monkeypatch.setattr(cli, "ingest_file", lambda path: 9)
    rc = cli.main(["ingest", "some.pdf"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "9 chunks" in out


def test_cli_reset(monkeypatch, capsys):
    called = {"n": 0}
    monkeypatch.setattr(
        cli, "clear_collection", lambda: called.__setitem__("n", called["n"] + 1)
    )
    rc = cli.main(["reset"])
    out = capsys.readouterr().out
    assert rc == 0
    assert called["n"] == 1
    assert "cleared" in out.lower()
