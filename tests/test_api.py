"""API tests with the agent mocked out."""
from __future__ import annotations

import io

from fastapi.testclient import TestClient

from lumen.agent.graph import AgentResult
from lumen.api import main


def _client() -> TestClient:
    return TestClient(main.app)


def test_health_lists_tools():
    resp = _client().get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "calculator" in body["tools"]
    assert "web_search" in body["tools"]


def test_ask_returns_answer_and_trace(monkeypatch):
    monkeypatch.setattr(
        main,
        "run_agent",
        lambda q, thread_id=None: AgentResult(
            answer="42", tool_calls=["calculator"], steps=2, reflections=1
        ),
    )
    resp = _client().post("/ask", json={"question": "6*7?"})
    assert resp.status_code == 200
    assert resp.json() == {
        "answer": "42",
        "tools_used": ["calculator"],
        "steps": 2,
        "reflections": 1,
    }


def test_ask_passes_session_id_for_memory(monkeypatch):
    seen = {}

    def _fake(q, thread_id=None):
        seen["thread_id"] = thread_id
        return AgentResult(answer="ok", tool_calls=[], steps=1)

    monkeypatch.setattr(main, "run_agent", _fake)
    _client().post("/ask", json={"question": "hi", "session_id": "abc-123"})
    assert seen["thread_id"] == "abc-123"


def test_ask_stream_emits_sse_events(monkeypatch):
    def _fake_stream(q, thread_id=None):
        yield {"type": "step", "node": "agent"}
        yield {"type": "tool", "name": "calculator"}
        yield {
            "type": "final",
            "result": AgentResult(
                answer="42", tool_calls=["calculator"], steps=2, reflections=0
            ),
        }

    monkeypatch.setattr(main, "stream_agent", _fake_stream)
    resp = _client().post("/ask/stream", json={"question": "6*7?"})
    assert resp.status_code == 200
    body = resp.text
    assert "data:" in body
    assert '"type": "tool"' in body
    assert '"answer": "42"' in body  # final frame carries the answer


def test_ask_rejects_empty_question():
    resp = _client().post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ingest_rejects_unsupported_type():
    files = {"file": ("bad.png", io.BytesIO(b"data"), "image/png")}
    resp = _client().post("/ingest", files=files)
    assert resp.status_code == 415


def test_ingest_indexes_supported_file(monkeypatch, tmp_path):
    from lumen import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("LUMEN_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main, "ingest_file", lambda path: 5)

    files = {"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    resp = _client().post("/ingest", files=files)

    assert resp.status_code == 200
    assert resp.json() == {"filename": "notes.txt", "chunks_indexed": 5}
    config.get_settings.cache_clear()
