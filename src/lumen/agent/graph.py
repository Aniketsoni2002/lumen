"""The agentic reasoning loop, built on LangGraph.

Graph shape (with the advanced features enabled)::

    ┌─────────┐  tool calls?  ┌──────────┐
    │  agent  │ ─────yes────▶ │  tools   │
    │ (LLM)   │ ◀─────────────│ (execute)│
    └────┬────┘               └──────────┘
         │ no tool calls
         ▼
    ┌────────────┐  UNGROUNDED & retries left   (adds a correction request,
    │ reflection │ ───────────────────────────▶  loops back to agent)
    │  (grader)  │
    └────┬───────┘
         │ GROUNDED / out of retries
         ▼
        END → final answer

Features
--------
* **ReAct tool loop** with a hard step cap (a confused model can't loop forever).
* **Conversation memory** via a LangGraph checkpointer keyed by ``thread_id``,
  so follow-up questions see prior turns.
* **Self-reflection**: a grader checks the answer against the gathered evidence
  and, on an UNGROUNDED verdict, gives the agent one shot to correct itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from lumen.agent import reflection
from lumen.agent.llm import build_chat_model
from lumen.agent.prompts import SYSTEM_PROMPT
from lumen.config import get_settings
from lumen.tools import ALL_TOOLS
from lumen.utils.logging import get_logger

logger = get_logger("agent")


class AgentState(dict):
    """Graph state.

    ``messages`` accumulates (reducer merges new messages in).
    ``steps`` bounds the tool loop.
    ``reflections`` bounds how many self-correction passes we allow.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    reflections: int


def _get_checkpointer():
    """Return the conversation-memory checkpointer.

    Two backends, chosen by ``LUMEN_MEMORY_BACKEND``:

    * ``"memory"`` (default) — an in-process saver. Thread-safe, no I/O, and the
      right choice for the Streamlit UI and cloud deploys, where the app is one
      long-lived process served across many threads. A shared on-disk SQLite
      connection there can deadlock across those threads.
    * ``"sqlite"`` — persists to disk so history survives across *separate*
      processes (e.g. multiple CLI ``ask`` runs). Best for single-threaded use.

    Cached for the process lifetime.
    """
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        settings = get_settings()
        backend = settings.memory_backend.lower().strip()

        if backend == "sqlite":
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            settings.memory_db.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(settings.memory_db), check_same_thread=False
            )
            _CHECKPOINTER = SqliteSaver(conn)
        else:
            from langgraph.checkpoint.memory import MemorySaver

            _CHECKPOINTER = MemorySaver()
    return _CHECKPOINTER


_CHECKPOINTER = None


def _build_llm(bind_tools: bool = True):
    """Build the configured chat model (Ollama or Groq), optionally tool-bound."""
    llm = build_chat_model()
    return llm.bind_tools(ALL_TOOLS) if bind_tools else llm


def _agent_node(state: AgentState) -> dict:
    """LLM turn: decide whether to call a tool or produce an answer."""
    llm = _build_llm(bind_tools=True)
    response = llm.invoke(state["messages"])
    return {"messages": [response], "steps": state.get("steps", 0) + 1}


def _tools_condition(state: AgentState) -> str:
    """After the agent turn: go to tools, to reflection, or straight to END."""
    settings = get_settings()
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)

    if tool_calls and state.get("steps", 0) < settings.max_agent_steps:
        names = ", ".join(tc["name"] for tc in tool_calls)
        logger.info("Agent step %d -> tools: %s", state.get("steps", 0), names)
        return "tools"

    # No more tools requested: the agent has produced an answer.
    if settings.enable_reflection:
        return "reflection"
    return END


def _reflection_node(state: AgentState) -> dict:
    """Grade the answer against gathered evidence; request a fix if ungrounded."""
    messages = state["messages"]
    answer = messages[-1].content if messages else ""
    evidence = reflection.collect_evidence(messages)
    question = reflection.first_question(messages)

    grader = _build_llm(bind_tools=False)
    prompt = reflection.build_reflection_prompt(question, evidence, str(answer))
    verdict_text = grader.invoke(
        [HumanMessage(content=prompt)]
    ).content
    grounded = reflection.parse_grade(str(verdict_text))

    logger.info(
        "Reflection verdict: %s", "GROUNDED" if grounded else "UNGROUNDED"
    )

    reflections = state.get("reflections", 0)

    # Grounded, or we've already spent our one corrective pass: accept as-is.
    if grounded or reflections >= 1:
        return {"reflections": reflections}

    # Ungrounded and retries remain: append a correction request to loop back.
    correction = HumanMessage(
        content=(
            "A reviewer found your previous answer was not fully supported by "
            "the evidence you gathered. Please reconsider: use the tools again "
            "if needed, ground every claim in tool results, and if the evidence "
            "is insufficient, say so honestly."
        )
    )
    return {
        "messages": [correction],
        "reflections": state.get("reflections", 0) + 1,
    }


def _reflection_condition(state: AgentState) -> str:
    """Loop back to the agent for one correction, else finish.

    The reflection node appends a HumanMessage correction request only when it
    wants a retry (and it self-limits to a single pass via the ``reflections``
    counter). So the routing rule is simply: if the last message is that
    correction request, re-run the agent; otherwise we're done.
    """
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return "agent"
    return END


def build_agent(*, with_memory: bool = True):
    """Compile and return the agent graph.

    ``with_memory=True`` attaches the checkpointer so callers can pass a
    ``thread_id`` and get multi-turn memory. Tests can disable it for isolation.
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("reflection", _reflection_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _tools_condition,
        {"tools": "tools", "reflection": "reflection", END: END},
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "reflection", _reflection_condition, {"agent": "agent", END: END}
    )

    if with_memory:
        return graph.compile(checkpointer=_get_checkpointer())
    return graph.compile()


@dataclass
class AgentResult:
    """Final answer plus a transparent trace of how it was produced."""

    answer: str
    tool_calls: list[str] = field(default_factory=list)
    steps: int = 0
    reflections: int = 0


def run_agent(question: str, *, thread_id: str | None = None) -> AgentResult:
    """Run the agent on a question.

    Pass a stable ``thread_id`` across calls to give the agent memory of the
    conversation; omit it for a one-off, stateless query.
    """
    use_memory = thread_id is not None
    agent = build_agent(with_memory=use_memory)

    # With memory, the checkpointer already holds prior turns, so we only send
    # the system prompt on the very first turn of a thread.
    inputs: dict = {"messages": [], "steps": 0, "reflections": 0}
    if use_memory:
        existing = agent.get_state(_thread_config(thread_id)).values
        if not existing.get("messages"):
            inputs["messages"].append(SystemMessage(content=SYSTEM_PROMPT))
    else:
        inputs["messages"].append(SystemMessage(content=SYSTEM_PROMPT))
    inputs["messages"].append(HumanMessage(content=question))

    config = _thread_config(thread_id) if use_memory else {}
    final = agent.invoke(inputs, config=config)

    return _result_from_state(final)


def stream_agent(question: str, *, thread_id: str | None = None):
    """Yield the agent's progress as structured events, then the final result.

    Each yielded item is a ``dict`` with a ``type``:
      - ``{"type": "step", "node": <name>}``            a graph node just ran
      - ``{"type": "tool", "name": <tool>}``            a tool was invoked
      - ``{"type": "reflection", "verdict": <str>}``    grader verdict
      - ``{"type": "final", "result": AgentResult}``    the finished answer

    This powers the SSE endpoint and a live UI trace without waiting for the
    whole run to complete.
    """
    use_memory = thread_id is not None
    agent = build_agent(with_memory=use_memory)

    inputs: dict = {"messages": [], "steps": 0, "reflections": 0}
    if use_memory:
        existing = agent.get_state(_thread_config(thread_id)).values
        if not existing.get("messages"):
            inputs["messages"].append(SystemMessage(content=SYSTEM_PROMPT))
    else:
        inputs["messages"].append(SystemMessage(content=SYSTEM_PROMPT))
    inputs["messages"].append(HumanMessage(content=question))

    config = _thread_config(thread_id) if use_memory else {}
    last_state: dict = {}

    for update in agent.stream(inputs, config=config, stream_mode="updates"):
        for node, node_state in update.items():
            yield {"type": "step", "node": node}
            msgs = node_state.get("messages", []) if node_state else []
            for msg in msgs:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield {"type": "tool", "name": tc["name"]}
            if node_state:
                # Track cumulative state so we can build the final result.
                last_state = _merge_state(last_state, node_state)

    # Rebuild a full-message view from the checkpointer (memory) or last update.
    if use_memory:
        final = agent.get_state(_thread_config(thread_id)).values
    else:
        final = last_state
    result = _result_from_state(final)
    yield {"type": "final", "result": result}


def _merge_state(acc: dict, update: dict) -> dict:
    """Accumulate streamed node updates into a single state view."""
    merged = dict(acc)
    for key, value in update.items():
        if key == "messages" and value:
            merged["messages"] = merged.get("messages", []) + list(value)
        else:
            merged[key] = value
    return merged


def _thread_config(thread_id: str | None) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _result_from_state(final: dict) -> AgentResult:
    messages = final["messages"]

    tool_calls: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls.extend(tc["name"] for tc in msg.tool_calls)

    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            answer = msg.content
            break
    if not answer:
        answer = _fallback_answer(messages)

    return AgentResult(
        answer=str(answer),
        tool_calls=tool_calls,
        steps=final.get("steps", 0),
        reflections=final.get("reflections", 0),
    )


def _fallback_answer(messages: list[AnyMessage]) -> str:
    """If the loop was cut off, summarise from the last tool output."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return (
                "I reached my reasoning-step limit. Based on the last "
                f"information I gathered:\n\n{msg.content}"
            )
    return "I was unable to produce an answer within the step limit."
