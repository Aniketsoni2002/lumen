"""The agentic reasoning loop, built on LangGraph.

The graph has two nodes and a conditional edge:

    ┌─────────┐   tool calls?   ┌──────────┐
    │  agent  │ ───────yes────▶ │  tools   │
    │ (LLM)   │ ◀───────────────│ (execute)│
    └────┬────┘                 └──────────┘
         │ no tool calls
         ▼
        END  → final answer

The LLM node decides on each turn whether it needs a tool or can answer. A
step counter in the state caps the number of tool-calling turns so a confused
model can never loop forever. This is the classic ReAct pattern expressed as a
state machine — the modern, production way to build agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agentrag.agent.prompts import SYSTEM_PROMPT
from agentrag.config import get_settings
from agentrag.tools import ALL_TOOLS
from agentrag.utils.logging import get_logger

logger = get_logger("agent")


class AgentState(dict):
    """Graph state. ``messages`` accumulates; ``steps`` bounds the loop."""

    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


def _build_llm():
    settings = get_settings()
    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
    )
    # Bind tools so the model can emit structured tool calls.
    return llm.bind_tools(ALL_TOOLS)


def _agent_node(state: AgentState) -> dict:
    """LLM turn: look at the conversation so far and decide what to do next."""
    llm = _build_llm()
    response = llm.invoke(state["messages"])
    return {"messages": [response], "steps": state.get("steps", 0) + 1}


def _should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Route to tools if the model asked for one and we're under the cap."""
    settings = get_settings()
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)

    if tool_calls and state.get("steps", 0) < settings.max_agent_steps:
        names = ", ".join(tc["name"] for tc in tool_calls)
        logger.info("Agent step %d -> tools: %s", state.get("steps", 0), names)
        return "tools"
    return END


def build_agent():
    """Compile and return the agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


@dataclass
class AgentResult:
    """The final answer plus the tool trace, for transparency in the UI."""

    answer: str
    tool_calls: list[str] = field(default_factory=list)
    steps: int = 0


def run_agent(question: str) -> AgentResult:
    """Run the agent to completion on a single question."""
    agent = build_agent()
    initial = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
        "steps": 0,
    }
    final = agent.invoke(initial)

    # Collect which tools were used, for a transparent "reasoning trace".
    tool_calls: list[str] = []
    for msg in final["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls.extend(tc["name"] for tc in msg.tool_calls)

    # The answer is the content of the last AI message with no tool calls.
    answer = ""
    for msg in reversed(final["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            answer = msg.content
            break
    if not answer:
        # Rare: hit the step cap mid-tool-call. Surface whatever we have.
        answer = _fallback_answer(final["messages"])

    return AgentResult(
        answer=answer, tool_calls=tool_calls, steps=final.get("steps", 0)
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
