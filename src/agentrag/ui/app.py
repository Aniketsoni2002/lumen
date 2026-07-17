"""Streamlit UI for AgentRAG — chat + a live view of the agent's tool use.

Run with:  streamlit run src/agentrag/ui/app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from agentrag.agent.graph import run_agent
from agentrag.config import get_settings
from agentrag.core.ingest import ingest_file
from agentrag.core.loader import SUPPORTED_SUFFIXES
from agentrag.core.vectorstore import clear_collection
from agentrag.tools import ALL_TOOLS

st.set_page_config(page_title="AgentRAG", page_icon="🤖", layout="centered")
settings = get_settings()

st.title("🤖 AgentRAG")
st.caption(
    "An agent that decides when to search your documents, the web, or compute — "
    "100% local & free."
)

with st.sidebar:
    st.header("Knowledge base")
    uploaded = st.file_uploader(
        "Upload PDF / TXT / Markdown",
        type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Index documents", type="primary"):
        for file in uploaded:
            dest = settings.upload_dir / Path(file.name).name
            dest.write_bytes(file.getbuffer())
            with st.spinner(f"Indexing {file.name}…"):
                n = ingest_file(dest)
            st.success(f"{file.name}: {n} chunks indexed")

    if st.button("🗑️ Clear knowledge base"):
        clear_collection()
        st.info("Knowledge base cleared.")

    st.divider()
    st.caption(f"LLM: `{settings.llm_model}`")
    st.caption("Tools: " + ", ".join(f"`{t.name}`" for t in ALL_TOOLS))

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning…"):
            result = run_agent(prompt)
        st.markdown(result.answer)
        if result.tool_calls:
            with st.expander(
                f"🔎 Agent used {len(result.tool_calls)} tool call(s) "
                f"in {result.steps} step(s)"
            ):
                for i, tc in enumerate(result.tool_calls, start=1):
                    st.write(f"{i}. `{tc}`")
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer}
    )
