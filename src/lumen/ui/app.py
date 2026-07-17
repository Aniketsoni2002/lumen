"""Streamlit UI for Lumen — chat + a live view of the agent's tool use.

Run with:  streamlit run src/lumen/ui/app.py
"""
from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from lumen.agent.graph import run_agent
from lumen.agent.llm import active_model_name
from lumen.config import get_settings
from lumen.core.ingest import ingest_file
from lumen.core.loader import SUPPORTED_SUFFIXES
from lumen.core.vectorstore import clear_collection
from lumen.tools import ALL_TOOLS

st.set_page_config(page_title="Lumen", page_icon="🤖", layout="centered")
settings = get_settings()

st.title("🤖 Lumen")
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

    if st.button("💬 New conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"LLM: `{active_model_name()}`")
    st.caption("Tools: " + ", ".join(f"`{t.name}`" for t in ALL_TOOLS))

# A stable per-browser-session id gives the agent conversation memory, so
# follow-up questions ("and what about the second one?") work.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
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
            result = run_agent(prompt, thread_id=st.session_state.session_id)
        st.markdown(result.answer)
        if result.tool_calls or result.reflections:
            label = (
                f"🔎 {len(result.tool_calls)} tool call(s) in "
                f"{result.steps} step(s)"
            )
            if result.reflections:
                label += f" · {result.reflections} self-correction(s)"
            with st.expander(label):
                for i, tc in enumerate(result.tool_calls, start=1):
                    st.write(f"{i}. `{tc}`")
                if result.reflections:
                    st.write(
                        "♻️ The agent re-checked its answer against the "
                        "evidence and revised it."
                    )
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer}
    )
