"""Streamlit Cloud entrypoint for Lumen.

Streamlit Community Cloud runs this file. It ensures the ``src`` layout is
importable, then hands off to the real UI module. On the cloud we default to the
Groq provider (Ollama isn't available there); locally you can still run the app
directly with ``streamlit run src/lumen/ui/app.py`` against Ollama.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the src/ package importable without an editable install.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# On a managed host, default to Groq unless the deployer overrode it. Streamlit
# secrets are exposed as st.secrets; mirror the key into the environment so the
# pydantic settings pick it up. Do this before importing the app.
try:
    import streamlit as st

    if "GROQ_API_KEY" in st.secrets and not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = str(st.secrets["GROQ_API_KEY"])
    # If a key is present and no provider was explicitly set, prefer Groq.
    if os.environ.get("GROQ_API_KEY") and not os.environ.get("LUMEN_LLM_PROVIDER"):
        os.environ["LUMEN_LLM_PROVIDER"] = "groq"
    # On the cloud we default embeddings to FastEmbed (no torch) so the
    # container stays light and boots fast. Override with LUMEN_EMBEDDING_PROVIDER.
    os.environ.setdefault("LUMEN_EMBEDDING_PROVIDER", "fastembed")
except Exception:
    # Running outside Streamlit (e.g. import checks) — no secrets to read.
    pass

# Importing the module runs the Streamlit page top-to-bottom.
import lumen.ui.app  # noqa: E402,F401
