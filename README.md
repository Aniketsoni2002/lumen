<div align="center">

# 🤖 AgentRAG

### An agentic RAG system — the LLM decides *when* to retrieve, search the web, or compute.

Unlike a classic RAG pipeline that blindly retrieves on every query, **AgentRAG**
gives an LLM a toolbelt and lets it *reason* about which tool to use: your private
documents, a live web search, or an exact calculator — looping through multiple
steps until it can answer. Built on **LangGraph**, running **100% locally and free**
(Ollama + HuggingFace + ChromaDB + DuckDuckGo). No API key.

[![CI](https://github.com/Aniketsoni2002/agentrag/actions/workflows/ci.yml/badge.svg)](https://github.com/Aniketsoni2002/agentrag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Built with LangGraph](https://img.shields.io/badge/built%20with-LangGraph-orange)

</div>

---

## ✨ What makes this different from "normal" RAG

| Classic RAG | **AgentRAG** |
|---|---|
| Always retrieves, then answers | Agent **decides** whether to retrieve at all |
| One retrieval, one shot | **Multi-step** loop — can chain tools, refine, retry |
| Documents only | **Documents + web search + calculator**, agent-routed |
| No transparency | Returns a **trace** of every tool the agent used |

The agent follows the **ReAct pattern** (Reason → Act → Observe), expressed as a
**LangGraph state machine** with a hard step-cap so it can never loop forever.

---

## 🧠 How the agent thinks

```
        ┌────────────────────────────────────────────────┐
        │                                                │
        ▼                                                │
   ┌─────────┐   needs a tool?    ┌──────────────────┐   │ observe result,
   │  agent  │ ───────yes───────▶ │      tools       │   │ think again
   │ (LLM +  │                    │ • search_documents│──┘
   │  tools) │ ◀──────────────────│ • web_search      │
   └────┬────┘                    │ • calculator      │
        │ no — I can answer       └──────────────────┘
        ▼
       END  →  final answer  +  tool-use trace
```

**Example** — *"What learning rate do our notes recommend, and what is that times 100?"*
1. Agent calls `search_documents` → finds "recommended default learning rate is 0.001"
2. Agent calls `calculator` with `0.001 * 100` → `0.1`
3. Agent answers: *"Your notes recommend 0.001; ×100 = 0.1"* — citing `sample_ml_notes.md`

Two different tools, chosen and sequenced by the model itself.

---

## 🏗️ Architecture

| Module | Responsibility |
|---|---|
| `agent/graph.py` | The LangGraph ReAct loop: agent node ⇄ tool node, step-capped |
| `agent/prompts.py` | System prompt that teaches the agent when to use each tool |
| `tools/retrieval.py` | `search_documents` — semantic search over your ChromaDB knowledge base |
| `tools/websearch.py` | `web_search` — free DuckDuckGo search, gracefully degrades on failure |
| `tools/calculator.py` | `calculator` — **AST-sandboxed** arithmetic (no `eval`, no code execution) |
| `core/` | Document loading, chunking, embeddings, vector store, ingestion |
| `api/main.py` | FastAPI REST service (`/ask`, `/ingest`, `/health`) |
| `ui/app.py` | Streamlit chat UI with a live tool-use trace |
| `cli.py` | `agentrag ingest / ask / reset` |

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) with a tool-capable model pulled:
```bash
ollama pull qwen2.5:3b
```

### 2. Install
```bash
git clone https://github.com/Aniketsoni2002/agentrag.git
cd agentrag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Try it

**CLI** (index the included sample notes, then ask a multi-tool question):
```bash
agentrag ingest data/uploads/sample_ml_notes.md
agentrag ask "What learning rate do the notes recommend, and what is it times 100?"
```

**Streamlit chat UI** (shows the agent's tool trace live):
```bash
streamlit run src/agentrag/ui/app.py
```

**REST API** (interactive docs at http://localhost:8000/docs):
```bash
uvicorn agentrag.api.main:app --reload
```

---

## 🐳 Run with Docker
```bash
docker compose up --build
docker compose exec ollama ollama pull qwen2.5:3b   # first time only
```

---

## ⚙️ Configuration

Override any setting via env vars or a `.env` file (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|---|---|---|
| `AGENTRAG_LLM_MODEL` | `qwen2.5:3b` | Ollama model (must support tool calling) |
| `AGENTRAG_MAX_AGENT_STEPS` | `6` | Hard cap on reasoning/tool turns |
| `AGENTRAG_WEB_RESULTS` | `4` | Web results returned per search |
| `AGENTRAG_TOP_K` | `4` | Chunks retrieved from the knowledge base |

> **A note on model choice:** agentic tool-routing needs a model that's good at
> function calling. The default `qwen2.5:3b` handles multi-step tool chains
> reliably even at 3B params. Smaller/weaker models (e.g. `llama3.2:3b`) work for
> single-tool questions but are less reliable at chaining tools — swap in a larger
> model via `AGENTRAG_LLM_MODEL` for the most robust behaviour.

---

## 🧪 Testing & quality

```bash
pytest              # runs fully offline — LLM, web, and vector store are faked
ruff check src tests
```

The suite covers the sandboxed calculator (including **code-injection attempts**),
document loading, both tools (with network mocked), and — notably — the **agent
graph itself**: a scripted fake LLM drives the full agent→tool→agent→answer loop
and verifies the step-cap actually stops an infinite tool loop. CI runs on Python
3.10 / 3.11 / 3.12.

---

## 🗺️ Roadmap
- [ ] Streaming intermediate steps to the UI in real time
- [ ] Conversation memory across turns (LangGraph checkpointer)
- [ ] Self-reflection node that grades its own answer before returning
- [ ] More tools: SQL query, Python REPL, arXiv search

---

## 🛠️ Tech stack
**LangGraph** · **LangChain** · **Ollama** · **ChromaDB** · **HuggingFace** ·
**DuckDuckGo Search** · **FastAPI** · **Streamlit** · **pytest** · **ruff** · **Docker**

---

## 📄 License
MIT © Aniketsoni2002
