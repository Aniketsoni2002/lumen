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
| Pure vector search | **Hybrid retrieval** — dense vectors + BM25, fused with RRF |
| Answers blindly | **Self-reflection** — grades its own answer, retries if ungrounded |
| Stateless | **Persistent memory** — remembers the conversation across turns |
| Blocking | **Streaming** — live step/tool events over SSE |
| No transparency | Returns a **trace** of every tool the agent used |

The agent follows the **ReAct pattern** (Reason → Act → Observe), expressed as a
**LangGraph state machine** with a hard step-cap so it can never loop forever.

## 🚀 Advanced features

- **🔀 Hybrid retrieval (dense + sparse).** Combines semantic vector search with
  BM25 keyword search and fuses the two rankings using **Reciprocal Rank Fusion**.
  Catches both meaning (*"how do I stop overfitting?"* → regularization) and exact
  terms (error codes, rare names) that pure vector search misses.
- **🪞 Self-reflection / answer grading.** After the agent answers, a grader LLM
  checks whether the answer is actually supported by the evidence it gathered. On
  an `UNGROUNDED` verdict the agent gets **one corrective pass** — a guardrail
  against the classic RAG failure of confident, unsupported answers.
- **🧠 Persistent conversation memory.** A **SQLite-backed LangGraph checkpointer**
  keyed by `session_id` remembers prior turns — so follow-ups like *"and multiply
  that by 3"* work — and it survives **across separate processes** and API
  restarts, not just within one run.
- **📡 Real-time streaming.** A `/ask/stream` **SSE** endpoint (and a `stream_agent`
  generator) emits each reasoning step, tool call, and reflection verdict as it
  happens, instead of blocking until the final answer.

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
| `agent/graph.py` | LangGraph loop: agent ⇄ tools ⇄ reflection, step-capped, memory-backed |
| `agent/reflection.py` | Answer-grading helpers (evidence collection, verdict parsing) |
| `agent/prompts.py` | System prompt + the fact-checking grader prompt |
| `core/hybrid.py` | **Hybrid retriever** — dense + BM25 fused via Reciprocal Rank Fusion |
| `tools/retrieval.py` | `search_documents` — hybrid/dense search over your ChromaDB knowledge base |
| `tools/websearch.py` | `web_search` — free DuckDuckGo search, gracefully degrades on failure |
| `tools/calculator.py` | `calculator` — **AST-sandboxed** arithmetic (no `eval`, no code execution) |
| `core/` | Document loading, chunking, embeddings, vector store, ingestion |
| `api/main.py` | FastAPI REST service (`/ask`, `/ask/stream` SSE, `/ingest`, `/health`) |
| `ui/app.py` | Streamlit chat UI with per-session memory + a live tool/reflection trace |
| `cli.py` | `agentrag ingest / ask [--session] / reset` |

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

**CLI** (index the sample notes, then ask multi-tool + memory-aware questions):
```bash
agentrag ingest data/uploads/sample_ml_notes.md
agentrag ask "What learning rate do the notes recommend, and what is it times 100?"

# Conversation memory — the follow-up remembers the first answer:
agentrag ask "What is the GPU budget per experiment?" --session demo
agentrag ask "Multiply that by 3."                     --session demo
```

**Streaming** (watch the agent reason in real time via SSE):
```bash
curl -N -X POST http://localhost:8000/ask/stream \
     -H "Content-Type: application/json" \
     -d '{"question": "What is 15 times 4?"}'
# → data: {"type":"tool","name":"calculator"} ... data: {"answer":"60", ...}
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
| `AGENTRAG_HYBRID_RETRIEVAL` | `true` | Fuse dense + BM25 retrieval |
| `AGENTRAG_ENABLE_REFLECTION` | `true` | Grade answers and self-correct once |
| `AGENTRAG_MEMORY_DB` | `data/memory.sqlite` | Conversation-memory store |

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

**48 tests**, all fully offline (LLM, web, and vector store are faked). Coverage
includes:
- the sandboxed calculator, including **code-injection attempts** (`__import__`, `open`, …)
- **Reciprocal Rank Fusion** math + hybrid fallback behaviour
- self-reflection grading and `GROUNDED`/`UNGROUNDED` parsing (fail-open on garbage)
- the **agent graph itself**: a scripted fake LLM drives the full
  agent→tools→agent→reflection loop, and dedicated tests prove the **step-cap stops
  an infinite tool loop**, the **reflection retry** fires exactly once, and
  **memory persists across calls**
- the API, including the **SSE streaming** endpoint and `session_id` plumbing

CI runs on Python 3.10 / 3.11 / 3.12.

---

## 🗺️ Roadmap
- [x] Hybrid retrieval (dense + BM25, RRF fusion)
- [x] Self-reflection node that grades its own answer before returning
- [x] Persistent conversation memory (SQLite checkpointer)
- [x] Streaming intermediate steps over SSE
- [ ] Token-level streaming of the final answer to the UI
- [ ] Cross-encoder re-ranking on top of hybrid retrieval
- [ ] More tools: SQL query, Python REPL, arXiv search
- [ ] RAGAS evaluation harness for retrieval + answer quality

---

## 🛠️ Tech stack
**LangGraph** (agent + SQLite checkpointer) · **LangChain** · **Ollama** ·
**ChromaDB** · **HuggingFace embeddings** · **BM25 / Reciprocal Rank Fusion** ·
**DuckDuckGo Search** · **FastAPI** (REST + SSE) · **Streamlit** · **pytest** ·
**ruff** · **Docker**

---

## 📄 License
MIT © Aniketsoni2002
