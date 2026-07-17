# Deploying Lumen to Streamlit Community Cloud

Lumen runs locally on **Ollama** (free, private) and in the cloud on **Groq**
(free hosted API). Streamlit Cloud can't run Ollama, so cloud deployments use
Groq — the app is provider-agnostic and switches automatically when a
`GROQ_API_KEY` is present.

## 1. Get a free Groq API key

1. Go to <https://console.groq.com> and sign in (free).
2. **API Keys → Create API Key**, copy it (starts with `gsk_...`).

Groq's free tier is generous and fast — perfect for a public demo.

## 2. Push the repo to GitHub

Already done if you're reading this in the repo. The deploy entrypoint is
[`streamlit_app.py`](streamlit_app.py) at the project root.

## 3. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub.**
3. Fill in:
   - **Repository:** `Aniketsoni2002/lumen`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Click **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_real_key_here"
   ```
5. Click **Deploy**. First build takes a few minutes (it installs
   `requirements.txt`, including the embedding model on first run).

Your app goes live at `https://<something>.streamlit.app` — a public link you
can put on your resume/LinkedIn.

## 4. Try it

Upload a PDF in the sidebar → **Index** → ask questions. The sidebar shows the
active model (e.g. `groq:qwen/qwen3.6-27b`).

## How it stays light on the free tier

The cloud build is **torch-free**. `requirements.txt` installs FastEmbed (an
ONNX embedding runtime) instead of `sentence-transformers`/PyTorch (~500 MB), and
`streamlit_app.py` defaults the app to **Groq** (LLM) + **FastEmbed** (embeddings)
when a `GROQ_API_KEY` secret is present. That keeps the container well under the
free-tier memory ceiling and makes builds fast.

## Notes & limits (free tier)

- **Storage is ephemeral.** The vector store lives on the container's disk and
  resets when the app sleeps/redeploys. Re-upload documents after a restart.
  For persistence, point ChromaDB at a hosted vector DB (out of scope here).
- **Keep uploads small.** `maxUploadSize = 10` MB is set in
  `.streamlit/config.toml`.
- **Never commit your key.** Put it in Streamlit's Secrets box only.
  `.streamlit/secrets.toml` is gitignored.

## Running locally (Ollama, unchanged)

```bash
ollama pull qwen2.5:7b
pip install -e ".[local,dev]"         # 'local' adds HuggingFace embeddings
streamlit run src/lumen/ui/app.py     # uses Ollama + HuggingFace by default
```

To test the Groq path locally:

```bash
export GROQ_API_KEY=gsk_...            # provider auto-switches to groq
export LUMEN_LLM_PROVIDER=groq
lumen ask "What is 15 times 4?"
```
