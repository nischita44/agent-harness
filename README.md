# agent-harness

A minimal, observable, cost-tracked agentic system. LangGraph + FastAPI,
deployable to both a self-managed Linux VM and Cloud Run.

This is a portfolio artifact, not a toy. It demonstrates the three things
applied-AI interviews actually probe:

1. **Structured control flow** — a `plan → act → verify` `StateGraph`, not a
   single prompt.
2. **Tool grounding** — the agent acts on real tool output (calculator, web fetch).
3. **A verification gate** — a node that rejects answers not supported by tool
   results. This is where backend rigor (idempotency, verification, failure
   handling) becomes an AI differentiator.

Every model call accumulates token usage and USD cost in graph state, returned
on every response.

## Architecture

```
        ┌──────┐   tool needed   ┌─────┐
  query │ plan │ ──────────────▶ │ act │
   ───▶ │      │ ◀────────────── │     │
        └──┬───┘   tool_result   └─────┘
           │ final
           ▼
        ┌────────┐
        │ verify │ ──▶ END  (answer + verified flag + cost)
        └────────┘
```

State (`app/state.py`) carries query, messages, tool results, the answer, the
verification verdict, and running cost. A step budget guards against loops.

## Run locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
uvicorn app.main:app --reload --port 8000

curl -s localhost:8000/invoke -H 'content-type: application/json' \
  -d '{"query":"What is 47*89 plus 13?"}' | python -m json.tool
```

## Observability (Langfuse)

```bash
export LANGFUSE_PUBLIC_KEY=pk-...
export LANGFUSE_SECRET_KEY=sk-...
export LANGFUSE_HOST=https://cloud.langfuse.com
```

With keys set, wrap `GRAPH.invoke` with the Langfuse callback handler (see
their LangGraph guide — drop the `CallbackHandler` into the `config` arg).
`/health` reports whether tracing is active.

## Deploy: VM

```bash
# on the VM
git clone <your-repo> && cd agent-harness
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
sudo cp deploy/agent-harness.service /etc/systemd/system/
# edit the unit: set ANTHROPIC_API_KEY
sudo systemctl daemon-reload && sudo systemctl enable --now agent-harness
sudo cp deploy/nginx.conf /etc/nginx/sites-available/agent-harness
sudo ln -s /etc/nginx/sites-available/agent-harness /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

## Deploy: Cloud Run

```bash
gcloud run deploy agent-harness --source . \
  --region us-central1 --allow-unauthenticated \
  --set-env-vars ANTHROPIC_API_KEY=sk-...
```
