"""FastAPI wrapper. Exposes /invoke and /health.

Langfuse tracing: set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
in the environment. If unset, the app runs fine without tracing.
"""
import os

from fastapi import FastAPI
from pydantic import BaseModel

from .graph import run

app = FastAPI(title="agent-harness", version="0.1")


class Query(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok", "tracing": bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))}


@app.post("/invoke")
def invoke(q: Query):
    return run(q.query)
