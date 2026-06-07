"""The agent graph: plan -> act -> verify.

This is the core artifact. It demonstrates the three things interviewers
for applied-AI roles actually probe:
  1. Structured multi-step control flow (LangGraph StateGraph)
  2. Tool grounding (the agent acts on real tool output)
  3. A verification gate (rejects answers not grounded in tool results)

Cost is tracked on every model call and accumulated in state.
"""
import json
import os

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langfuse.callback import CallbackHandler

from .state import AgentState
from .tools import TOOLS, TOOL_SPECS

MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-4-5")

# Rough per-million-token prices; update from current docs before relying on these.
PRICE_IN = float(os.environ.get("PRICE_IN_PER_MTOK", "3.0")) / 1_000_000
PRICE_OUT = float(os.environ.get("PRICE_OUT_PER_MTOK", "15.0")) / 1_000_000

llm = ChatAnthropic(model=MODEL, max_tokens=1024, temperature=0)


def _track(state: AgentState, resp) -> None:
    """Accumulate token usage and cost from a model response."""
    usage = getattr(resp, "usage_metadata", None) or {}
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    state["input_tokens"] += in_tok
    state["output_tokens"] += out_tok
    state["cost_usd"] += in_tok * PRICE_IN + out_tok * PRICE_OUT
    state["step_count"] += 1


# ---- Node 1: planner ----
def plan(state: AgentState) -> AgentState:
    prompt = (
        "You are a planner. Decide the next step to answer the user's query.\n"
        f"Available tools:\n{TOOL_SPECS}\n\n"
        f"Query: {state['query']}\n"
        f"Tool result so far: {state.get('tool_result') or 'none'}\n\n"
        "Respond ONLY with JSON: "
        '{"action": "<tool_name|final>", "arg": "<tool arg or final answer>"}'
    )
    resp = llm.invoke(prompt)
    _track(state, resp)
    try:
        decision = json.loads(resp.content.strip().strip("`").replace("json", "", 1))
    except Exception:  # noqa: BLE001
        decision = {"action": "final", "arg": resp.content}
    state["next_action"] = decision.get("action", "final")
    if state["next_action"] == "final":
        state["answer"] = decision.get("arg", "")
    else:
        state["messages"] = [{"tool": state["next_action"], "arg": decision.get("arg", "")}]
    return state


# ---- Node 2: act (tool execution) ----
def act(state: AgentState) -> AgentState:
    call = state["messages"][-1]
    tool_fn = TOOLS.get(call["tool"])
    state["tool_result"] = tool_fn(call["arg"]) if tool_fn else f"ERROR: unknown tool {call['tool']}"
    return state


# ---- Node 3: verify (the grounding gate) ----
def verify(state: AgentState) -> AgentState:
    prompt = (
        "You are a strict verifier. Is the ANSWER fully supported by the "
        "TOOL RESULT and the query? Reject if it invents facts.\n"
        f"Query: {state['query']}\n"
        f"Tool result: {state.get('tool_result') or 'none'}\n"
        f"Answer: {state['answer']}\n\n"
        'Respond ONLY with JSON: {"verified": true|false, "reason": "<short>"}'
    )
    resp = llm.invoke(prompt)
    _track(state, resp)
    try:
        v = json.loads(resp.content.strip().strip("`").replace("json", "", 1))
    except Exception:  # noqa: BLE001
        v = {"verified": False, "reason": "verifier parse failure"}
    state["verified"] = bool(v.get("verified"))
    state["reject_reason"] = "" if state["verified"] else v.get("reason", "unspecified")
    return state


# ---- Routing ----
def route_after_plan(state: AgentState) -> str:
    if state["next_action"] == "final":
        return "verify"
    if state["step_count"] > 6:  # hard loop guard
        state["answer"] = "Step budget exceeded."
        return "verify"
    return "act"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("verify", verify)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route_after_plan, {"act": "act", "verify": "verify"})
    g.add_edge("act", "plan")
    g.add_edge("verify", END)
    return g.compile()


GRAPH = build_graph()


def run(query: str) -> dict:
    init: AgentState = {
        "query": query, "messages": [], "next_action": "", "tool_result": "",
        "answer": "", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
        "step_count": 0, "verified": False, "reject_reason": "",
    }
    langfuse_handler = CallbackHandler()
    final = GRAPH.invoke(init, config={"callbacks": [langfuse_handler]})
    langfuse_handler.flush()
    return {
        "answer": final["answer"],
        "verified": final["verified"],
        "reject_reason": final["reject_reason"],
        "cost_usd": round(final["cost_usd"], 6),
        "tokens": {"in": final["input_tokens"], "out": final["output_tokens"]},
        "steps": final["step_count"],
    }
