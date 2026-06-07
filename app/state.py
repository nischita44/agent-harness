"""Typed state for the agent graph. State flows through every node."""
from typing import TypedDict, Annotated
from operator import add


class AgentState(TypedDict):
    # The user's original request
    query: str
    # Running message history (LangGraph appends via the `add` reducer)
    messages: Annotated[list, add]
    # Planner's decision: which tool to call, or "final"
    next_action: str
    # Tool output, if a tool ran this turn
    tool_result: str
    # The verified final answer
    answer: str
    # --- observability / cost tracking ---
    input_tokens: int
    output_tokens: int
    cost_usd: float
    step_count: int
    # Verification gate result
    verified: bool
    reject_reason: str
