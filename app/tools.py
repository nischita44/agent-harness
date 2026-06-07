"""Tools the agent can call. Start with one real tool; add more later.

The verification-loop philosophy: tools return GROUNDED facts. The verifier
later checks that the final answer is supported by tool output, not invented.
"""
import json
import urllib.request


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Safe subset only."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "ERROR: expression contains disallowed characters"
    try:
        # eval restricted to arithmetic — no builtins, no names
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def web_fetch(url: str) -> str:
    """Fetch a URL and return the first 2000 chars of text. Grounding source."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "agent-harness/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read(4000).decode("utf-8", errors="ignore")[:2000]
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


# Registry the planner picks from
TOOLS = {
    "calculator": calculator,
    "web_fetch": web_fetch,
}

TOOL_SPECS = json.dumps(
    {
        "calculator": "Evaluate arithmetic. Arg: a math expression string.",
        "web_fetch": "Fetch text from a URL. Arg: a full https URL.",
    },
    indent=2,
)
