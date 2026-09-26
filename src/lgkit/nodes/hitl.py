"""HITL node — parks the run with interrupt() until a human answers.

IMPORTANT: code before interrupt() re-executes when the run resumes
(LangGraph replay semantics), so NOTHING is emitted before that line.
After resume the node emits one normal node.start/node.end pair.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.types import interrupt

from lgkit.events import emit_event
from lgkit.nodes._template import flat_state


def run(state, params, prompt, ctx=None, resolved=None) -> dict[str, Any]:
    node_id = params.get("__node_id") or "hitl"
    choices = list(params.get("choices") or [])
    message = str(params.get("message") or "").format_map(flat_state(state))
    started = time.perf_counter()

    answer = interrupt({"node": node_id, "message": message, "choices": choices})

    choice = (answer or {}).get("choice")
    comment = (answer or {}).get("comment", "")
    if choice not in choices:
        raise ValueError(f"hitl '{node_id}': resume choice {choice!r} not in {choices}")

    emit_event(
        "node.start",
        node_id,
        {
            "input": {"message": message},
            "iteration": (state.get("scratch") or {}).get("iteration", 0),
        },
    )
    result = {"choice": choice, "comment": comment}
    key = params.get("result_key") or node_id
    scratch_out = {**state.get("scratch", {}), key: result}
    goto_mode = params.get("route_mode") == "goto"
    route_key = "goto" if goto_mode else "signal"
    emit_event(
        "node.end",
        node_id,
        {
            "result": result,
            "signal": choice if not goto_mode else None,
            "goto": choice if goto_mode else None,
            "duration_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return {
        route_key: choice,
        "scratch": scratch_out,
        "events": [{"node": node_id, "result": result, "tools": []}],
    }


__all__ = ["run"]
