"""Approval gate — a human or an agent picks one of ``choices``.

approver="agent": the advisor's answer is used when it is valid, not
"escalate" and confidence >= min_confidence; otherwise a human is asked.
approver="human": a human is always asked; advice (if an advisor ran) is
shown alongside.

An invalid resume answer (a choice not in ``choices``) re-prompts: the
gate calls interrupt() again with the same payload plus an ``error`` key,
so one bad answer never bricks the thread.

Nothing with side effects runs before interrupt(): on resume LangGraph
re-executes this node from the top, so only pure reads happen before it.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.types import interrupt

from lgkit.events import emit_event
from lgkit.nodes._template import flat_state


def _usable(advice: Any) -> dict[str, Any] | None:
    if not isinstance(advice, dict):
        return None
    return advice if advice and "error" not in advice else None


def _confidence(advice: dict[str, Any]) -> float:
    """Best-effort read: hand-written advice may carry a non-numeric
    confidence (e.g. "high"); that counts as unusable advice (asks a
    human) instead of crashing float()."""
    try:
        return float(advice.get("confidence", 0.0))
    except (TypeError, ValueError):
        return -1.0


def _agent_can_decide(advice: dict[str, Any] | None, choices: list[str], min_confidence: float) -> bool:
    return (
        advice is not None
        and advice.get("choice") in choices
        and _confidence(advice) >= min_confidence
    )


def _parse_answer(answer: Any) -> tuple[Any, str]:
    if isinstance(answer, str):
        return answer, ""
    if not isinstance(answer, dict):
        return None, ""
    return answer.get("choice"), str(answer.get("comment") or "")


def run(state, params, prompt, ctx=None, resolved=None) -> dict[str, Any]:
    node_id = params.get("__node_id") or "approval"
    choices = list(params.get("choices") or [])
    approver = params.get("approver", "human")
    min_confidence = float(params.get("min_confidence", 0.7))
    advice_key = params.get("advice_key")
    raw_advice = (state.get("scratch") or {}).get(advice_key) if advice_key else None
    if not isinstance(raw_advice, dict):
        raw_advice = None
    advice = _usable(raw_advice)
    message = str(params.get("message") or "").format_map(flat_state(state))
    started = time.perf_counter()

    if approver == "agent" and _agent_can_decide(advice, choices, min_confidence):
        choice, comment, by = advice["choice"], str(advice.get("reason") or ""), "agent"
    else:
        error: str | None = None
        while True:
            payload = {
                "node": node_id,
                "message": message,
                "choices": choices,
                "advice": advice,
                "advice_error": (raw_advice or {}).get("error"),
            }
            if error is not None:
                payload["error"] = error
            answer = interrupt(payload)
            choice, comment = _parse_answer(answer)
            if choice in choices:
                break
            error = f"choice {choice!r} not in {choices} (answer was {answer!r})"
        by = "human"

    result = {"choice": choice, "comment": comment, "by": by, "advice": advice}
    emit_event("node.start", node_id, {"input": {"message": message}})
    emit_event(
        "node.end",
        node_id,
        {
            "result": result,
            "signal": choice,
            "duration_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return {
        "signal": choice,
        "scratch": {params.get("result_key") or node_id: result},
        "events": [{"node": node_id, "result": result, "tools": []}],
    }


__all__ = ["run"]
