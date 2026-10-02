"""loop_limit node — counts passes through a loop and stops it.

Emits "again" while the count is below ``max_rounds`` and "exhausted" once it
is reached (also setting a flag in scratch). It is a pattern's OWN budget;
the builder's run-wide iteration guard stays the outer safety net.
"""

from __future__ import annotations

from typing import Any

AGAIN = "again"
EXHAUSTED = "exhausted"


def run(state, params, prompt, ctx=None, resolved=None) -> dict[str, Any]:
    node_id = params.get("__node_id") or "loop_limit"
    max_rounds = int(params["max_rounds"])
    counter_key = params.get("counter_key") or f"{node_id}__count"
    exhausted_key = params.get("exhausted_key") or f"{node_id}__exhausted"

    count = int((state.get("scratch") or {}).get(counter_key) or 0) + 1
    if count < max_rounds:
        return {"signal": AGAIN, "scratch": {counter_key: count}}
    return {"signal": EXHAUSTED, "scratch": {counter_key: count, exhausted_key: True}}


__all__ = ["AGAIN", "EXHAUSTED", "run"]
