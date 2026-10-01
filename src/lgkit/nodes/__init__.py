"""Node kinds lgkit ships. Registered lazily by build_graph, never overriding
a kind an app registered first (an embedding app registers its own ``hitl``)."""

from __future__ import annotations

from lgkit.nodes import advisor, approval, hitl, llm
from lgkit.registry import NodeDef, all_node_defs, register


def _builtin_defs() -> list[NodeDef]:
    return [
        NodeDef(
            kind="hitl",
            fn=hitl.run,
            default_prompt="",
            description="Waits for a human decision — choices + optional comment; the choice becomes the routing signal.",
            default_params={"message": "", "choices": ["approve", "reject"], "result_key": ""},
            is_router=True,
        ),
        NodeDef(
            kind="advisor",
            fn=advisor.run,
            default_prompt=advisor.SYSTEM_PROMPT,
            description="LLM recommends one of an approval gate's choices (or escalates).",
            default_params={"max_attempts": 2},
            uses_node_prompt=True,
        ),
        NodeDef(
            kind="approval",
            fn=approval.run,
            default_prompt="",
            description="Approval gate — a human or an agent picks a choice; the choice becomes the routing signal.",
            default_params={"approver": "human", "min_confidence": 0.7},
            is_router=True,
        ),
        NodeDef(
            kind="llm",
            fn=llm.run,
            default_prompt="",
            description="One LLM call — text or structured output into scratch; a structured field can be the routing signal.",
            default_params={"message": "{task}", "max_attempts": 2},
            uses_node_prompt=True,
        ),
    ]


def ensure_registered() -> None:
    existing = all_node_defs()
    for d in _builtin_defs():
        if d.kind not in existing:
            register(d.kind, d)


__all__ = ["ensure_registered"]
