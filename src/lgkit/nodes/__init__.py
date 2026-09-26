"""Node kinds lgkit ships. Registered lazily by build_graph, never overriding
a kind an app registered first (an embedding app registers its own ``hitl``)."""

from __future__ import annotations

from lgkit.nodes import hitl
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
    ]


def ensure_registered() -> None:
    existing = all_node_defs()
    for d in _builtin_defs():
        if d.kind not in existing:
            register(d.kind, d)


__all__ = ["ensure_registered"]
