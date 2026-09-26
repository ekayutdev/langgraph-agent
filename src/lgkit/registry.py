"""Node registry — maps a ``kind`` to its implementation + default prompt.

The graph builder looks nodes up by kind here. Each entry holds:
- ``fn``: the node function ``(state, params, prompt, ctx, resolved) -> dict``
- ``default_prompt``: system prompt used unless the workflow overrides it
- ``description``: shown in the UI palette
- ``default_params``: sensible param defaults for the node
- ``is_router``: whether this node emits routing signals
- ``uses_node_prompt``: whether the node fn actually reads the ``prompt``
  argument. ``build_graph`` passes ``node.prompt or default_prompt`` to every
  node fn, but almost every kind ignores it — so the editor needs to know which
  kinds the field means anything for, or it shows a text box that does nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeDef:
    kind: str
    fn: Callable[..., dict]
    default_prompt: str
    description: str
    default_params: dict[str, Any] = field(default_factory=dict)
    is_router: bool = False
    uses_node_prompt: bool = False


_REGISTRY: dict[str, NodeDef] = {}


def register(kind: str, def_: NodeDef) -> None:
    _REGISTRY[kind] = def_


def get_node_def(kind: str) -> NodeDef:
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown node kind '{kind}'. Registered: {list(_REGISTRY)}")
    return _REGISTRY[kind]


def all_node_defs() -> dict[str, NodeDef]:
    return dict(_REGISTRY)


def node_palette() -> list[dict[str, Any]]:
    """Compact description for the frontend palette."""
    return [
        {
            "kind": d.kind,
            "description": d.description,
            "default_prompt": d.default_prompt,
            "default_params": d.default_params,
            "is_router": d.is_router,
            "uses_node_prompt": d.uses_node_prompt,
        }
        for d in _REGISTRY.values()
    ]


__all__ = [
    "NodeDef",
    "register",
    "get_node_def",
    "all_node_defs",
    "node_palette",
]
