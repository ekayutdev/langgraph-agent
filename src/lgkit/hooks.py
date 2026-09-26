"""Hooks the graph builder calls instead of importing an app's stores.

lgkit knows nothing about where agents/workflows are stored or how event
scripts run. An app passes its own resolvers; the defaults resolve
nothing and dispatch nothing, which is what a standalone workflow needs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def _resolve_nothing(_ref: Any) -> None:
    return None


@dataclass(frozen=True)
class BuildHooks:
    agent_resolver: Callable[[Any], Any] = _resolve_nothing
    """agent id -> AgentSpec | None (must not raise for unknown ids)."""

    workflow_resolver: Callable[[str], Any] = _resolve_nothing
    """workflow ref -> WorkflowSpec | None."""

    dispatch_event: Callable[..., dict[str, Any]] | None = None
    """dispatch_event(event_type, state, *, scope, origin, node_id, spec, trace)
    -> {"state_delta": dict, "route": str | None}. None = event scripts do not run."""


DEFAULT_HOOKS = BuildHooks()

__all__ = ["BuildHooks", "DEFAULT_HOOKS"]
