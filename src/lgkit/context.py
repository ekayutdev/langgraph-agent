"""Per-run execution context, propagated via a ContextVar.

A run's live services (workspace, llm, budget) flow at run time through
``_CURRENT``, so asyncio gives each concurrent run its own copy.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lgkit.spec import AgentSpec, WorkflowSpec


@dataclass
class Budget:
    max_iterations: int
    recursion_limit: int


@dataclass
class RunContext:
    """Live, per-run services. session_id == run_id == thread_id."""

    session_id: str
    workspace: Path
    llm: Any
    budget: Budget
    dry_run: Any = None


_CURRENT: ContextVar[RunContext | None] = ContextVar("run_context", default=None)


def current_ctx() -> RunContext | None:
    return _CURRENT.get()


def use_ctx(ctx: RunContext) -> Token:
    return _CURRENT.set(ctx)


def clear_ctx(token: Token) -> None:
    _CURRENT.reset(token)


def get_resolved_workflow(name: str, resolved=None) -> WorkflowSpec | None:
    """Return the snapshot workflow spec for ``name`` from the resolved unit."""
    if resolved is not None:
        return resolved.workflows.get(name)
    return None


def get_resolved_agent(name: str, resolved=None) -> AgentSpec | None:
    """Return the snapshotted agent spec for ``name`` from the resolved unit."""
    if resolved is not None:
        return resolved.agents.get(name)
    return None


__all__ = [
    "Budget",
    "RunContext",
    "current_ctx",
    "use_ctx",
    "clear_ctx",
    "get_resolved_workflow",
    "get_resolved_agent",
]
