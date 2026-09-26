"""Event emission + trace nesting for streaming.

Uses langgraph's get_stream_writer() to emit custom events that the
runner forwards as SSE. Trace frames let nested subworkflow/agent-tool
nodes prefix their events with the parent path.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langgraph.config import get_stream_writer

_trace_path: ContextVar[tuple[str, ...]] = ContextVar("trace_path", default=())


def emit_event(
    event_type: str,
    node: str | None = None,
    payload: dict[str, Any] | None = None,
    path: list[str] | None = None,
) -> None:
    """Emit a custom streaming event.

    Events are picked up by the runner's stream_mode="custom" channel and
    forwarded as SSE to the frontend.
    """
    writer = get_stream_writer()
    writer(
        {
            "type": event_type,
            "node": node,
            "payload": payload or {},
            "path": path or list(_trace_path.get()),
        }
    )


@contextmanager
def push_trace_frame(node_id: str) -> Generator[None, None, None]:
    """Push a node id onto the trace path for nested event prefixing."""
    token = _trace_path.set((*_trace_path.get(), node_id))
    try:
        yield
    finally:
        _trace_path.reset(token)


__all__ = ["emit_event", "push_trace_frame", "_trace_path"]
