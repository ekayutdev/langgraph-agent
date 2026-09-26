"""Universal agent state schema shared across all nodes.

AgentState carries the task, conversation history, routing signal, and a
domain scratch dict. It stays universal — per-workflow domain state lives
in scratch or in custom state_fields (compiled into a dynamic TypedDict
by graph_builder._build_state_schema).
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State passed through the graph.

    ``messages`` uses add_messages reducer so each node appends rather than
    overwrites the conversation history. ``scratch`` uses merge_scratch so
    parallel branches writing disjoint keys don't clobber each other.
    """

    task: str
    conversation: list[dict[str, Any]]
    messages: Annotated[list[BaseMessage], add_messages]
    signal: str
    phase: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    scratch: Annotated[dict[str, Any], merge_scratch]
    events: Annotated[list[dict[str, Any]], "add_events"]
    error: str


def add_events(left: list, right: list) -> list:
    """Reducer that concatenates event logs."""
    return [*left, *right]


def merge_scratch(left: dict, right: dict) -> dict:
    """Reducer that shallow-merges concurrent scratch writes (right wins).

    Commutative for disjoint keys — the invariant parallel branches rely on.
    """
    return {**(left or {}), **(right or {})}


def union_reducer(left: list, right: list) -> list:
    """Reducer that returns the set-union of two lists (preserving order, no dups)."""
    seen: set = set()
    out: list = []
    for item in (left or []) + (right or []):
        key = (
            json.dumps(item, sort_keys=True, default=str)
            if isinstance(item, (dict, list))
            else item
        )
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def intersection_reducer(left: list, right: list) -> list:
    """Reducer that returns items present in both lists."""
    right_set = set(right or [])
    return [x for x in (left or []) if x in right_set]


def append_unique(left: list, right: list) -> list:
    """Reducer that appends items from right to left, skipping duplicates."""
    out = list(left or [])
    seen = set(out)
    for item in right or []:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def get_reducer(name: str):
    """Return the reducer function by name. Falls back to None (overwrite)."""
    reducers = {
        "default": None,
        "append": add_events,
        "merge": merge_scratch,
        "union": union_reducer,
        "intersection": intersection_reducer,
        "last_value": None,
        "topic": "topic",
    }
    return reducers.get(name)


__all__ = [
    "AgentState",
    "add_messages",
    "add_events",
    "merge_scratch",
    "union_reducer",
    "intersection_reducer",
    "append_unique",
    "get_reducer",
]
