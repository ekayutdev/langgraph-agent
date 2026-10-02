"""Test/demo helpers: a scripted fake LLM and a context manager that hands it
to nodes the same way an app's runner does (RunContext.llm)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from lgkit.context import Budget, RunContext, clear_ctx, use_ctx


class ScriptedLLM:
    """A fake chat model that returns scripted replies, in order.

    ``with_structured_output(schema).invoke()`` validates a dict reply into
    ``schema`` (so an invalid reply raises the same ValidationError a real
    model's bad output would). Plain ``invoke()`` returns an AIMessage whose
    content is the reply (a dict reply is JSON-dumped). An exception reply is
    raised either way. ``calls`` counts every invoke; ``messages`` records the
    messages each call received.
    """

    def __init__(self, replies: list[Any]):
        self._replies = list(replies)
        self.calls = 0
        self.messages: list[Any] = []

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> _Structured:
        return _Structured(self, schema)

    def invoke(self, messages: Any, **_kwargs: Any) -> AIMessage:
        reply = self._next(messages)
        if isinstance(reply, BaseException):
            raise reply
        return AIMessage(content=reply if isinstance(reply, str) else json.dumps(reply))

    def _next(self, messages: Any = None) -> Any:
        self.calls += 1
        self.messages.append(messages)
        if not self._replies:
            raise RuntimeError("ScriptedLLM: no replies left")
        return self._replies.pop(0)


class _Structured:
    def __init__(self, llm: ScriptedLLM, schema: Any):
        self._llm = llm
        self._schema = schema

    def invoke(self, messages: Any, **_kwargs: Any) -> Any:
        reply = self._llm._next(messages)
        if isinstance(reply, BaseException):
            raise reply
        return self._schema.model_validate(reply)


@contextmanager
def using_llm(llm: Any) -> Iterator[None]:
    token = use_ctx(
        RunContext(
            session_id="lgkit-local",
            workspace=Path("."),
            llm=llm,
            budget=Budget(max_iterations=25, recursion_limit=100),
        )
    )
    try:
        yield
    finally:
        clear_ctx(token)


__all__ = ["ScriptedLLM", "using_llm"]
