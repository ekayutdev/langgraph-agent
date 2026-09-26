"""Test/demo helpers: a scripted fake LLM and a context manager that hands it
to nodes the same way an app's runner does (RunContext.llm)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lgkit.context import Budget, RunContext, clear_ctx, use_ctx


class ScriptedLLM:
    """Returns scripted replies from ``with_structured_output(schema).invoke()``.

    A dict reply is validated into ``schema`` (so an invalid reply raises the
    same ValidationError a real model's bad output would); an exception reply
    is raised. ``calls`` counts invoke() calls across all structured views.
    """

    def __init__(self, replies: list[Any]):
        self._replies = list(replies)
        self.calls = 0

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> _Structured:
        return _Structured(self, schema)

    def _next(self) -> Any:
        self.calls += 1
        if not self._replies:
            raise RuntimeError("ScriptedLLM: no replies left")
        return self._replies.pop(0)


class _Structured:
    def __init__(self, llm: ScriptedLLM, schema: Any):
        self._llm = llm
        self._schema = schema

    def invoke(self, _messages: Any, **_kwargs: Any) -> Any:
        reply = self._llm._next()
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
