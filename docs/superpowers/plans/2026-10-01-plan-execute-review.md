# `llm` / `loop_limit` nodes + `plan_execute_review` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic single-call `llm` node, a `loop_limit` counter node and the `plan_execute_review()` pattern to lgkit, and make lgtools' editor support the two new kinds.

**Architecture:** `llm` and `loop_limit` are ordinary registry kinds with the standard node signature; the pattern returns spec data (a `Fragment`) wiring plan → execute → review → limit with two named exits. LLM selection is shared with `advisor` through one helper so dry-run always wins. The builder ends the run (instead of taking the first edge) when a signal from an `approval`, `llm` or `loop_limit` source matches no condition.

**Tech Stack:** Python ≥3.12, LangGraph 1.2.x, langchain-core, Pydantic 2, pytest, ruff; lgtools frontend: SvelteKit + vitest.

**Spec:** `docs/superpowers/specs/2026-10-01-plan-execute-review-design.md`

**Paths:**
- `$LGA` = `/Users/ekayut/Project/product/langgraph-agent` (package `lgkit`; `uv run pytest -q`, `uv run ruff check src tests examples`)
- `$LGT` = `/Users/ekayut/Project/product/laggraph-hermes` (lgtools; `backend/`: `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`; `frontend/`: `npm run check`, `npm test -- --reporter=dot`)

## Global Constraints

- Exhausting `max_rounds` is never an approval: the pattern leaves through the `exhausted` exit and sets `scratch["<id>__exhausted"] = True`.
- A signal that matches none of a source's conditions must END the run for sources of kind `approval`, `llm`, `loop_limit`; every other kind keeps today's behaviour exactly.
- Dry-run never builds a real model: `ctx.dry_run` wins over a node's `llm` param and `ctx.llm`; a dry-run bound in the ambient context that did not reach the node raises `RuntimeError("dry-run context lost: refusing to build a real model")`.
- `lgkit` never imports `lgtools`; `grep -rn lgtools $LGA/src` prints nothing. No module-level mutable config.
- `advisor` behaviour is unchanged by the refactor (its existing tests pass untouched).
- `approval_gate` keeps working unchanged (`Fragment.exit` / `Fragment.choices` stay).
- lgtools suites must not regress: backend currently 1140 passed / 4 skipped; frontend 33 files / 374 tests; `npm run check` 0 errors 0 warnings.
- Work on branch `feat/plan-execute-review` in `$LGA` and `feat/llm-node-editor` in `$LGT`. Do not push.
- Commits end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb
  ```

## Review Focus

1. **Template references a nested key that does not exist yet** — the executor's message uses `{scratch.<id>__review.feedback}` on round one, when there is no review. Expectation: renders empty, no crash. Pinned in Task 1 (`flat_state` chained missing keys).
2. **Reviewer returns a verdict outside approve/revise** — expectation: counted as a failed attempt, retried, then the run fails loudly; it never routes. Pinned in Task 2.
3. **Global iteration budget runs out mid-loop** — expectation: the run ends; it does not loop forever through `again`. Pinned in Task 3.
4. **Two patterns in one workflow** — expectation: independent round counters. Pinned in Task 4.
5. **Per-node `llm` param during a dry-run** — expectation: the scripted model is used, no real model built. Pinned in Task 2.

---

### Task 1: Shared LLM resolver, plain-text `ScriptedLLM`, tolerant templates

**Files:**
- Create: `$LGA/src/lgkit/nodes/_llm.py`
- Modify: `$LGA/src/lgkit/nodes/advisor.py` (`_llm_for` delegates), `$LGA/src/lgkit/testing.py`, `$LGA/src/lgkit/nodes/_template.py`
- Test: `$LGA/tests/test_resolve_llm.py` (new), `$LGA/tests/test_testing.py`, `$LGA/tests/test_hitl.py`

**Interfaces:**
- Produces:
  - `lgkit.nodes._llm.resolve_llm(params: dict, ctx: Any, node_id: str) -> Any` — precedence: `ctx.dry_run.model_for(node_id)` → refuse if `current_ctx()` has a dry-run → `build_llm(**params["llm"])` → `ctx.llm` → `build_llm()`.
  - `ScriptedLLM.invoke(messages) -> AIMessage` (plain text: a `str` reply becomes the content, a `dict` reply is JSON-dumped, an exception reply is raised); `ScriptedLLM.messages: list` records the messages of every call (plain and structured), in order.
  - `flat_state`: a missing key renders `""` and tolerates further attribute access (`{scratch.missing.deeper}` → `""`).

- [ ] **Step 1: Write the failing tests**

`tests/test_resolve_llm.py`:

```python
from pathlib import Path

import pytest

import lgkit.llm
from lgkit.context import Budget, RunContext, clear_ctx, use_ctx
from lgkit.nodes._llm import resolve_llm


class _DryRun:
    def __init__(self):
        self.asked = []

    def model_for(self, node_id, spec=None):
        self.asked.append(node_id)
        return "FAKE"


def _ctx(llm=None, dry_run=None):
    return RunContext(
        session_id="t", workspace=Path("."), llm=llm, budget=Budget(10, 50), dry_run=dry_run
    )


@pytest.fixture
def no_real_llm(monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(k or a)
        raise AssertionError("real LLM built")

    monkeypatch.setattr(lgkit.llm, "build_llm", boom)
    return calls


def test_dry_run_wins_over_node_llm_and_ctx_llm(no_real_llm):
    dry = _DryRun()
    got = resolve_llm({"llm": {"provider": "openai"}}, _ctx(llm=object(), dry_run=dry), "n1")
    assert got == "FAKE" and dry.asked == ["n1"] and not no_real_llm


def test_lost_dry_run_context_refuses_before_node_llm(no_real_llm):
    token = use_ctx(_ctx(dry_run=_DryRun()))
    try:
        with pytest.raises(RuntimeError, match="dry-run context lost"):
            resolve_llm({"llm": {"provider": "openai"}}, None, "n1")
    finally:
        clear_ctx(token)
    assert not no_real_llm


def test_node_llm_then_ctx_llm_then_default(monkeypatch):
    built = []
    monkeypatch.setattr(lgkit.llm, "build_llm", lambda *a, **k: built.append(k) or "BUILT")
    assert resolve_llm({"llm": {"provider": "openai", "model": "m"}}, _ctx(llm="CTX"), "n") == "BUILT"
    assert built == [{"provider": "openai", "model": "m"}]
    assert resolve_llm({}, _ctx(llm="CTX"), "n") == "CTX"
    assert resolve_llm({}, None, "n") == "BUILT"
```

Append to `tests/test_testing.py`:

```python
def test_scripted_llm_plain_invoke_returns_ai_message_and_records_messages():
    llm = ScriptedLLM(["hello", {"a": 1}, RuntimeError("boom")])
    first = llm.invoke(["m1"])
    assert first.content == "hello"
    assert llm.invoke(["m2"]).content == '{"a": 1}'
    with pytest.raises(RuntimeError, match="boom"):
        llm.invoke(["m3"])
    assert llm.calls == 3
    assert llm.messages == [["m1"], ["m2"], ["m3"]]


def test_scripted_llm_structured_calls_are_recorded_too():
    llm = ScriptedLLM([{"x": 1}])
    llm.with_structured_output(Out).invoke(["q"])
    assert llm.messages == [["q"]]
```

Append to `tests/test_hitl.py`:

```python
def test_flat_state_tolerates_chained_access_on_missing_keys():
    flat = flat_state({"task": "t", "scratch": {"a": {"b": "x"}}})
    assert "[{scratch.nope.deeper}][{scratch.a.zz.more}][{scratch.a.b}]".format_map(flat) == "[][][x]"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd $LGA && git switch -c feat/plan-execute-review && uv run pytest tests/test_resolve_llm.py tests/test_testing.py tests/test_hitl.py -q`
Expected: `ModuleNotFoundError: No module named 'lgkit.nodes._llm'`, `AttributeError: 'ScriptedLLM' object has no attribute 'invoke'`, and an `AttributeError` from the chained template.

- [ ] **Step 3: Create `src/lgkit/nodes/_llm.py`**

```python
"""Pick the model a node should call. Shared by every node that calls an LLM.

A dry-run ALWAYS wins: it must never build a real model, even when the node
carries its own ``llm`` spec or the run context has one. If a dry-run is bound
in the ambient context but did not reach the node as ``ctx``, the ctx plumbing
regressed — refuse to build a real model rather than silently spend API money.
"""

from __future__ import annotations

from typing import Any

from lgkit.context import current_ctx


def resolve_llm(params: dict[str, Any], ctx: Any, node_id: str) -> Any:
    """Precedence: ctx dry-run -> refuse on a lost dry-run -> the node's own
    ``llm`` spec -> the run context's llm -> lgkit's default."""
    from lgkit.llm import build_llm

    if ctx is not None and getattr(ctx, "dry_run", None) is not None:
        return ctx.dry_run.model_for(node_id)
    bound_ctx = current_ctx()
    if bound_ctx is not None and getattr(bound_ctx, "dry_run", None) is not None:
        # Checked before the node's llm spec: that spec builds a real model too.
        raise RuntimeError("dry-run context lost: refusing to build a real model")
    if params.get("llm"):
        return build_llm(**params["llm"])
    if ctx is not None and getattr(ctx, "llm", None) is not None:
        return ctx.llm
    return build_llm()


__all__ = ["resolve_llm"]
```

- [ ] **Step 4: Make `advisor._llm_for` delegate**

In `src/lgkit/nodes/advisor.py` replace the whole `_llm_for` function with:

```python
def _llm_for(params: dict[str, Any], ctx: Any) -> Any:
    """The model this advisor calls — see lgkit.nodes._llm.resolve_llm."""
    return resolve_llm(params, ctx, params.get("__node_id") or f"{params['gate_id']}__advisor")
```

Add `from lgkit.nodes._llm import resolve_llm` to the imports and remove the now-unused `from lgkit.context import current_ctx` import.

- [ ] **Step 5: Extend `ScriptedLLM` in `src/lgkit/testing.py`**

Replace the `ScriptedLLM` and `_Structured` classes with:

```python
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
```

Add `import json` and `from langchain_core.messages import AIMessage` to the imports.

- [ ] **Step 6: Make missing template keys chainable in `src/lgkit/nodes/_template.py`**

Add above `_DefaultDict`:

```python
class _Blank(str):
    """What a missing key renders as: "" that also tolerates further
    attribute access, so ``{scratch.review.feedback}`` is empty (not an
    AttributeError) before any review exists."""

    def __getattr__(self, key: str):
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        return self


_BLANK = _Blank("")
```

In `_DefaultDict.__missing__` and `_AttrDict.__missing__` return `_BLANK` instead of `""`, and in `_AttrDict.__getattr__` change `value = self.get(key, "")` to `value = self.get(key, _BLANK)`.

- [ ] **Step 7: Run everything**

Run: `uv run pytest -q && uv run ruff check src tests examples && grep -rn lgtools src; true`
Expected: all pass (128 + 3 + 2 + 1 = 134); ruff clean; grep prints nothing.

- [ ] **Step 8: Commit**

```bash
git add src/lgkit/nodes/_llm.py src/lgkit/nodes/advisor.py src/lgkit/testing.py src/lgkit/nodes/_template.py tests
git commit -m "refactor(lgkit): shared resolve_llm; ScriptedLLM plain invoke; chainable blank templates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 2: `llm` node

**Files:**
- Create: `$LGA/src/lgkit/nodes/llm.py`
- Modify: `$LGA/src/lgkit/nodes/__init__.py` (register), `$LGA/src/lgkit/spec.py` (`NodeKind`), `$LGA/src/lgkit/node_validation.py`, `$LGA/src/lgkit/builder.py` (end-on-miss kinds)
- Test: `$LGA/tests/test_llm_node.py`

**Interfaces:**
- Consumes: `resolve_llm`, `flat_state`, `ScriptedLLM`, `using_llm`.
- Produces: kind `"llm"`; `lgkit.nodes.llm.run(state, params, prompt, ctx=None, resolved=None) -> dict` returning `{"scratch": {result_key: str | dict}, "events": [...]}` plus `"signal"` when `signal_field` is set. Params: `message` (default `"{task}"`), `result_key` (default node id), `output_fields: list[{name, type?, description?, required?}]`, `signal_field`, `signal_values`, `llm`, `max_attempts` (default 2). Failure after all attempts raises `RuntimeError("llm node '<id>' failed after <N> attempts: <last error>")`. Dry-run: if `ctx.dry_run` has a `next_mock(node_id) -> str | None` method (lgtools' `DryRunState` does), the node takes ITS NEXT scripted reply from there on every visit — `model_for()` hands out a fresh fake model per call, which would replay the first reply on every pass of a loop; `None` (script exhausted) is a failed attempt. A dry-run object without `next_mock` falls back to `model_for(node_id)`. `lgkit.builder._END_ON_MISS_KINDS = frozenset({"approval", "llm"})` (Task 3 adds `loop_limit`).

- [ ] **Step 1: Write the failing tests** — `tests/test_llm_node.py`

```python
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

import lgkit.llm
from lgkit.builder import GraphBuildError, build_graph
from lgkit.context import Budget, RunContext, clear_ctx, use_ctx
from lgkit.registry import NodeDef, register
from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec
from lgkit.testing import ScriptedLLM, using_llm

VERDICT_FIELDS = [
    {"name": "verdict", "type": "string"},
    {"name": "feedback", "type": "string", "required": False},
]


def _one(params=None, prompt=None, edges=()) -> WorkflowSpec:
    return WorkflowSpec(
        nodes=[NodeSpec(id="n", kind="llm", prompt=prompt, params=params or {})],
        edges=list(edges),
        entry="n",
    )


def _run(spec, llm, state=None):
    app = build_graph(spec)
    with using_llm(llm):
        return app.invoke(state or {"task": "write a haiku"})


def test_plain_text_goes_to_scratch_under_the_node_id():
    llm = ScriptedLLM(["five seven five"])
    out = _run(_one(prompt="You are a poet."), llm)
    assert out["scratch"]["n"] == "five seven five"
    system, human = llm.messages[0]
    assert system.content == "You are a poet." and human.content == "write a haiku"


def test_message_template_and_result_key():
    llm = ScriptedLLM(["ok"])
    spec = _one({"message": "Plan: {scratch.plan} / {scratch.missing.deeper}", "result_key": "out"})
    out = _run(spec, llm, {"task": "t", "scratch": {"plan": "P"}})
    assert out["scratch"]["out"] == "ok"
    assert llm.messages[0][-1].content == "Plan: P / "


def test_no_system_message_when_prompt_is_empty():
    llm = ScriptedLLM(["ok"])
    _run(_one(), llm)
    assert len(llm.messages[0]) == 1


def test_structured_output_and_signal():
    llm = ScriptedLLM([{"verdict": "revise", "feedback": "add examples"}])
    spec = _one(
        {"output_fields": VERDICT_FIELDS, "signal_field": "verdict", "signal_values": ["approve", "revise"]},
        edges=[
            EdgeSpec(source="n", target="END", condition="approve"),
            EdgeSpec(source="n", target="END", condition="revise"),
        ],
    )
    out = _run(spec, llm)
    assert out["scratch"]["n"] == {"verdict": "revise", "feedback": "add examples"}
    assert out["signal"] == "revise"


def test_signal_outside_allowed_values_is_a_failed_attempt_then_raises():
    bad = {"verdict": "maybe", "feedback": ""}
    llm = ScriptedLLM([bad, bad])
    spec = _one({"output_fields": VERDICT_FIELDS, "signal_field": "verdict", "signal_values": ["approve", "revise"]})
    with pytest.raises(RuntimeError, match=r"llm node 'n' failed after 2 attempts: ValidationError"):
        _run(spec, llm)
    assert llm.calls == 2


def test_retry_recovers():
    llm = ScriptedLLM([RuntimeError("blip"), "fine"])
    assert _run(_one(), llm)["scratch"]["n"] == "fine"


def test_failure_raises_with_node_name_and_last_error():
    llm = ScriptedLLM([RuntimeError("down"), RuntimeError("still down")])
    with pytest.raises(RuntimeError, match=r"llm node 'n' failed after 2 attempts: RuntimeError: still down"):
        _run(_one(), llm)


def test_max_attempts_one():
    llm = ScriptedLLM([RuntimeError("down")])
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        _run(_one({"max_attempts": 1}), llm)
    assert llm.calls == 1


class _FakeModel:
    """Like an embedding app's dry-run model: with_structured_output returns
    itself and invoke returns a plain AIMessage."""

    def __init__(self, reply):
        self.reply = reply

    def with_structured_output(self, schema, **_k):
        return self

    def invoke(self, _messages, **_k):
        return AIMessage(content=self.reply)


class _DryRun:
    def __init__(self, reply):
        self.reply = reply
        self.asked = []

    def model_for(self, node_id, spec=None):
        self.asked.append(node_id)
        return _FakeModel(self.reply)


def _dry_ctx(dry):
    return RunContext(session_id="t", workspace=Path("."), llm=object(), budget=Budget(10, 50), dry_run=dry)


def test_dry_run_wins_over_node_llm_and_parses_json_content(monkeypatch):
    monkeypatch.setattr(lgkit.llm, "build_llm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real LLM")))
    dry = _DryRun('{"verdict": "approve", "feedback": ""}')
    spec = _one({
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "output_fields": VERDICT_FIELDS, "signal_field": "verdict", "signal_values": ["approve", "revise"],
    })
    token = use_ctx(_dry_ctx(dry))
    try:
        out = build_graph(spec).invoke({"task": "t"})
    finally:
        clear_ctx(token)
    assert dry.asked == ["n"] and out["signal"] == "approve"


def test_dry_run_without_a_script_fails_with_a_hint(monkeypatch):
    monkeypatch.setattr(lgkit.llm, "build_llm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real LLM")))
    spec = _one({"output_fields": VERDICT_FIELDS, "signal_field": "verdict", "signal_values": ["approve", "revise"]})
    token = use_ctx(_dry_ctx(_DryRun("{}")))
    try:
        with pytest.raises(RuntimeError, match=r"mock_scripts\['n'\]"):
            build_graph(spec).invoke({"task": "t"})
    finally:
        clear_ctx(token)


class _SequencedDryRun(_DryRun):
    """Like lgtools' DryRunState: next_mock() hands out one scripted reply per
    visit, None once the script is exhausted."""

    def __init__(self, replies):
        super().__init__("{}")
        self.replies = list(replies)

    def next_mock(self, node_id):
        self.asked.append(node_id)
        return self.replies.pop(0) if self.replies else None


def test_dry_run_with_next_mock_advances_per_visit_and_fails_when_exhausted(monkeypatch):
    # A node revisited in a loop must get its NEXT scripted reply, not the first again.
    monkeypatch.setattr(lgkit.llm, "build_llm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real LLM")))
    dry = _SequencedDryRun(["first", "second"])
    app = build_graph(_one())
    token = use_ctx(_dry_ctx(dry))
    try:
        assert app.invoke({"task": "t"})["scratch"]["n"] == "first"
        assert app.invoke({"task": "t"})["scratch"]["n"] == "second"
        with pytest.raises(RuntimeError, match=r"no scripted reply left.*mock_scripts\['n'\]"):
            app.invoke({"task": "t"})
    finally:
        clear_ctx(token)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"output_fields": [{"name": "a"}], "signal_field": "verdict", "signal_values": ["x"]}, "signal_field"),
        ({"output_fields": [{"name": "verdict"}], "signal_field": "verdict"}, "signal_values"),
        ({"signal_field": "verdict", "signal_values": ["x"]}, "signal_field"),
        ({"output_fields": [{"type": "string"}]}, "output_fields"),
    ],
)
def test_bad_llm_params_fail_at_build_time(params, match):
    with pytest.raises(GraphBuildError, match=match):
        build_graph(_one(params))


def test_unmatched_signal_from_an_llm_node_ends_the_run():
    ran = []
    register(
        "custom",
        NodeDef(kind="custom", fn=lambda s, p, pr, c=None, r=None: ran.append(p["__node_id"]) or {},
                default_prompt="", description="probe"),
    )
    spec = WorkflowSpec(
        nodes=[
            NodeSpec(id="n", kind="llm", params={
                "output_fields": [{"name": "verdict"}], "signal_field": "verdict", "signal_values": ["go", "stop"],
            }),
            NodeSpec(id="first", kind="custom"),
        ],
        # Only "go" has an edge: "stop" matches nothing and must END, not take the first edge.
        edges=[EdgeSpec(source="n", target="first", condition="go")],
        entry="n",
    )
    _run(spec, ScriptedLLM([{"verdict": "stop"}]))
    assert ran == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_llm_node.py -q`
Expected: every test fails — pydantic rejects `kind="llm"` (`Input should be 'custom', 'agent', …`).

- [ ] **Step 3: Add the kind to `NodeKind`** — in `src/lgkit/spec.py`, after `"approval",` add `"llm",`.

- [ ] **Step 4: Create `src/lgkit/nodes/llm.py`**

```python
"""llm node — one LLM call. The result goes to scratch; a structured result
can also emit the routing signal.

It has no tools and no inner loop: looping is the graph's job. On failure it
RAISES (after ``max_attempts``) — unlike the advisor there is no human waiting
to catch it, and a node that swallowed the failure would leave a stale or
missing signal for the router to act on.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel, Field, create_model

from lgkit.nodes._llm import resolve_llm
from lgkit.nodes._template import flat_state

_TYPES: dict[str, Any] = {
    "string": str,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def output_model(
    fields: list[dict[str, Any]], signal_field: str | None, signal_values: list[str]
) -> type[BaseModel]:
    """Build the structured-output schema. The signal field is constrained to
    ``signal_values`` so the model cannot answer something the graph has no
    edge for."""
    defs: dict[str, Any] = {}
    for f in fields:
        name = f["name"]
        if name == signal_field:
            typ: Any = Literal[tuple(signal_values)]
        else:
            typ = _TYPES.get(str(f.get("type") or "string"), str)
        desc = f.get("description")
        if f.get("required", True):
            defs[name] = (typ, Field(..., description=desc))
        else:
            defs[name] = (typ | None, Field(None, description=desc))
    return create_model("LlmOutput", **defs)


def _text(out: Any) -> str:
    content = out.content if isinstance(out, BaseMessage) else out
    return content if isinstance(content, str) else str(content)


def _structured(out: Any, schema: type[BaseModel]) -> dict[str, Any]:
    # A fake/scripted model returns a plain AIMessage (itself a pydantic
    # model) whose .content is the reply: parse that as JSON before the
    # BaseModel branch can misread it.
    if isinstance(out, BaseMessage):
        data = json.loads(_text(out))
    elif isinstance(out, BaseModel):
        data = out.model_dump()
    else:
        data = out
    return schema.model_validate(data).model_dump()


def _call(params, ctx, node_id: str, messages: list, schema: type[BaseModel] | None) -> Any:
    """One model call, returning the raw reply.

    A dry-run that can hand out per-visit scripted replies (``next_mock``) is
    asked directly: ``model_for()`` builds a fresh fake model on every call,
    which would replay the FIRST scripted reply each time a loop revisits this
    node.
    """
    dry = getattr(ctx, "dry_run", None) if ctx is not None else None
    if dry is not None and hasattr(dry, "next_mock"):
        raw = dry.next_mock(node_id)
        if raw is None:
            raise RuntimeError("no scripted reply left")
        return AIMessage(content=raw)
    model = resolve_llm(params, ctx, node_id)
    return (model if schema is None else model.with_structured_output(schema)).invoke(messages)


def run(state, params, prompt, ctx=None, resolved=None) -> dict[str, Any]:
    node_id = params.get("__node_id") or "llm"
    message = str(params.get("message") or "{task}").format_map(flat_state(state))
    fields = list(params.get("output_fields") or [])
    signal_field = params.get("signal_field")
    schema = (
        output_model(fields, signal_field, list(params.get("signal_values") or []))
        if fields
        else None
    )
    messages = ([SystemMessage(prompt)] if prompt else []) + [HumanMessage(message)]
    attempts = max(1, int(params.get("max_attempts", 2)))

    last_error = ""
    for _ in range(attempts):
        try:
            out = _call(params, ctx, node_id, messages, schema)
            result: Any = _text(out) if schema is None else _structured(out, schema)
            break
        except GraphBubbleUp:
            raise
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
    else:
        hint = ""
        if ctx is not None and getattr(ctx, "dry_run", None) is not None:
            hint = f" — dry-run: script this node's reply with mock_scripts['{node_id}']"
        raise RuntimeError(
            f"llm node '{node_id}' failed after {attempts} attempts: {last_error}{hint}"
        )

    delta: dict[str, Any] = {
        "scratch": {params.get("result_key") or node_id: result},
        "events": [{"node": node_id, "result": result, "tools": []}],
    }
    if signal_field:
        delta["signal"] = result[signal_field]
    return delta


__all__ = ["output_model", "run"]
```

- [ ] **Step 5: Register the kind** — in `src/lgkit/nodes/__init__.py` change the import to `from lgkit.nodes import advisor, approval, hitl, llm` and append to `_builtin_defs()`:

```python
        NodeDef(
            kind="llm",
            fn=llm.run,
            default_prompt="",
            description="One LLM call — text or structured output into scratch; a structured field can be the routing signal.",
            default_params={"message": "{task}", "max_attempts": 2},
            uses_node_prompt=True,
        ),
```

- [ ] **Step 6: Build-time validation** — in `src/lgkit/node_validation.py`, add another branch to `_problems` after the `approval` branch:

```python
        elif node.kind == "llm":
            fields = params.get("output_fields") or []
            names = [f.get("name") for f in fields if isinstance(f, dict)]
            if len(names) != len(fields) or any(_is_blank(n) for n in names):
                found.append((f"{where}: every entry of 'output_fields' needs a 'name'", True))
            signal_field = params.get("signal_field")
            if not _is_blank(signal_field):
                if signal_field not in names:
                    found.append(
                        (f"{where}: 'signal_field' must name one of 'output_fields'", True)
                    )
                if _is_blank(params.get("signal_values")):
                    found.append(
                        (f"{where}: 'signal_values' is required when 'signal_field' is set", True)
                    )
```

- [ ] **Step 7: Builder — end the run on an unmatched signal from an `llm` source**

In `src/lgkit/builder.py`:

a) below `_NEEDS_CHECKPOINTER` add:
```python
# Sources whose unmatched signal ENDs the run instead of falling through to an
# exit-key choice or the first conditional edge. For these kinds a fallthrough
# would act on a decision nobody made (auto-approve, or loop forever).
_END_ON_MISS_KINDS: frozenset[str] = frozenset({"approval", "llm"})
```
b) replace
```python
            from_approval = src_node is not None and src_node.kind == "approval"
            router = _signal_router(conditions, end_on_miss=from_approval)
            if from_approval:
                mapping.setdefault("END", END)
```
with
```python
            end_on_miss = src_node is not None and src_node.kind in _END_ON_MISS_KINDS
            router = _signal_router(conditions, end_on_miss=end_on_miss)
            if end_on_miss:
                mapping.setdefault("END", END)
```
c) update the two comments that say "approval" only (above this block and inside `_signal_router`) to say "approval / llm / loop_limit sources (see _END_ON_MISS_KINDS)".

- [ ] **Step 8: Run everything**

Run: `uv run pytest -q && uv run ruff check src tests examples && grep -rn lgtools src; true`
Expected: all pass (134 + 15 = 149); ruff clean; grep prints nothing.

- [ ] **Step 9: Commit**

```bash
git add src/lgkit tests/test_llm_node.py
git commit -m "feat(lgkit): llm node — one LLM call with optional structured output and signal

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 3: `loop_limit` node

**Files:**
- Create: `$LGA/src/lgkit/nodes/loop_limit.py`
- Modify: `$LGA/src/lgkit/nodes/__init__.py`, `$LGA/src/lgkit/spec.py` (`NodeKind`), `$LGA/src/lgkit/node_validation.py`, `$LGA/src/lgkit/builder.py` (`_END_ON_MISS_KINDS`)
- Test: `$LGA/tests/test_loop_limit.py`

**Interfaces:**
- Produces: kind `"loop_limit"`; `lgkit.nodes.loop_limit.run(...)` returning `{"signal": "again" | "exhausted", "scratch": {...}}`. Params: `max_rounds` (int ≥ 1, required), `counter_key` (default `"<node id>__count"`), `exhausted_key` (default `"<node id>__exhausted"`). `count < max_rounds` → `again`; otherwise `exhausted` and `scratch[exhausted_key] = True`. `_END_ON_MISS_KINDS` now includes `"loop_limit"`.

- [ ] **Step 1: Write the failing tests** — `tests/test_loop_limit.py`

```python
import pytest

from lgkit.builder import GraphBuildError, build_graph
from lgkit.nodes import loop_limit
from lgkit.registry import NodeDef, register
from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec


def _call(scratch, **params):
    return loop_limit.run({"scratch": scratch}, {"__node_id": "lim", **params}, "")


def test_counts_and_says_again_until_the_limit():
    first = _call({}, max_rounds=3)
    assert first == {"signal": "again", "scratch": {"lim__count": 1}}
    second = _call({"lim__count": 1}, max_rounds=3)
    assert second == {"signal": "again", "scratch": {"lim__count": 2}}
    third = _call({"lim__count": 2}, max_rounds=3)
    assert third == {"signal": "exhausted", "scratch": {"lim__count": 3, "lim__exhausted": True}}


def test_max_rounds_one_is_exhausted_immediately():
    assert _call({}, max_rounds=1)["signal"] == "exhausted"


def test_custom_keys():
    out = _call({}, max_rounds=1, counter_key="round", exhausted_key="gave_up")
    assert out["scratch"] == {"round": 1, "gave_up": True}


@pytest.mark.parametrize("params", [{}, {"max_rounds": 0}, {"max_rounds": "many"}, {"max_rounds": 1.5}])
def test_bad_max_rounds_fails_at_build_time(params):
    spec = WorkflowSpec(nodes=[NodeSpec(id="lim", kind="loop_limit", params=params)], entry="lim")
    with pytest.raises(GraphBuildError, match="max_rounds"):
        build_graph(spec)


def _loop_spec(max_rounds: int) -> WorkflowSpec:
    return WorkflowSpec(
        nodes=[
            NodeSpec(id="work", kind="custom"),
            NodeSpec(id="lim", kind="loop_limit", params={"max_rounds": max_rounds}),
        ],
        edges=[
            EdgeSpec(source="work", target="lim"),
            EdgeSpec(source="lim", target="work", condition="again"),
            EdgeSpec(source="lim", target="END", condition="exhausted"),
        ],
        entry="work",
    )


def _count_work():
    ran = []
    register(
        "custom",
        NodeDef(kind="custom", fn=lambda s, p, pr, c=None, r=None: ran.append(1) or {},
                default_prompt="", description="work"),
    )
    return ran


def test_loop_runs_exactly_max_rounds_times():
    ran = _count_work()
    out = build_graph(_loop_spec(3)).invoke({"task": "t"})
    assert len(ran) == 3
    assert out["scratch"]["lim__exhausted"] is True and out["signal"] == "exhausted"


def test_global_iteration_budget_ends_the_run_instead_of_looping():
    # The builder puts an iteration guard after `lim` (a back-edge source).
    # Once the run-wide budget is spent the guard emits "max_iterations", which
    # is neither "again" nor "exhausted": the run must END, not take the first
    # edge ("again") forever.
    ran = _count_work()
    out = build_graph(_loop_spec(50)).invoke({"task": "t", "scratch": {"max_iterations": 2}})
    assert len(ran) == 2
    assert out["signal"] == "max_iterations"
    assert "lim__exhausted" not in out["scratch"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_loop_limit.py -q`
Expected: `ImportError: cannot import name 'loop_limit' from 'lgkit.nodes'`.

- [ ] **Step 3: Add the kind to `NodeKind`** — in `src/lgkit/spec.py`, after `"llm",` add `"loop_limit",`.

- [ ] **Step 4: Create `src/lgkit/nodes/loop_limit.py`**

```python
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
```

- [ ] **Step 5: Register** — in `src/lgkit/nodes/__init__.py` change the import to `from lgkit.nodes import advisor, approval, hitl, llm, loop_limit` and append to `_builtin_defs()`:

```python
        NodeDef(
            kind="loop_limit",
            fn=loop_limit.run,
            default_prompt="",
            description="Counts passes through a loop: 'again' until max_rounds, then 'exhausted'.",
            default_params={"max_rounds": 3},
            is_router=True,
        ),
```

Note: `default_params` are merged under a node's own params at run time, but validation reads the node's own params — the editor seeds new nodes from `default_params`, and hand-written specs must set `max_rounds`.

- [ ] **Step 6: Validation** — in `src/lgkit/node_validation.py` add after the `llm` branch:

```python
        elif node.kind == "loop_limit":
            max_rounds = params.get("max_rounds")
            if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
                found.append((f"{where}: 'max_rounds' must be a whole number >= 1", True))
```

- [ ] **Step 7: Builder** — in `src/lgkit/builder.py` change `_END_ON_MISS_KINDS` to `frozenset({"approval", "llm", "loop_limit"})`.

- [ ] **Step 8: Run everything**

Run: `uv run pytest -q && uv run ruff check src tests examples`
Expected: all pass (149 + 9 = 158); ruff clean.

- [ ] **Step 9: Commit**

```bash
git add src/lgkit tests/test_loop_limit.py
git commit -m "feat(lgkit): loop_limit node — a pattern's own round budget

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 4: `Fragment.exits` and `plan_execute_review()`

**Files:**
- Create: `$LGA/src/lgkit/patterns/_fragment.py`, `$LGA/src/lgkit/patterns/plan_execute_review.py`, `$LGA/examples/plan_execute_review_pattern.py`
- Modify: `$LGA/src/lgkit/patterns/approval_gate.py` (import `Fragment`, fill `exits`), `$LGA/src/lgkit/patterns/__init__.py`, `$LGA/README.md`
- Test: `$LGA/tests/test_plan_execute_review.py`, `$LGA/tests/test_approval_gate.py` (one added test), `$LGA/tests/test_examples.py` (one added test)

**Interfaces:**
- Consumes: kinds `llm`, `loop_limit`, `approval_gate`, `ScriptedLLM`, `using_llm`.
- Produces:
  - `lgkit.patterns.Fragment` gains `exits: dict[str, tuple[str, str]]` (exit name → `(node id, condition)`, default `{}`). `to_workflow()` routes every `exits` entry to `END`; with empty `exits` it falls back to `exit`/`choices` as before.
  - `approval_gate(...)` fills `exits = {choice: (id, choice)}`.
  - `plan_execute_review(id, *, planner: str, executor: str | NodeSpec, reviewer: str, max_rounds: int = 3, llm: dict | None = None, executor_result_key: str | None = None) -> Fragment` with nodes `<id>__plan`, `<id>__execute` (or the given `NodeSpec`), `<id>__review`, `<id>__limit`; scratch keys `<id>__plan`, `<id>__result` (or `executor_result_key`), `<id>__review` (`{verdict, feedback}`), `<id>__round`, `<id>__exhausted`; `entry = "<id>__plan"`, `exit = "<id>__review"`, `choices = ()`, `exits = {"approved": ("<id>__review", "approve"), "exhausted": ("<id>__limit", "exhausted")}`.

- [ ] **Step 1: Write the failing tests** — `tests/test_plan_execute_review.py`

```python
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from lgkit.builder import build_graph
from lgkit.patterns import Fragment, approval_gate, plan_execute_review
from lgkit.registry import NodeDef, register
from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec
from lgkit.testing import ScriptedLLM, using_llm

APPROVE = {"verdict": "approve", "feedback": ""}


def _per(**kw) -> Fragment:
    args = {"planner": "Plan it.", "executor": "Do it.", "reviewer": "Check it.", **kw}
    return plan_execute_review("doc", **args)


def _run(wf, llm, state=None, **build_kw):
    app = build_graph(wf, **build_kw)
    with using_llm(llm):
        return app.invoke(state or {"task": "write docs"}, {"configurable": {"thread_id": "t"}})


def test_fragment_shape():
    frag = _per()
    assert [(n.id, n.kind) for n in frag.nodes] == [
        ("doc__plan", "llm"), ("doc__execute", "llm"), ("doc__review", "llm"), ("doc__limit", "loop_limit"),
    ]
    assert frag.entry == "doc__plan"
    assert frag.exits == {"approved": ("doc__review", "approve"), "exhausted": ("doc__limit", "exhausted")}
    assert {(e.source, e.target, e.condition) for e in frag.edges} == {
        ("doc__plan", "doc__execute", None),
        ("doc__execute", "doc__review", None),
        ("doc__review", "doc__limit", "revise"),
        ("doc__limit", "doc__execute", "again"),
    }


def test_llm_spec_is_applied_to_every_llm_node():
    frag = _per(llm={"provider": "openai", "model": "gpt-4o-mini"})
    assert all(n.params["llm"] == {"provider": "openai", "model": "gpt-4o-mini"} for n in frag.nodes[:3])


def test_approved_on_the_first_round():
    llm = ScriptedLLM(["the plan", "the work", APPROVE])
    out = _run(_per().to_workflow(), llm)
    assert out["scratch"]["doc__plan"] == "the plan"
    assert out["scratch"]["doc__result"] == "the work"
    assert out["scratch"]["doc__review"]["verdict"] == "approve"
    assert "doc__exhausted" not in out["scratch"]
    assert llm.calls == 3


def test_revise_sends_feedback_back_to_the_executor_and_plans_once():
    llm = ScriptedLLM([
        "the plan",
        "draft one",
        {"verdict": "revise", "feedback": "add an example"},
        "draft two",
        APPROVE,
    ])
    out = _run(_per().to_workflow(), llm)
    assert out["scratch"]["doc__result"] == "draft two"
    assert llm.calls == 5  # plan once, execute twice, review twice
    first_exec, second_exec = llm.messages[1][-1].content, llm.messages[3][-1].content
    assert "the plan" in first_exec and "add an example" not in first_exec
    assert "add an example" in second_exec and "draft one" in second_exec
    assert "draft two" in llm.messages[4][-1].content  # the reviewer sees the new work


def test_exhausted_after_max_rounds_is_not_an_approval():
    revise = {"verdict": "revise", "feedback": "no"}
    llm = ScriptedLLM(["plan", "w1", revise, "w2", revise])
    out = _run(_per(max_rounds=2).to_workflow(), llm)
    assert out["scratch"]["doc__exhausted"] is True
    assert out["scratch"]["doc__round"] == 2
    assert out["scratch"]["doc__review"]["verdict"] == "revise"
    assert out["scratch"]["doc__result"] == "w2"
    assert llm.calls == 5  # execute ran exactly max_rounds times


def test_custom_executor_node():
    def apply(state, params, prompt, ctx=None, resolved=None):
        n = int((state.get("scratch") or {}).get("applied", 0)) + 1
        return {"scratch": {"applied": n, "patch": f"patch v{n}"}}

    register("custom", NodeDef(kind="custom", fn=apply, default_prompt="", description="apply"))
    frag = _per(executor=NodeSpec(id="apply", kind="custom"), executor_result_key="patch")
    assert [n.id for n in frag.nodes] == ["doc__plan", "apply", "doc__review", "doc__limit"]
    llm = ScriptedLLM(["plan", {"verdict": "revise", "feedback": "again"}, APPROVE])
    out = _run(frag.to_workflow(), llm)
    assert out["scratch"]["patch"] == "patch v2"
    assert "patch v2" in llm.messages[2][-1].content  # the reviewer reads the custom key


def test_two_patterns_in_one_workflow_count_rounds_separately():
    a = plan_execute_review("a", planner="p", executor="e", reviewer="r", max_rounds=2)
    b = plan_execute_review("b", planner="p", executor="e", reviewer="r", max_rounds=3)
    wf = WorkflowSpec(
        nodes=[*a.nodes, *b.nodes],
        edges=[
            *a.edges,
            *b.edges,
            EdgeSpec(source=a.exits["approved"][0], target=b.entry, condition=a.exits["approved"][1]),
            EdgeSpec(source=a.exits["exhausted"][0], target=b.entry, condition=a.exits["exhausted"][1]),
            EdgeSpec(source=b.exits["approved"][0], target="END", condition=b.exits["approved"][1]),
            EdgeSpec(source=b.exits["exhausted"][0], target="END", condition=b.exits["exhausted"][1]),
        ],
        entry=a.entry,
    )
    revise = {"verdict": "revise", "feedback": "x"}
    llm = ScriptedLLM(["pa", "a1", revise, "a2", revise, "pb", "b1", revise, "b2", APPROVE])
    out = _run(wf, llm, {"task": "t", "scratch": {"max_iterations": 50}})
    assert out["scratch"]["a__round"] == 2 and out["scratch"]["a__exhausted"] is True
    assert out["scratch"]["b__round"] == 1 and "b__exhausted" not in out["scratch"]


def test_exhausted_exit_can_feed_an_approval_gate():
    per = _per(max_rounds=1)
    gate = approval_gate("g", message="Accept anyway?\n{scratch.doc__result}", choices=["accept", "drop"])
    wf = WorkflowSpec(
        nodes=[*per.nodes, *gate.nodes],
        edges=[
            *per.edges,
            *gate.edges,
            EdgeSpec(source="doc__review", target="END", condition="approve"),
            EdgeSpec(source="doc__limit", target=gate.entry, condition="exhausted"),
            EdgeSpec(source="g", target="END", condition="accept"),
            EdgeSpec(source="g", target="END", condition="drop"),
        ],
        entry=per.entry,
    )
    app = build_graph(wf, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "g"}}
    with using_llm(ScriptedLLM(["plan", "w1", {"verdict": "revise", "feedback": "no"}])):
        app.invoke({"task": "t"}, cfg)
    ask = app.get_state(cfg).interrupts[0].value
    assert ask["node"] == "g" and "w1" in ask["message"]
    out = app.invoke(Command(resume="drop"), cfg)
    assert out["scratch"]["g"]["choice"] == "drop" and out["scratch"]["doc__exhausted"] is True


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"max_rounds": 0}, "max_rounds"),
        ({"planner": "  "}, "planner"),
        ({"reviewer": ""}, "reviewer"),
        ({"executor": ""}, "executor"),
        ({"executor": NodeSpec(id="x", kind="custom")}, "executor_result_key"),
        ({"executor_result_key": "k"}, "executor_result_key"),
        ({"executor": NodeSpec(id="doc__review", kind="custom"), "executor_result_key": "k"}, "doc__review"),
    ],
)
def test_invalid_arguments_fail_at_build_time(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _per(**kwargs)
```

Append to `tests/test_approval_gate.py`:

```python
def test_approval_gate_fills_named_exits():
    frag = approval_gate("g", message="Ok?", choices=["yes", "no"])
    assert frag.exits == {"yes": ("g", "yes"), "no": ("g", "no")}
```

Append to `tests/test_examples.py`:

```python
def test_plan_execute_review_pattern_example_runs_offline(monkeypatch):
    monkeypatch.delenv("LGKIT_REAL_LLM", raising=False)
    out = _load("plan_execute_review_pattern").main()
    assert out["approved"] is True and out["rounds"] == 1 and out["result"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_plan_execute_review.py tests/test_approval_gate.py tests/test_examples.py -q`
Expected: `ImportError: cannot import name 'plan_execute_review' from 'lgkit.patterns'`.

- [ ] **Step 3: Create `src/lgkit/patterns/_fragment.py`** and remove the `Fragment` class from `approval_gate.py`

```python
"""Fragment — the spec data a pattern returns."""

from __future__ import annotations

from dataclasses import dataclass, field

from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec


@dataclass(frozen=True)
class Fragment:
    """Nodes + internal edges of a pattern.

    Wire an edge into ``entry``. Leave through ``exits``: each named exit is a
    ``(node id, condition)`` pair — add ``EdgeSpec(source=node, target=...,
    condition=condition)``. ``exit``/``choices`` are the single-gate shorthand
    approval_gate has always had.
    """

    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    entry: str
    exit: str
    choices: tuple[str, ...]
    exits: dict[str, tuple[str, str]] = field(default_factory=dict)

    def to_workflow(self, name: str = "approval_gate") -> WorkflowSpec:
        """A standalone workflow: every exit routes to END."""
        if self.exits:
            ends = [EdgeSpec(source=n, target="END", condition=c) for n, c in self.exits.values()]
        else:
            ends = [EdgeSpec(source=self.exit, target="END", condition=c) for c in self.choices]
        return WorkflowSpec(
            name=name, nodes=list(self.nodes), edges=[*self.edges, *ends], entry=self.entry
        )


__all__ = ["Fragment"]
```

In `src/lgkit/patterns/approval_gate.py`: delete the `Fragment` dataclass and its now-unused imports (`dataclass`, `WorkflowSpec` if unused), add `from lgkit.patterns._fragment import Fragment`, and change the final `return Fragment(...)` to pass `exits={c: (id, c) for c in choices}`. Keep `"Fragment"` in that module's `__all__`.

- [ ] **Step 4: Create `src/lgkit/patterns/plan_execute_review.py`**

```python
"""plan_execute_review — plan once, then execute and review until the reviewer
approves or the round budget is spent. Returns plain spec data (a Fragment).

    <id>__plan -> <id>__execute -> <id>__review --approve--> exit "approved"
                      ^                 |
                      |               revise
                      |                 v
                      +---again--- <id>__limit --exhausted--> exit "exhausted"

Running out of rounds is NOT an approval: the run leaves through "exhausted"
and scratch["<id>__exhausted"] is True.
"""

from __future__ import annotations

from typing import Any

from lgkit.nodes.loop_limit import AGAIN, EXHAUSTED
from lgkit.patterns._fragment import Fragment
from lgkit.spec import EdgeSpec, NodeSpec

APPROVE = "approve"
REVISE = "revise"


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"plan_execute_review: {name} must be a non-empty prompt")
    return value


def plan_execute_review(
    id: str,
    *,
    planner: str,
    executor: str | NodeSpec,
    reviewer: str,
    max_rounds: int = 3,
    llm: dict[str, Any] | None = None,
    executor_result_key: str | None = None,
) -> Fragment:
    """``max_rounds`` is the most times the execute step may run."""
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError(f"plan_execute_review: max_rounds must be a whole number >= 1, got {max_rounds!r}")
    _require_text("planner", planner)
    _require_text("reviewer", reviewer)

    plan_id, exec_id, review_id, limit_id = (f"{id}__plan", f"{id}__execute", f"{id}__review", f"{id}__limit")
    plan_key, review_key = plan_id, review_id
    llm_params: dict[str, Any] = {"llm": dict(llm)} if llm else {}

    if isinstance(executor, NodeSpec):
        if not executor_result_key:
            raise ValueError(
                "plan_execute_review: executor_result_key is required when executor is a NodeSpec "
                "(the scratch key your node writes its work to)"
            )
        if executor.id in (plan_id, review_id, limit_id):
            raise ValueError(f"plan_execute_review: executor id '{executor.id}' collides with a pattern node")
        exec_node = executor
        result_key = executor_result_key
    else:
        _require_text("executor", executor)
        if executor_result_key:
            raise ValueError(
                "plan_execute_review: executor_result_key only applies when executor is a NodeSpec"
            )
        result_key = f"{id}__result"
        exec_node = NodeSpec(
            id=exec_id,
            kind="llm",
            prompt=executor,
            params={
                **llm_params,
                "result_key": result_key,
                "message": (
                    "Task:\n{task}\n\n"
                    f"Plan:\n{{scratch.{plan_key}}}\n\n"
                    f"Your previous attempt (empty on the first round):\n{{scratch.{result_key}}}\n\n"
                    f"Reviewer feedback to address (empty on the first round):\n{{scratch.{review_key}.feedback}}"
                ),
            },
        )

    nodes = (
        NodeSpec(
            id=plan_id,
            kind="llm",
            prompt=planner,
            params={**llm_params, "result_key": plan_key, "message": "Task:\n{task}"},
        ),
        exec_node,
        NodeSpec(
            id=review_id,
            kind="llm",
            prompt=reviewer,
            params={
                **llm_params,
                "result_key": review_key,
                "message": (
                    "Task:\n{task}\n\n"
                    f"Plan:\n{{scratch.{plan_key}}}\n\n"
                    f"Work to review:\n{{scratch.{result_key}}}"
                ),
                "output_fields": [
                    {"name": "verdict", "type": "string",
                     "description": f"'{APPROVE}' only if the work meets the criteria, otherwise '{REVISE}'"},
                    {"name": "feedback", "type": "string", "required": False,
                     "description": "What must change; empty when approving"},
                ],
                "signal_field": "verdict",
                "signal_values": [APPROVE, REVISE],
            },
        ),
        NodeSpec(
            id=limit_id,
            kind="loop_limit",
            params={
                "max_rounds": max_rounds,
                "counter_key": f"{id}__round",
                "exhausted_key": f"{id}__exhausted",
            },
        ),
    )
    edges = (
        EdgeSpec(source=plan_id, target=exec_node.id),
        EdgeSpec(source=exec_node.id, target=review_id),
        EdgeSpec(source=review_id, target=limit_id, condition=REVISE),
        EdgeSpec(source=limit_id, target=exec_node.id, condition=AGAIN),
    )
    return Fragment(
        nodes=nodes,
        edges=edges,
        entry=plan_id,
        exit=review_id,
        choices=(),
        exits={"approved": (review_id, APPROVE), "exhausted": (limit_id, EXHAUSTED)},
    )


__all__ = ["APPROVE", "REVISE", "plan_execute_review"]
```

- [ ] **Step 5: Export** — `src/lgkit/patterns/__init__.py`:

```python
"""Ready-made workflow patterns. Each returns spec data, not a runtime."""

from lgkit.patterns._fragment import Fragment
from lgkit.patterns.approval_gate import approval_gate
from lgkit.patterns.plan_execute_review import plan_execute_review

__all__ = ["Fragment", "approval_gate", "plan_execute_review"]
```

- [ ] **Step 6: Example** — `examples/plan_execute_review_pattern.py`

```python
"""plan_execute_review demo: plan once, write, review, revise until approved.

    uv run python examples/plan_execute_review_pattern.py            # scripted LLM, no API key
    LGKIT_REAL_LLM=1 OPENAI_API_KEY=sk-... uv run python examples/plan_execute_review_pattern.py
"""

import os

from lgkit.builder import build_graph
from lgkit.patterns import plan_execute_review
from lgkit.testing import ScriptedLLM, using_llm


def make_llm():
    if os.environ.get("LGKIT_REAL_LLM"):
        from lgkit.llm import build_llm

        return build_llm()
    return ScriptedLLM([
        "1. Say what the endpoint does. 2. Show a request. 3. Show a response.",
        "GET /health returns service status.",
        {"verdict": "revise", "feedback": "Add the request and response examples from the plan."},
        "GET /health returns service status.\nRequest: curl /health\nResponse: {\"status\": \"ok\"}",
        {"verdict": "approve", "feedback": ""},
    ])


def main() -> dict:
    per = plan_execute_review(
        "doc",
        planner="Break the task into the steps a good answer must cover.",
        executor="Write the answer following the plan. Address any reviewer feedback.",
        reviewer="Approve only if every step of the plan is covered.",
        max_rounds=3,
    )
    app = build_graph(per.to_workflow(name="doc"))
    with using_llm(make_llm()):
        state = app.invoke({"task": "Document the GET /health endpoint."})
    scratch = state["scratch"]
    summary = {
        "approved": scratch["doc__review"]["verdict"] == "approve" and not scratch.get("doc__exhausted"),
        "rounds": scratch.get("doc__round", 0),
        "result": scratch["doc__result"],
    }
    print(summary)
    return summary


if __name__ == "__main__":
    main()
```

(`rounds` counts revise passes through the limit node — 1 in the scripted run.)

- [ ] **Step 7: README** — append to `README.md`:

````markdown
## Plan → execute → review

```python
from lgkit.builder import build_graph
from lgkit.patterns import plan_execute_review

per = plan_execute_review(
    "doc",
    planner="Break the task into steps.",
    executor="Write the document following the plan.",   # or a NodeSpec + executor_result_key
    reviewer="Approve only if every step of the plan is covered.",
    max_rounds=3,                                         # most times the execute step runs
)
app = build_graph(per.to_workflow())
state = app.invoke({"task": "Document the GET /health endpoint."})
state["scratch"]["doc__result"]      # the work
state["scratch"]["doc__review"]      # {"verdict": "approve" | "revise", "feedback": ...}
state["scratch"].get("doc__exhausted")  # True when the rounds ran out — NOT an approval
```

The fragment has two named exits, `per.exits["approved"]` and `per.exits["exhausted"]`, each a `(node id, condition)` pair — wire the exhausted one into an `approval_gate` to let a human decide.

Its building blocks are ordinary node kinds you can use directly: `llm` (one LLM call; text or structured output; a structured field can be the routing signal) and `loop_limit` (`again` until `max_rounds`, then `exhausted`).

Run the offline demo: `uv run python examples/plan_execute_review_pattern.py`
````

- [ ] **Step 8: Run everything**

Run: `uv run pytest -q && uv run ruff check src tests examples && uv run python examples/plan_execute_review_pattern.py && grep -rn lgtools src; true`
Expected: all pass (158 + 15 + 1 + 1 = 175); ruff clean; the demo prints `{'approved': True, 'rounds': 1, 'result': ...}`; grep prints nothing.

- [ ] **Step 9: Commit**

```bash
git add src/lgkit/patterns examples/plan_execute_review_pattern.py README.md tests
git commit -m "feat(lgkit): plan_execute_review pattern; Fragment gets named exits

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 5: lgtools — editor support for `llm` and `loop_limit`

**Files (all under `$LGT`):**
- Create: `backend/tests/test_plan_execute_review_dry_run.py`
- Modify: `frontend/src/lib/api.ts` (`NodeKind`), `frontend/src/lib/workflow-flow.ts` (`NODE_COLORS`), `frontend/src/routes/workflows/Inspector.svelte` (two panels), `frontend/src/routes/workflows/Inspector.svelte.test.ts`, `frontend/src/lib/workflow-flow.test.ts`

**Interfaces:**
- Consumes: lgkit kinds `llm`, `loop_limit` (registered at startup by `register_all()` → `lgkit.nodes.ensure_registered()`), `lgkit.patterns.plan_execute_review`.
- Produces: the frontend recognises both kinds (union member, colour, Inspector panel), so `backend/tests/test_node_kind_contract_with_frontend.py` passes again.

Context for the implementer: lgtools imports lgkit as an editable path dependency (`../../langgraph-agent`), so the lgkit branch from Tasks 1–4 must be checked out there. After Tasks 2–3 the contract test `backend/tests/test_node_kind_contract_with_frontend.py` FAILS until this task lands — that failure is this task's RED for the frontend part.

- [ ] **Step 1: Branch and confirm the RED**

```bash
cd $LGT && git status --short   # must be empty
git switch -c feat/llm-node-editor
cd backend && uv run pytest tests/test_node_kind_contract_with_frontend.py -q
```
Expected: FAIL naming `llm` and `loop_limit` (no `NodeKind` member / colour / Inspector panel).

- [ ] **Step 2: Backend integration test (dry-run through lgtools' own machinery)** — `backend/tests/test_plan_execute_review_dry_run.py`

```python
"""plan_execute_review runs as an lgtools dry-run test case: every llm node is
scripted through mock_scripts and no real model can be built."""

import json

import pytest
from lgkit.patterns import plan_execute_review

from lgtools.kernel.schemas import TestCaseSpec
from lgtools.kernel.testing import run_test_cases
from lgtools.templates.primitives import register_all


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real LLM built")

    monkeypatch.setattr("lgkit.llm.build_llm", boom)
    monkeypatch.setattr("lgtools.llm.build_llm", boom)
    register_all()


def _spec(max_rounds=3):
    return plan_execute_review(
        "doc", planner="Plan.", executor="Write.", reviewer="Check.", max_rounds=max_rounds
    ).to_workflow(name="doc")


def _state_assert(path, value):
    return {"kind": "state", "path": path, "op": "equals", "value": value}


async def test_revise_then_approve_from_mock_scripts():
    case = TestCaseSpec(
        id="c1",
        task="document it",
        mock_scripts={
            "doc__plan": ["the plan"],
            "doc__execute": ["draft one", "draft two"],
            "doc__review": [
                json.dumps({"verdict": "revise", "feedback": "more"}),
                json.dumps({"verdict": "approve", "feedback": ""}),
            ],
        },
        expect_status="done",
        assertions=[
            _state_assert("scratch.doc__result", "draft two"),
            _state_assert("scratch.doc__round", "1"),
        ],
    )
    (result,) = await run_test_cases(_spec(), [case])
    assert result["passed"], result


async def test_exhausted_is_reported_not_approved():
    revise = json.dumps({"verdict": "revise", "feedback": "no"})
    case = TestCaseSpec(
        id="c2",
        task="document it",
        mock_scripts={"doc__plan": ["p"], "doc__execute": ["w1", "w2"], "doc__review": [revise, revise]},
        expect_status="done",
        assertions=[
            _state_assert("scratch.doc__exhausted", "True"),
            _state_assert("scratch.doc__review.verdict", "revise"),
        ],
    )
    (result,) = await run_test_cases(_spec(max_rounds=2), [case])
    assert result["passed"], result


async def test_unscripted_reviewer_fails_the_case_instead_of_calling_a_model():
    case = TestCaseSpec(
        id="c3", task="document it",
        mock_scripts={"doc__plan": ["p"], "doc__execute": ["w"]},
        expect_status="done",
    )
    (result,) = await run_test_cases(_spec(), [case])
    assert not result["passed"]
    assert "doc__review" in (result["error"] or "")
```

Run: `uv run pytest tests/test_plan_execute_review_dry_run.py -q`
Expected: PASS (3 passed) — lgkit already implements the behaviour; this pins it through lgtools' `DryRunState` (its `next_mock()` gives each visit of a node the next scripted reply, which is what makes the revise loop scriptable). If a test fails because of how `FakeModel` scripts replies, fix the TEST's script shape (not lgkit) and note it in the report; if it reveals a real lgkit defect, stop and report BLOCKED.

- [ ] **Step 3: Kind maps**

- `frontend/src/lib/api.ts`: add `| 'llm'` and `| 'loop_limit'` to the `NodeKind` union (next to `'advisor'` / `'approval'`).
- `frontend/src/lib/workflow-flow.ts`: add to `NODE_COLORS` `llm: '#0891b2'` and `loop_limit: '#64748b'` — if either value is already used by another kind, pick a close unused shade.
- `frontend/src/lib/workflow-flow.test.ts`: extend the existing colour test (the one that lists `advisor`/`approval`) to cover both new kinds.

- [ ] **Step 4: Inspector panels** — `frontend/src/routes/workflows/Inspector.svelte`, following the existing `{#if selectedNode.data.kind === '…'}` blocks and reusing the existing helpers (`inputValue`, `textareaValue`, `noteNumberEdit`/`commitNumberParam` wrapped in `{#key selectedNode.id}` for number boxes, `commitLlmParam` for provider/model). Write component tests first in `Inspector.svelte.test.ts` using the existing `renderGateKind`-style helper pattern (generalise it or add a sibling helper).

`llm` panel (params object `lp`):
- `message` — Textarea, placeholder `{task}`; `oninput` → `onParamChange('message', value)`.
- `result_key` — Input (optional; hint: defaults to the node id).
- `output_fields` — a small rows editor: each row has `name` (Input), `type` (Select: string, number, boolean, object, array), `required` (checkbox, default checked), remove button; a "+ field" button appends `{name: '', type: 'string'}`. Writes the whole list through `onParamChange('output_fields', next)`; removing the last row removes the key via `onParamRemove('output_fields')`.
- `signal_field` — Select listing the current `output_fields` names plus a "none" placeholder. Choosing none removes both `signal_field` and `signal_values` (`onParamRemove`).
- `signal_values` — shown only when `signal_field` is set: the shared `choicesEditor` snippet (the same add/remove list hitl/approval use), with an inline destructive hint when the list is empty ("required when a signal field is set").
- `llm` provider + model — the same controls and `commitLlmParam` the advisor panel uses (blank → key removed).
- `max_attempts` — number box, min 1 max 10 integer, default 2, same commit rules as the advisor's.
- The node prompt box is already shown for this kind via the palette's `uses_node_prompt`.

`loop_limit` panel (params object `ll`):
- `max_rounds` — number box, min 1 max 100 integer, default 3; empty must NOT remove the key silently without warning: show an inline destructive hint "max_rounds is required" when unset.
- `counter_key`, `exhausted_key` — optional Inputs (hint: default `<node id>__count` / `<node id>__exhausted`); blank → `onParamRemove`.
- A muted one-line explanation: emits `again` until max_rounds, then `exhausted`.

Required component tests (add to `Inspector.svelte.test.ts`):
1. llm: typing in message calls `onParamChange('message', …)`.
2. llm: "+ field" then naming a field writes `output_fields` with that row; removing the only row calls `onParamRemove('output_fields')`.
3. llm: `signal_field` Select offers exactly the field names; selecting the placeholder calls `onParamRemove` for both `signal_field` and `signal_values`.
4. llm: with `signal_field` set and no `signal_values`, the required hint is shown.
5. llm: blank provider + model removes the `llm` key.
6. loop_limit: changing `max_rounds` commits the clamped integer to the node's id; with `max_rounds` unset the required hint is shown.

- [ ] **Step 5: Verify everything**

```bash
cd $LGT/frontend && npm run check && npm test -- --reporter=dot
cd ../backend && uv run pytest -q && uv run ruff check . && uv run ruff format --check .
```
Expected: `npm run check` 0 errors / 0 warnings; frontend tests pass (374 + the new ones); backend passes (1140 + 3 new, 4 skipped) including `test_node_kind_contract_with_frontend.py`; ruff clean.

- [ ] **Step 6: Commit** (logical commits are fine: backend test, kind maps, each panel)

```bash
cd $LGT && git add backend/tests/test_plan_execute_review_dry_run.py frontend
git commit -m "feat(editor): llm and loop_limit node kinds — colours, Inspector panels, dry-run test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

## Follow-ups (not in this plan)

- A one-click "insert plan → execute → review" action in the lgtools editor (today: build the fragment in code, or drag the four nodes).
- Patterns still to come: keyword→LLM router, fan-out → synthesize (committee) — both can be built from the `llm` node.
- `advisor` still asks `model_for()` in a dry-run, so an advisor revisited in a loop replays its first scripted reply; move it to the same per-visit `next_mock` path as `llm`.
- An `llm` node without `signal_field` leaves the previous `signal` in state; routing on it is a spec mistake the validator does not flag yet.
