"""Advisor dry-run safety.

In a dry-run the advisor must use the scripted fake model from
``ctx.dry_run.model_for(...)`` and NEVER build a real LLM, no matter what
per-gate ``llm`` spec or ``ctx.llm`` would otherwise win. A dry-run bound in
the ambient context that does not reach the node as ``ctx`` means the ctx
plumbing regressed — the advisor must refuse to build a real model rather
than silently spending API money.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import lgkit.llm
from lgkit.builder import build_graph
from lgkit.context import Budget, RunContext, clear_ctx, use_ctx
from lgkit.nodes import advisor
from lgkit.nodes.advisor import advice_key
from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec

CHOICES = ["approve", "revise"]
GOOD_REPLY = '{"choice":"approve","confidence":0.9,"reason":"ok"}'
GOOD = {"choice": "approve", "confidence": 0.9, "reason": "ok"}


class _FakeModel:
    """with_structured_output() returns self; invoke() returns an AIMessage."""

    def __init__(self, reply: str):
        self._reply = reply

    def with_structured_output(self, schema, **_kwargs):  # noqa: ARG002
        return self

    def invoke(self, _messages, **_kwargs):
        return AIMessage(content=self._reply)


class _FakeDryRun:
    """Records the node_id model_for() was called with."""

    def __init__(self, reply: str = "{}"):
        self.reply = reply
        self.asked: list[str] = []

    def model_for(self, node_id, spec=None):  # noqa: ARG002
        self.asked.append(node_id)
        return _FakeModel(self.reply)


def _no_real_llm(monkeypatch, calls: list | None = None):
    """Any real build_llm call fails the test."""

    def _boom(*args, **kwargs):
        if calls is not None:
            calls.append(1)
        raise AssertionError("real LLM built")

    monkeypatch.setattr(lgkit.llm, "build_llm", _boom)


def _ctx(fake) -> RunContext:
    return RunContext(
        session_id="t", workspace=Path("."), llm=None, budget=Budget(10, 50), dry_run=fake
    )


def _spec() -> WorkflowSpec:
    adv = NodeSpec(
        id="g__advisor",
        kind="advisor",
        params={"gate_id": "g", "message": "Publish?", "choices": CHOICES, "criteria": "x"},
    )
    gate = NodeSpec(
        id="g",
        kind="approval",
        params={
            "message": "Publish?",
            "choices": CHOICES,
            "approver": "agent",
            "min_confidence": 0.7,
            "advice_key": advice_key("g"),
        },
    )
    edges = [EdgeSpec(source="g__advisor", target="g")]
    edges += [EdgeSpec(source="g", target="END", condition=c) for c in CHOICES]
    return WorkflowSpec(nodes=[adv, gate], edges=edges, entry="g__advisor")


def _run_graph(fake, thread="t"):
    app = build_graph(_spec(), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": thread}}
    token = use_ctx(_ctx(fake))
    try:
        out = app.invoke({"task": "t"}, cfg)
    finally:
        clear_ctx(token)
    return app, cfg, out


def test_dry_run_wins_over_gate_llm_and_ctx_llm(monkeypatch):
    """Precedence (a): ctx.dry_run beats a per-gate llm spec and ctx.llm;
    model_for is called with the advisor node id."""
    _no_real_llm(monkeypatch)
    fake = _FakeDryRun(GOOD_REPLY)
    ctx = RunContext(
        session_id="t",
        workspace=Path("."),
        llm=object(),  # would win under the old precedence — must be ignored
        budget=Budget(10, 50),
        dry_run=fake,
    )
    params = {
        "gate_id": "g",
        "message": "Publish?",
        "choices": CHOICES,
        "criteria": "x",
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "__node_id": "g__advisor",
    }
    out = advisor.run({}, params, None, ctx)
    assert fake.asked == ["g__advisor"]
    assert out["scratch"][advice_key("g")] == GOOD


def test_agent_decides_from_scripted_reply_without_interrupt(monkeypatch):
    """A confident scripted approve lets the agent decide; no real LLM."""
    _no_real_llm(monkeypatch)
    fake = _FakeDryRun(GOOD_REPLY)
    app, cfg, out = _run_graph(fake)
    assert fake.asked == ["g__advisor"]
    assert not app.get_state(cfg).interrupts
    assert out["signal"] == "approve"
    assert out["scratch"]["g"] == {**out["scratch"]["g"], "by": "agent"}
    assert out["scratch"]["g"]["advice"] == GOOD


def test_default_reply_records_error_and_asks_human(monkeypatch):
    """The FakeModel default reply "{}" is unusable advice: recorded as an
    error, the gate interrupts and asks a human."""
    _no_real_llm(monkeypatch)
    fake = _FakeDryRun()  # default reply "{}"
    app, cfg, out = _run_graph(fake)
    assert fake.asked == ["g__advisor"] * 2  # retried once, then recorded
    snap = app.get_state(cfg)
    assert snap.interrupts, "gate must pause for a human"
    ask = snap.interrupts[0].value
    assert ask["advice"] is None
    assert ask["advice_error"]


def test_context_lost_refuses_real_model(monkeypatch):
    """A dry-run bound in the ambient context that did NOT reach the node as
    ctx: the advisor records an error naming the lost dry-run and never
    calls build_llm."""
    calls: list = []
    _no_real_llm(monkeypatch, calls)
    token = use_ctx(_ctx(_FakeDryRun()))
    try:
        params = {"gate_id": "g", "message": "Publish?", "choices": CHOICES, "criteria": "x"}
        with pytest.raises(RuntimeError, match="dry-run context lost"):
            advisor._llm_for(params, None)
        out = advisor.run({}, params, None, None)
    finally:
        clear_ctx(token)
    rec = out["scratch"][advice_key("g")]
    assert "dry-run context lost" in rec["error"]
    assert not calls  # build_llm was never reached
