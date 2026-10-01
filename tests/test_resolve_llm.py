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
