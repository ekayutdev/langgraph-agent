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


def test_signal_field_is_always_required_even_if_the_entry_says_optional():
    # "required": false on the signal field must be ignored: an optional
    # signal would let a reply that omits it validate and emit None — a
    # missing signal the router would act on.
    llm = ScriptedLLM([{}, {}])
    spec = _one(
        {
            "output_fields": [{"name": "verdict", "required": False}],
            "signal_field": "verdict",
            "signal_values": ["approve", "revise"],
        }
    )
    with pytest.raises(RuntimeError, match=r"llm node 'n' failed after 2 attempts"):
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
        super().__init__({})
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


def test_conditional_edges_require_a_signal_field():
    # Without signal_field the node emits no fresh signal, so the router
    # would act on whatever signal a previous node left behind.
    spec = _one({}, edges=[EdgeSpec(source="n", target="END", condition="go")])
    with pytest.raises(GraphBuildError, match="signal_field"):
        build_graph(spec)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"output_fields": [{"name": "a"}], "signal_field": "verdict", "signal_values": ["x"]}, "signal_field"),
        ({"output_fields": [{"name": "verdict"}], "signal_field": "verdict"}, "signal_values"),
        ({"signal_field": "verdict", "signal_values": ["x"]}, "signal_field"),
        ({"output_fields": [{"type": "string"}]}, "output_fields"),
        (
            {
                "output_fields": [{"name": "verdict"}, {"name": "feedback"}],
                "signal_field": "verdict",
                "signal_values": "approve",
            },
            "signal_values",
        ),
        (
            {
                "output_fields": [{"name": "verdict"}, {"name": "feedback"}],
                "signal_field": "verdict",
                "signal_values": [1, 2],
            },
            "signal_values",
        ),
        (
            {
                "output_fields": [{"name": "verdict"}, {"name": "feedback"}],
                "signal_field": "verdict",
                "signal_values": ["", "go"],
            },
            "signal_values",
        ),
        ({"output_fields": [{"name": "verdict", "type": "integer"}]}, "type"),
        (
            {
                "output_fields": [
                    {"name": "verdict", "type": "string"},
                    {"name": "verdict", "type": "string"},
                ]
            },
            "unique",
        ),
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
