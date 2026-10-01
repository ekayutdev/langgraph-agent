import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from lgkit.builder import build_graph
from lgkit.nodes._template import flat_state
from lgkit.registry import NodeDef, all_node_defs, register
from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec


def _gate_spec() -> WorkflowSpec:
    return WorkflowSpec(
        nodes=[
            NodeSpec(
                id="gate",
                kind="hitl",
                params={"message": "Ship {scratch.item}?", "choices": ["yes", "no"]},
            )
        ],
        edges=[
            EdgeSpec(source="gate", target="END", condition="yes"),
            EdgeSpec(source="gate", target="END", condition="no"),
        ],
        entry="gate",
    )


def test_flat_state_tolerates_missing_and_nested_keys():
    flat = flat_state({"task": "t", "scratch": {"a": {"b": "deep"}}})
    assert "{task}|{nope}|{scratch.a.b}|{scratch.zz}".format_map(flat) == "t||deep|"


def test_flat_state_tolerates_chained_access_on_missing_keys():
    flat = flat_state({"task": "t", "scratch": {"a": {"b": "x"}}})
    assert "[{scratch.nope.deeper}][{scratch.a.zz.more}][{scratch.a.b}]".format_map(flat) == "[][][x]"


def test_flat_state_renders_a_present_but_none_value_as_blank():
    # A reviewer reply with no feedback stores None; "None" leaking into a
    # message template reads like feedback, so it must render like a missing
    # key — on nested attribute paths and top-level keys alike.
    flat = flat_state({"x": None, "scratch": {"r": {"feedback": None}}})
    assert "[{scratch.r.feedback}][{x}]".format_map(flat) == "[][]"


def test_hitl_pauses_then_resumes_with_choice():
    app = build_graph(_gate_spec(), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "h1"}}
    app.invoke({"task": "t", "scratch": {"item": "v2"}}, cfg)
    ask = app.get_state(cfg).interrupts[0].value
    assert ask == {"node": "gate", "message": "Ship v2?", "choices": ["yes", "no"]}

    out = app.invoke(Command(resume={"choice": "yes", "comment": "go"}), cfg)
    assert out["signal"] == "yes"
    assert out["scratch"]["gate"] == {"choice": "yes", "comment": "go"}


def test_hitl_rejects_unknown_choice():
    app = build_graph(_gate_spec(), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "h2"}}
    app.invoke({"task": "t"}, cfg)
    with pytest.raises(ValueError, match="not in"):
        app.invoke(Command(resume={"choice": "maybe"}), cfg)


def test_ensure_registered_never_overrides_an_app_kind():
    def app_hitl(state, params, prompt, ctx=None, resolved=None):
        return {}

    register("hitl", NodeDef(kind="hitl", fn=app_hitl, default_prompt="", description="app"))
    build_graph(_gate_spec(), checkpointer=InMemorySaver())
    assert all_node_defs()["hitl"].fn is app_hitl
