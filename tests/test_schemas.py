import pytest

from lgkit.spec import (
    AgentOutput,
    AgentSpec,
    EdgeSpec,
    IOField,
    NodeSpec,
    WorkflowSpec,
    namespace_from_scope,
)


def test_node_spec_valid():
    n = NodeSpec(id="planner", kind="agent")
    assert n.id == "planner"
    assert n.kind == "agent"
    assert n.params == {}


def test_node_spec_rejects_bad_id():
    with pytest.raises(Exception):
        NodeSpec(id="my node", kind="agent")
    with pytest.raises(Exception):
        NodeSpec(id="my-node", kind="agent")


def test_workflow_requires_nodes():
    with pytest.raises(Exception):
        WorkflowSpec(nodes=[], entry="x")


def test_workflow_entry_must_exist():
    with pytest.raises(Exception):
        WorkflowSpec(nodes=[NodeSpec(id="a", kind="agent")], entry="b")


def test_workflow_unique_ids():
    with pytest.raises(Exception):
        WorkflowSpec(
            nodes=[NodeSpec(id="a", kind="agent"), NodeSpec(id="a", kind="agent")],
            entry="a",
        )


def test_agent_spec_defaults():
    a = AgentSpec(id="coder")
    assert a.max_tool_turns == 8
    assert a.input_template == "{task}"
    assert a.memory_scope == ["global"]


def test_agent_output_rejects_empty_signal_field():
    with pytest.raises(Exception):
        AgentOutput(signal_field="", values=["a"], default="a")


def test_agent_output_strips_whitespace():
    o = AgentOutput(signal_field="  decision  ", values=[" yes ", "no"], default="yes")
    assert o.signal_field == "decision"
    assert o.values == ["yes", "no"]


def test_namespace_from_scope():
    assert namespace_from_scope(["global"]) == ("global",)
    assert namespace_from_scope(["app", "users"]) == ("app", "users")


def test_namespace_rejects_langgraph():
    with pytest.raises(ValueError):
        namespace_from_scope(["langgraph"])


def test_namespace_rejects_dots():
    with pytest.raises(ValueError):
        namespace_from_scope(["a.b"])


def test_validate_edges_empty_ok():
    spec = WorkflowSpec(
        nodes=[NodeSpec(id="a", kind="agent", params={"agent": "x"})],
        edges=[],
        entry="a",
    )
    assert spec.validate_edges() == []


def test_validate_edges_bad_source():
    spec = WorkflowSpec(
        nodes=[NodeSpec(id="a", kind="agent", params={"agent": "x"})],
        edges=[EdgeSpec(source="b", target="a")],
        entry="a",
    )
    errors = spec.validate_edges()
    assert any("b" in e for e in errors)


def test_validate_inputs_required_missing():
    spec = WorkflowSpec(
        nodes=[NodeSpec(id="a", kind="agent", params={"agent": "x"})],
        entry="a",
        inputs=[IOField(name="query", type="string", required=True)],
    )
    with pytest.raises(ValueError, match="query"):
        spec.validate_inputs({})


def test_validate_inputs_optional_default():
    spec = WorkflowSpec(
        nodes=[NodeSpec(id="a", kind="agent", params={"agent": "x"})],
        entry="a",
        inputs=[IOField(name="count", type="number", required=False, default=5)],
    )
    out = spec.validate_inputs({})
    assert out["count"] == 5


def test_a_stored_spec_carrying_exit_conditions_still_loads():
    """`exit_conditions` was removed because nothing ever read it — every
    template set it, the editor round-tripped it, and no code path consulted it.
    Specs saved before the removal still carry the key, so parsing must ignore
    it rather than reject the workflow someone saved months ago.
    """
    spec = WorkflowSpec.model_validate(
        {
            "nodes": [{"id": "a", "kind": "script", "params": {"code": "pass"}}],
            "entry": "a",
            "exit_conditions": ["done", "failed", "max_iterations"],
        }
    )
    assert spec.entry == "a"
    assert not hasattr(spec, "exit_conditions")
