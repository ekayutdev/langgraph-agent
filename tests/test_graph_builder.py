"""Tests for the graph builder — WorkflowSpec → StateGraph compiler."""

import pytest

from lgkit.builder import GraphBuildError, build_graph
from lgkit.registry import NodeDef, register
from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec


def _stub_node_fn(state, params, prompt, ctx=None, resolved=None):
    """A minimal node function that just sets signal to 'done'."""
    return {"signal": "done", "scratch": {"result": "ok"}}


@pytest.fixture(autouse=True)
def register_stub_node():
    register(
        "custom", NodeDef(kind="custom", fn=_stub_node_fn, default_prompt="", description="stub")
    )
    register(
        "agent",
        NodeDef(kind="agent", fn=_stub_node_fn, default_prompt="", description="stub agent"),
    )


def test_build_simple_graph():
    spec = WorkflowSpec(
        nodes=[
            NodeSpec(id="start", kind="custom"),
            NodeSpec(id="end", kind="custom"),
        ],
        edges=[EdgeSpec(source="start", target="end")],
        entry="start",
    )
    app = build_graph(spec)
    assert app is not None


def test_build_invalid_entry_raises():
    with pytest.raises(Exception):
        build_graph(
            WorkflowSpec(
                nodes=[NodeSpec(id="a", kind="custom")],
                entry="b",
            )
        )


def test_build_conditional_edges():
    spec = WorkflowSpec(
        nodes=[
            NodeSpec(id="start", kind="custom"),
            NodeSpec(id="yes", kind="custom"),
            NodeSpec(id="no", kind="custom"),
        ],
        edges=[
            EdgeSpec(source="start", target="yes", condition="yes"),
            EdgeSpec(source="start", target="no", condition="no"),
        ],
        entry="start",
    )
    app = build_graph(spec)
    assert app is not None


def test_build_with_end_target():
    spec = WorkflowSpec(
        nodes=[NodeSpec(id="only", kind="custom")],
        edges=[EdgeSpec(source="only", target="END")],
        entry="only",
    )
    app = build_graph(spec)
    assert app is not None


def test_build_with_custom_state_fields():
    spec = WorkflowSpec(
        nodes=[NodeSpec(id="n", kind="custom")],
        edges=[EdgeSpec(source="n", target="END")],
        entry="n",
        state_fields=[
            {"name": "user_name", "type": "string", "reducer": "default"},
            {"name": "count", "type": "number", "reducer": "last_value"},
        ],
    )
    app = build_graph(spec)
    assert app is not None


def test_state_field_shadows_universal_raises():
    spec = WorkflowSpec(
        nodes=[NodeSpec(id="n", kind="custom")],
        edges=[EdgeSpec(source="n", target="END")],
        entry="n",
        state_fields=[{"name": "signal", "type": "string"}],
    )
    with pytest.raises(GraphBuildError, match="shadows"):
        build_graph(spec)


def test_build_graph_rejects_invalid_events():
    from lgkit.spec import EventScript, ScriptRef

    n = NodeSpec(
        id="n",
        kind="custom",
        events=[EventScript(id="e1", on="not.a.lifecycle", script=ScriptRef(code="x"))],
    )
    spec = WorkflowSpec(nodes=[n], entry="n")
    with pytest.raises(GraphBuildError, match="lifecycle"):
        build_graph(spec)


def test_build_graph_accepts_valid_events():
    from lgkit.spec import EventScript, ScriptRef

    n = NodeSpec(
        id="n",
        kind="custom",
        events=[EventScript(id="e1", on="node.on_enter", script=ScriptRef(code="x"))],
    )
    spec = WorkflowSpec(nodes=[n], entry="n")
    app = build_graph(spec)
    assert app is not None
