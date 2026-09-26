"""Tests for EventScript/ScriptRef schemas and WorkflowSpec.validate_events."""

import pytest

from lgkit.spec import EventScript, NodeSpec, ScriptRef, WorkflowSpec


def test_script_ref_requires_exactly_one_of_code_or_file():
    with pytest.raises(Exception):
        ScriptRef()
    with pytest.raises(Exception):
        ScriptRef(code="x", file="y.py")
    assert ScriptRef(code="x").code == "x"
    assert ScriptRef(file="y.py").file == "y.py"


def test_event_script_id_must_be_identifier():
    with pytest.raises(Exception):
        EventScript(id="has space", on="node.on_enter", script=ScriptRef(code="x"))
    with pytest.raises(Exception):
        EventScript(id="", on="node.on_enter", script=ScriptRef(code="x"))


def test_event_script_on_must_be_nonempty():
    with pytest.raises(Exception):
        EventScript(id="e1", on="   ", script=ScriptRef(code="x"))


def test_validate_events_rejects_bad_node_lifecycle_type():
    n = NodeSpec(
        id="n",
        kind="custom",
        events=[EventScript(id="e1", on="node.bad", script=ScriptRef(code="x"))],
    )
    spec = WorkflowSpec(nodes=[n], entry="n")
    errs = spec.validate_events()
    assert any("node lifecycle" in e for e in errs)


def test_validate_events_rejects_workflow_hook_wrong_scope():
    es = EventScript(id="w1", on="workflow.on_start", scope="*", script=ScriptRef(code="x"))
    spec = WorkflowSpec(nodes=[NodeSpec(id="n", kind="custom")], entry="n", events=[es])
    errs = spec.validate_events()
    assert any("scope='workflow'" in e for e in errs)


def test_validate_events_rejects_unknown_node_scope():
    es = EventScript(id="w1", on="custom.evt", scope="ghost", script=ScriptRef(code="x"))
    spec = WorkflowSpec(nodes=[NodeSpec(id="n", kind="custom")], entry="n", events=[es])
    errs = spec.validate_events()
    assert any("not a known node id" in e for e in errs)


def test_validate_events_rejects_duplicate_node_event_id():
    n = NodeSpec(
        id="n",
        kind="custom",
        events=[
            EventScript(id="dup", on="node.on_enter", script=ScriptRef(code="x")),
            EventScript(id="dup", on="node.on_exit", script=ScriptRef(code="y")),
        ],
    )
    spec = WorkflowSpec(nodes=[n], entry="n")
    assert any("duplicate event id" in e for e in spec.validate_events())


def test_validate_events_accepts_valid_config():
    n = NodeSpec(
        id="n",
        kind="custom",
        events=[EventScript(id="e1", on="node.on_enter", script=ScriptRef(code="x"))],
    )
    es = EventScript(id="w1", on="workflow.on_start", scope="workflow", script=ScriptRef(code="x"))
    sub = EventScript(id="s1", on="approval.requested", scope="*", script=ScriptRef(code="z"))
    spec = WorkflowSpec(nodes=[n], entry="n", events=[es, sub])
    assert spec.validate_events() == []
