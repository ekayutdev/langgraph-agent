"""FATAL node-param checks: advisor/approval must fail build_graph early.

An advisor with no gate_id, or a gate with blank/escalate-tainted choices,
died mid-run with a KeyError or silently mis-routed — long after the author
left the editor. These turn that into a GraphBuildError at build time, the
same way subworkflow/harness ref checks already do.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from lgkit.builder import GraphBuildError, build_graph
from lgkit.patterns import approval_gate
from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec


def _wf(*nodes: NodeSpec) -> WorkflowSpec:
    return WorkflowSpec(
        nodes=list(nodes),
        edges=[EdgeSpec(source=n.id, target="END") for n in nodes],
        entry=nodes[0].id,
    )


def test_advisor_with_blank_gate_id_is_fatal():
    spec = _wf(NodeSpec(id="a", kind="advisor", params={"gate_id": "", "choices": ["yes"], "criteria": "x"}))
    with pytest.raises(GraphBuildError, match="advisor node 'a'.*'gate_id'"):
        build_graph(spec, checkpointer=InMemorySaver())


def test_advisor_with_blank_choices_is_fatal():
    spec = _wf(NodeSpec(id="a", kind="advisor", params={"gate_id": "g", "choices": [], "criteria": "x"}))
    with pytest.raises(GraphBuildError, match="advisor node 'a'.*'choices'"):
        build_graph(spec, checkpointer=InMemorySaver())


def test_advisor_with_whitespace_gate_id_is_fatal():
    spec = _wf(NodeSpec(id="a", kind="advisor", params={"gate_id": "  ", "choices": ["yes"], "criteria": "x"}))
    with pytest.raises(GraphBuildError, match="advisor node 'a'.*'gate_id'"):
        build_graph(spec, checkpointer=InMemorySaver())


def test_approval_with_blank_choices_is_fatal():
    spec = _wf(NodeSpec(id="g", kind="approval", params={"message": "Ok?", "choices": []}))
    with pytest.raises(GraphBuildError, match="approval node 'g'.*'choices'"):
        build_graph(spec, checkpointer=InMemorySaver())


def test_approval_with_whitespace_choices_is_fatal():
    spec = _wf(NodeSpec(id="g", kind="approval", params={"message": "Ok?", "choices": ["  "]}))
    with pytest.raises(GraphBuildError, match="approval node 'g'.*'choices'"):
        build_graph(spec, checkpointer=InMemorySaver())


def test_approval_with_escalate_choice_is_fatal():
    spec = _wf(NodeSpec(id="g", kind="approval", params={"message": "Ok?", "choices": ["yes", "escalate"]}))
    with pytest.raises(GraphBuildError, match="approval node 'g'.*'escalate'"):
        build_graph(spec, checkpointer=InMemorySaver())


def test_approval_gate_fragments_still_build():
    for approver, advise in (("human", False), ("human", True), ("agent", False)):
        frag = approval_gate(
            "g",
            message="Ok?",
            choices=["yes", "no"],
            approver=approver,
            advise=advise,
            criteria="be sure" if advise or approver == "agent" else None,
        )
        app = build_graph(frag.to_workflow(), checkpointer=InMemorySaver())
        assert app is not None
