import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from lgkit.builder import build_graph
from lgkit.patterns import Fragment, approval_gate
from lgkit.registry import NodeDef, register
from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec
from lgkit.testing import ScriptedLLM, using_llm


def test_human_gate_is_a_single_node():
    frag = approval_gate("g", message="Ok?", choices=["yes", "no"])
    assert isinstance(frag, Fragment)
    assert [n.kind for n in frag.nodes] == ["approval"]
    assert frag.entry == frag.exit == "g"
    assert frag.edges == ()


def test_agent_gate_puts_advisor_first():
    frag = approval_gate(
        "g", message="Ok?", choices=["yes", "no"], approver="agent", criteria="be sure",
        llm={"provider": "openai", "model": "gpt-4o-mini"},
    )
    adv, gate = frag.nodes
    assert (adv.id, adv.kind, gate.kind) == ("g__advisor", "advisor", "approval")
    assert adv.params["llm"] == {"provider": "openai", "model": "gpt-4o-mini"}
    assert gate.params["advice_key"] == "g__advice"
    assert frag.edges == (EdgeSpec(source="g__advisor", target="g"),)
    assert (frag.entry, frag.exit) == ("g__advisor", "g")


def test_to_workflow_routes_every_choice_to_end():
    wf = approval_gate("g", message="Ok?", choices=["yes", "no"]).to_workflow(name="w")
    assert wf.name == "w" and wf.entry == "g"
    assert {(e.source, e.target, e.condition) for e in wf.edges} == {
        ("g", "END", "yes"),
        ("g", "END", "no"),
    }


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"choices": []}, "choices"),
        ({"choices": ["a", "a"]}, "unique"),
        ({"choices": ["a", "escalate"]}, "escalate"),
        ({"approver": "robot"}, "approver"),
        ({"approver": "agent"}, "criteria"),
        ({"advise": True}, "criteria"),
        ({"approver": "agent", "advise": True, "criteria": "x"}, "advise"),
        ({"min_confidence": 1.5}, "min_confidence"),
        ({"message": "Approve {draft"}, "message"),
        ({"message": "Approve {}"}, "message"),
        ({"message": "Approve {0}"}, "message"),
        ({"llm": {"provider": "openai"}}, "llm"),
    ],
)
def test_invalid_arguments_fail_at_build_time(kwargs, match):
    args = {"message": "Ok?", "choices": ["yes", "no"], **kwargs}
    with pytest.raises(ValueError, match=match):
        approval_gate("g", **args)


def test_end_to_end_human_gate():
    frag = approval_gate("g", message="Ship {scratch.v}?", choices=["yes", "no"])
    app = build_graph(frag.to_workflow(), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "e1"}}
    app.invoke({"task": "t", "scratch": {"v": "1.2"}}, cfg)
    assert app.get_state(cfg).interrupts[0].value["message"] == "Ship 1.2?"
    out = app.invoke(Command(resume={"choice": "yes"}), cfg)
    assert out["scratch"]["g"]["by"] == "human"


def _drafter(state, params, prompt, ctx=None, resolved=None):
    n = int((state.get("scratch") or {}).get("round", 0)) + 1
    return {"scratch": {"round": n, "draft": f"draft v{n}"}}


def test_max_iterations_on_approval_gate_ends_run_not_first_choice():
    """A guard-injected max_iterations signal must END the run, not fall
    through to the gate's first conditional edge (which would publish work
    the human chose to revise)."""
    ran: list[str] = []

    def _recorder(state, params, prompt, ctx=None, resolved=None):
        ran.append(params.get("__node_id"))
        return {"scratch": {"touched": params.get("__node_id")}}

    register("custom", NodeDef(kind="custom", fn=_recorder, default_prompt="", description="recorder"))
    gate = approval_gate("g", message="Publish?", choices=["approve", "revise"])
    wf = WorkflowSpec(
        nodes=[NodeSpec(id="draft", kind="custom"), *gate.nodes, NodeSpec(id="publish", kind="custom")],
        edges=[
            EdgeSpec(source="draft", target="g"),
            EdgeSpec(source="g", target="publish", condition="approve"),  # listed FIRST
            EdgeSpec(source="g", target="draft", condition="revise"),
            EdgeSpec(source="publish", target="END"),
        ],
        entry="draft",
    )
    app = build_graph(wf, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "max"}}
    app.invoke({"task": "t", "scratch": {"max_iterations": 1}}, cfg)
    app.invoke(Command(resume={"choice": "revise"}), cfg)
    assert "publish" not in ran
    assert not app.get_state(cfg).interrupts


def test_revise_loop_uses_fresh_advice_each_round():
    register("custom", NodeDef(kind="custom", fn=_drafter, default_prompt="", description="drafter"))
    frag = approval_gate(
        "g", message="{scratch.draft}", choices=["approve", "revise"],
        approver="agent", criteria="approve v2 only",
    )
    wf = WorkflowSpec(
        nodes=[NodeSpec(id="draft", kind="custom"), *frag.nodes],
        edges=[
            EdgeSpec(source="draft", target=frag.entry),
            *frag.edges,
            EdgeSpec(source="g", target="draft", condition="revise"),
            EdgeSpec(source="g", target="END", condition="approve"),
        ],
        entry="draft",
    )
    llm = ScriptedLLM(
        [
            {"choice": "revise", "confidence": 0.9, "reason": "v1 incomplete"},
            {"choice": "approve", "confidence": 0.9, "reason": "v2 good"},
        ]
    )
    app = build_graph(wf, checkpointer=InMemorySaver())
    with using_llm(llm):
        out = app.invoke({"task": "t", "scratch": {"max_iterations": 10}}, {"configurable": {"thread_id": "loop"}})
    assert llm.calls == 2
    assert out["scratch"]["round"] == 2
    assert out["scratch"]["g"]["comment"] == "v2 good"
    assert out["signal"] == "approve"
