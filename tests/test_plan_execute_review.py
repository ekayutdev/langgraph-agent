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
    pattern_id = args.pop("id", None)
    return plan_execute_review(pattern_id if pattern_id is not None else "doc", **args)


def _run(wf, llm, state=None, **build_kw):
    app = build_graph(wf, **build_kw)
    with using_llm(llm):
        return app.invoke(state or {"task": "write docs"}, {"configurable": {"thread_id": "t"}})


def test_to_workflow_names_the_workflow_after_the_pattern_id():
    assert _per().to_workflow().name == "doc"
    assert _per().to_workflow(name="explicit").name == "explicit"


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


def test_revise_without_feedback_renders_blank_not_none_in_round_two():
    # A reviewer reply that omits feedback stores None; the round-two
    # executor message must not contain the literal "None".
    llm = ScriptedLLM(["plan", "draft one", {"verdict": "revise"}, "draft two", APPROVE])
    out = _run(_per().to_workflow(), llm)
    assert out["scratch"]["doc__result"] == "draft two"
    assert "None" not in llm.messages[3][-1].content


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
        ({"id": "  "}, "plan_execute_review: id"),
    ],
)
def test_invalid_arguments_fail_at_build_time(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _per(**kwargs)
