import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from lgkit.builder import GraphBuildError, build_graph
from lgkit.nodes.advisor import advice_key
from lgkit.spec import CacheConfig, EdgeSpec, NodePolicy, NodeSpec, WorkflowSpec
from lgkit.testing import ScriptedLLM, using_llm

CHOICES = ["approve", "revise"]


def _spec(approver: str, with_advisor: bool, min_confidence: float = 0.7) -> WorkflowSpec:
    nodes = []
    edges = []
    gate_params = {
        "message": "Publish {scratch.draft}?",
        "choices": CHOICES,
        "approver": approver,
        "min_confidence": min_confidence,
    }
    if with_advisor:
        nodes.append(
            NodeSpec(
                id="g__advisor",
                kind="advisor",
                params={
                    "gate_id": "g",
                    "message": "Publish {scratch.draft}?",
                    "choices": CHOICES,
                    "criteria": "complete and polite",
                },
            )
        )
        edges.append(EdgeSpec(source="g__advisor", target="g"))
        gate_params["advice_key"] = advice_key("g")
    nodes.append(NodeSpec(id="g", kind="approval", params=gate_params))
    edges += [EdgeSpec(source="g", target="END", condition=c) for c in CHOICES]
    return WorkflowSpec(nodes=nodes, edges=edges, entry=nodes[0].id)


def _run(spec, llm, thread="t"):
    app = build_graph(spec, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": thread}}
    with using_llm(llm):
        out = app.invoke({"task": "t", "scratch": {"draft": "notes"}}, cfg)
    return app, cfg, out


def _ask(app, cfg):
    snap = app.get_state(cfg)
    return snap.interrupts[0].value if snap.interrupts else None


GOOD = {"choice": "approve", "confidence": 0.9, "reason": "fine"}


def test_human_mode_interrupts_without_advice():
    app, cfg, _ = _run(_spec("human", False), ScriptedLLM([]))
    assert _ask(app, cfg) == {
        "node": "g",
        "message": "Publish notes?",
        "choices": CHOICES,
        "advice": None,
        "advice_error": None,
    }
    out = app.invoke(Command(resume={"choice": "revise", "comment": "typo"}), cfg)
    assert out["signal"] == "revise"
    assert out["scratch"]["g"] == {"choice": "revise", "comment": "typo", "by": "human", "advice": None}


def test_human_mode_accepts_bare_string_resume():
    app, cfg, _ = _run(_spec("human", False), ScriptedLLM([]))
    out = app.invoke(Command(resume="approve"), cfg)
    assert out["scratch"]["g"]["choice"] == "approve"
    assert out["scratch"]["g"]["comment"] == ""


def test_human_mode_rejects_unknown_choice():
    app, cfg, _ = _run(_spec("human", False), ScriptedLLM([]))
    with pytest.raises(ValueError, match="not in"):
        app.invoke(Command(resume={"choice": "maybe"}), cfg)


def test_human_with_advice_shows_advice_and_calls_llm_once_across_resume():
    llm = ScriptedLLM([GOOD])
    app, cfg, _ = _run(_spec("human", True), llm)
    assert _ask(app, cfg)["advice"] == GOOD
    with using_llm(llm):
        out = app.invoke(Command(resume={"choice": "approve"}), cfg)
    assert llm.calls == 1  # regression: advisor must not re-run on resume
    assert out["scratch"]["g"]["by"] == "human"
    assert out["scratch"]["g"]["advice"] == GOOD


def test_agent_confident_decides_without_interrupt():
    app, cfg, out = _run(_spec("agent", True), ScriptedLLM([GOOD]))
    assert _ask(app, cfg) is None
    assert out["signal"] == "approve"
    assert out["scratch"]["g"] == {"choice": "approve", "comment": "fine", "by": "agent", "advice": GOOD}


def test_agent_at_exact_threshold_decides():
    reply = {"choice": "approve", "confidence": 0.7, "reason": "ok"}
    app, cfg, out = _run(_spec("agent", True, min_confidence=0.7), ScriptedLLM([reply]))
    assert _ask(app, cfg) is None
    assert out["scratch"]["g"]["by"] == "agent"


@pytest.mark.parametrize(
    "reply",
    [
        {"choice": "approve", "confidence": 0.3, "reason": "unsure"},
        {"choice": "escalate", "confidence": 0.9, "reason": "policy question"},
    ],
)
def test_agent_low_confidence_or_escalate_asks_human(reply):
    app, cfg, _ = _run(_spec("agent", True), ScriptedLLM([reply]))
    ask = _ask(app, cfg)
    assert ask["advice"] == reply
    out = app.invoke(Command(resume={"choice": "revise"}), cfg)
    assert out["scratch"]["g"]["by"] == "human"


def test_advisor_failure_is_retried_then_escalated():
    llm = ScriptedLLM([RuntimeError("timeout"), RuntimeError("timeout again")])
    app, cfg, _ = _run(_spec("agent", True), llm)
    ask = _ask(app, cfg)
    assert llm.calls == 2
    assert ask["advice"] is None
    assert ask["advice_error"] == "RuntimeError: timeout again"


def test_advisor_invalid_choice_counts_as_failure():
    bad = {"choice": "maybe", "confidence": 0.99, "reason": "?"}
    app, cfg, _ = _run(_spec("agent", True), ScriptedLLM([bad, bad]))
    ask = _ask(app, cfg)
    assert ask["advice"] is None
    assert ask["advice_error"].startswith("ValidationError")


def test_advisor_retry_recovers_on_second_attempt():
    app, cfg, out = _run(_spec("agent", True), ScriptedLLM([RuntimeError("blip"), GOOD]))
    assert _ask(app, cfg) is None
    assert out["scratch"]["g"]["by"] == "agent"


def test_approval_requires_checkpointer():
    with pytest.raises(GraphBuildError, match="checkpointer"):
        build_graph(_spec("human", False))


def test_approval_is_never_cached():
    spec = _spec("human", False)
    spec.nodes[-1].policy = NodePolicy(cache=CacheConfig())
    app = build_graph(spec, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "c"}}
    app.invoke({"task": "t"}, cfg)
    out = app.invoke(Command(resume={"choice": "approve"}), cfg)  # a cached gate crashes resume
    assert out["signal"] == "approve"
