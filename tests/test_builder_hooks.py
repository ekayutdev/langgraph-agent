import pytest

from lgkit.builder import GraphBuildError, build_graph
from lgkit.hooks import BuildHooks
from lgkit.registry import NodeDef, register
from lgkit.spec import EventScript, NodeSpec, ScriptRef, WorkflowSpec

SEEN: dict = {}


def _probe(state, params, prompt, ctx=None, resolved=None):
    SEEN["resolved"] = resolved
    SEEN.setdefault("ran", []).append(params.get("__node_id"))
    return {"signal": "done"}


@pytest.fixture(autouse=True)
def _probe_kind():
    SEEN.clear()
    register("custom", NodeDef(kind="custom", fn=_probe, default_prompt="", description="probe"))


def _one_node(**node_kw) -> WorkflowSpec:
    return WorkflowSpec(nodes=[NodeSpec(id="n", kind="custom", **node_kw)], entry="n")


def test_default_hooks_skip_event_scripts():
    # A valid on_enter script would need lgtools' subprocess runtime; with the
    # default hooks it must simply not run.
    ev = EventScript(id="e1", on="node.on_enter", script=ScriptRef(code="raise SystemExit(1)"))
    app = build_graph(_one_node(events=[ev]))
    assert app.invoke({"task": "t"})["signal"] == "done"


def test_dispatch_hook_sees_lifecycle_events():
    calls: list[str] = []

    def dispatch(event_type, state, **kw):
        calls.append(event_type)
        return {}

    app = build_graph(_one_node(), hooks=BuildHooks(dispatch_event=dispatch))
    app.invoke({"task": "t"})
    assert calls == ["node.on_enter", "node.on_exit"]


def test_dispatch_hook_route_becomes_goto():
    register("custom", NodeDef(kind="custom", fn=_probe, default_prompt="", description="p"))
    spec = WorkflowSpec(
        nodes=[NodeSpec(id="a", kind="custom"), NodeSpec(id="b", kind="custom")],
        entry="a",
    )

    def dispatch(event_type, state, **kw):
        return {"route": "b"} if (event_type == "node.on_exit" and kw["node_id"] == "a") else {}

    app = build_graph(spec, hooks=BuildHooks(dispatch_event=dispatch))
    app.invoke({"task": "t"})  # would raise if goto 'b' were not honored as a node
    assert "b" in SEEN.get("ran", [])


def test_resolved_is_passed_to_node_fns():
    marker = object()
    build_graph(_one_node(), resolved=marker).invoke({"task": "t"})
    assert SEEN["resolved"] is marker


def test_workflow_resolver_drives_cycle_detection():
    looping = WorkflowSpec(
        nodes=[NodeSpec(id="s", kind="subworkflow", params={"ref": "a"})], entry="s"
    )
    hooks = BuildHooks(workflow_resolver=lambda ref: looping if ref == "a" else None)
    with pytest.raises(GraphBuildError, match="cycle"):
        build_graph(looping, hooks=hooks)
