import pytest

from lgkit.builder import GraphBuildError, build_graph
from lgkit.nodes import loop_limit
from lgkit.registry import NodeDef, register
from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec


def _call(scratch, **params):
    return loop_limit.run({"scratch": scratch}, {"__node_id": "lim", **params}, "")


def test_counts_and_says_again_until_the_limit():
    first = _call({}, max_rounds=3)
    assert first == {"signal": "again", "scratch": {"lim__count": 1}}
    second = _call({"lim__count": 1}, max_rounds=3)
    assert second == {"signal": "again", "scratch": {"lim__count": 2}}
    third = _call({"lim__count": 2}, max_rounds=3)
    assert third == {"signal": "exhausted", "scratch": {"lim__count": 3, "lim__exhausted": True}}


def test_max_rounds_one_is_exhausted_immediately():
    assert _call({}, max_rounds=1)["signal"] == "exhausted"


def test_custom_keys():
    out = _call({}, max_rounds=1, counter_key="round", exhausted_key="gave_up")
    assert out["scratch"] == {"round": 1, "gave_up": True}


@pytest.mark.parametrize("params", [{}, {"max_rounds": 0}, {"max_rounds": "many"}, {"max_rounds": 1.5}])
def test_bad_max_rounds_fails_at_build_time(params):
    spec = WorkflowSpec(nodes=[NodeSpec(id="lim", kind="loop_limit", params=params)], entry="lim")
    with pytest.raises(GraphBuildError, match="max_rounds"):
        build_graph(spec)


def _loop_spec(max_rounds: int) -> WorkflowSpec:
    return WorkflowSpec(
        nodes=[
            NodeSpec(id="work", kind="custom"),
            NodeSpec(id="lim", kind="loop_limit", params={"max_rounds": max_rounds}),
        ],
        edges=[
            EdgeSpec(source="work", target="lim"),
            EdgeSpec(source="lim", target="work", condition="again"),
            EdgeSpec(source="lim", target="END", condition="exhausted"),
        ],
        entry="work",
    )


def _count_work():
    ran = []
    register(
        "custom",
        NodeDef(kind="custom", fn=lambda s, p, pr, c=None, r=None: ran.append(1) or {},
                default_prompt="", description="work"),
    )
    return ran


def test_loop_runs_exactly_max_rounds_times():
    ran = _count_work()
    out = build_graph(_loop_spec(3)).invoke({"task": "t"})
    assert len(ran) == 3
    assert out["scratch"]["lim__exhausted"] is True and out["signal"] == "exhausted"


def test_global_iteration_budget_ends_the_run_instead_of_looping():
    # The builder puts an iteration guard after `lim` (a back-edge source).
    # Once the run-wide budget is spent the guard emits "max_iterations", which
    # is neither "again" nor "exhausted": the run must END, not take the first
    # edge ("again") forever.
    ran = _count_work()
    out = build_graph(_loop_spec(50)).invoke({"task": "t", "scratch": {"max_iterations": 2}})
    assert len(ran) == 2
    assert out["signal"] == "max_iterations"
    assert "lim__exhausted" not in out["scratch"]
