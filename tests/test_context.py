from lgkit.context import (
    Budget,
    RunContext,
    clear_ctx,
    current_ctx,
    get_resolved_agent,
    get_resolved_workflow,
    use_ctx,
)


def test_budget():
    b = Budget(max_iterations=10, recursion_limit=50)
    assert b.max_iterations == 10


def test_run_context():
    from pathlib import Path

    ctx = RunContext(
        session_id="test",
        workspace=Path("/tmp"),
        llm=None,
        budget=Budget(max_iterations=10, recursion_limit=50),
    )
    assert ctx.session_id == "test"
    assert ctx.dry_run is None


def test_ctx_contextvar():
    assert current_ctx() is None
    from pathlib import Path

    ctx = RunContext(
        session_id="r1",
        workspace=Path("/tmp"),
        llm=None,
        budget=Budget(max_iterations=5, recursion_limit=20),
    )
    token = use_ctx(ctx)
    assert current_ctx() is ctx
    clear_ctx(token)
    assert current_ctx() is None


def test_get_resolved_workflow_no_resolved():
    assert get_resolved_workflow("foo") is None


def test_get_resolved_agent_no_resolved():
    assert get_resolved_agent("bar") is None
