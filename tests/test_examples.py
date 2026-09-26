import importlib.util
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plan_execute_review_loops_until_approved():
    mod = _load("plan_execute_review")
    out = mod.run("Create API documentation")
    assert out["approved"] is True
    assert out["attempts"] == 2


def test_plan_execute_review_stops_at_max_rounds():
    mod = _load("plan_execute_review")
    assert mod.route({"approved": False, "attempts": mod.MAX_ROUNDS}) == "end"
    assert mod.route({"approved": False, "attempts": 1}) == "executor"


def test_approval_demo_escalates_and_records_human(monkeypatch):
    mod = _load("approval_demo")
    monkeypatch.delenv("LGKIT_REAL_LLM", raising=False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "approve")
    result = mod.main()
    assert result["by"] == "human"
    assert result["choice"] == "approve"
    assert result["advice"]["confidence"] == 0.55
