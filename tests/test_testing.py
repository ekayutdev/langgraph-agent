import pytest
from pydantic import BaseModel

from lgkit.context import current_ctx
from lgkit.testing import ScriptedLLM, using_llm


class Out(BaseModel):
    x: int


def test_scripted_llm_returns_replies_in_order_and_counts_calls():
    llm = ScriptedLLM([{"x": 1}, {"x": 2}])
    s = llm.with_structured_output(Out)
    assert s.invoke([]).x == 1
    assert s.invoke([]).x == 2
    assert llm.calls == 2


def test_scripted_llm_raises_scripted_exceptions_and_when_empty():
    llm = ScriptedLLM([RuntimeError("boom")])
    s = llm.with_structured_output(Out)
    with pytest.raises(RuntimeError, match="boom"):
        s.invoke([])
    with pytest.raises(RuntimeError, match="no replies left"):
        s.invoke([])


def test_using_llm_sets_and_clears_context():
    llm = ScriptedLLM([])
    with using_llm(llm):
        assert current_ctx().llm is llm
    assert current_ctx() is None


def test_scripted_llm_plain_invoke_returns_ai_message_and_records_messages():
    llm = ScriptedLLM(["hello", {"a": 1}, RuntimeError("boom")])
    first = llm.invoke(["m1"])
    assert first.content == "hello"
    assert llm.invoke(["m2"]).content == '{"a": 1}'
    with pytest.raises(RuntimeError, match="boom"):
        llm.invoke(["m3"])
    assert llm.calls == 3
    assert llm.messages == [["m1"], ["m2"], ["m3"]]


def test_scripted_llm_structured_calls_are_recorded_too():
    llm = ScriptedLLM([{"x": 1}])
    llm.with_structured_output(Out).invoke(["q"])
    assert llm.messages == [["q"]]
