from lgkit.state import (
    AgentState,
    add_events,
    append_unique,
    get_reducer,
    intersection_reducer,
    merge_scratch,
    union_reducer,
)


def test_agent_state_has_core_fields():
    annotations = AgentState.__annotations__
    for field in ("task", "messages", "signal", "scratch", "events"):
        assert field in annotations, f"AgentState missing {field}"


def test_merge_scratch_right_wins():
    assert merge_scratch({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}


def test_merge_scratch_empty():
    assert merge_scratch({}, {"a": 1}) == {"a": 1}
    assert merge_scratch({"a": 1}, {}) == {"a": 1}


def test_add_events_concatenates():
    assert add_events([{"x": 1}], [{"y": 2}]) == [{"x": 1}, {"y": 2}]


def test_union_reducer_dedupes():
    assert union_reducer([1, 2], [2, 3]) == [1, 2, 3]


def test_intersection_reducer():
    assert intersection_reducer([1, 2, 3], [2, 3, 4]) == [2, 3]


def test_append_unique():
    assert append_unique([1, 2], [2, 3]) == [1, 2, 3]


def test_get_reducer():
    assert get_reducer("merge") is merge_scratch
    assert get_reducer("append") is add_events
    assert get_reducer("default") is None
    assert get_reducer("nonexistent") is None
