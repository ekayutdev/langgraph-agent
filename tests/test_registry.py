"""The node registry, exercised without leaving anything behind.

``_REGISTRY`` is a module-level dict, so a kind registered here stays visible to
every later test — ``test_node_kind_contract_with_frontend`` had to filter on
``fn.__module__`` to avoid asserting against this file's throwaway ``test_kind``,
which is a workaround for a leak rather than a fix for one. The sibling leak in
test_tool_registry did break the suite under a different file order.
"""

import pytest

from lgkit.registry import NodeDef, all_node_defs, get_node_def, node_palette, register


def _register_probe() -> None:
    def fn(state, params, prompt, ctx=None, resolved=None):
        return {}

    register(
        "test_kind", NodeDef(kind="test_kind", fn=fn, default_prompt="hi", description="test node")
    )


def test_register_and_get():
    _register_probe()
    d = get_node_def("test_kind")
    assert d.kind == "test_kind"
    assert d.default_prompt == "hi"
    assert d.description == "test node"


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get_node_def("nonexistent_kind_xyz_123")


def test_all_node_defs():
    _register_probe()
    defs = all_node_defs()
    assert isinstance(defs, dict)
    assert "test_kind" in defs


def test_node_palette():
    palette = node_palette()
    assert isinstance(palette, list)
    assert all("kind" in p and "description" in p for p in palette)


def test_the_probe_kind_does_not_outlive_this_file():
    """Asserted directly, so the contract test's module filter stops being the
    only thing standing between this file and a false failure elsewhere."""
    assert "test_kind" not in all_node_defs()
