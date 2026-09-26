import pytest


@pytest.fixture(autouse=True)
def _restore_node_registry():
    """Snapshot/restore the module-level node registry so kinds registered by
    one test never leak into another (mirrors lgtools' conftest)."""
    from lgkit import registry

    kinds = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(kinds)
