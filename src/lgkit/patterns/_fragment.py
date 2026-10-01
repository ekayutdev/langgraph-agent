"""Fragment — the spec data a pattern returns."""

from __future__ import annotations

from dataclasses import dataclass, field

from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec


@dataclass(frozen=True)
class Fragment:
    """Nodes + internal edges of a pattern.

    Wire an edge into ``entry``. Leave through ``exits``: each named exit is a
    ``(node id, condition)`` pair — add ``EdgeSpec(source=node, target=...,
    condition=condition)``. ``exit``/``choices`` are the single-gate shorthand
    approval_gate has always had.
    """

    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    entry: str
    exit: str
    choices: tuple[str, ...]
    exits: dict[str, tuple[str, str]] = field(default_factory=dict)
    name: str = "workflow"

    def to_workflow(self, name: str | None = None) -> WorkflowSpec:
        """A standalone workflow: every exit routes to END. ``name`` wins over
        the pattern's own ``name``."""
        if self.exits:
            ends = [EdgeSpec(source=n, target="END", condition=c) for n, c in self.exits.values()]
        else:
            ends = [EdgeSpec(source=self.exit, target="END", condition=c) for c in self.choices]
        return WorkflowSpec(
            name=name or self.name, nodes=list(self.nodes), edges=[*self.edges, *ends], entry=self.entry
        )


__all__ = ["Fragment"]
