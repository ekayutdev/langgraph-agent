"""approval_gate — a checkpoint where a human, or an agent that can escalate
to a human, picks one of ``choices``. Returns plain spec data (a Fragment), so
the result opens in any spec-based editor and compiles with build_graph."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Literal

from lgkit.nodes.advisor import ESCALATE, advice_key
from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec


@dataclass(frozen=True)
class Fragment:
    """Nodes + internal edges of a pattern. Wire an edge into ``entry`` and
    route out of ``exit`` with ``condition=<choice>``."""

    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    entry: str
    exit: str
    choices: tuple[str, ...]

    def to_workflow(self, name: str = "approval_gate") -> WorkflowSpec:
        ends = [EdgeSpec(source=self.exit, target="END", condition=c) for c in self.choices]
        return WorkflowSpec(
            name=name, nodes=list(self.nodes), edges=[*self.edges, *ends], entry=self.entry
        )


def _check_template(message: str) -> None:
    try:
        fields = list(string.Formatter().parse(message))
    except ValueError as e:
        raise ValueError(f"approval_gate: message is not a valid template ({e}): {message!r}") from e
    for _, field, _, _ in fields:
        if field is not None and (not field.strip() or field.strip().isdigit()):
            # "" (Approve {}) and "0" (Approve {0}) are positional-index
            # syntax that flat_state format_map cannot resolve — require
            # named fields only.
            raise ValueError(
                f"approval_gate: message template fields must be named state paths, "
                f"got {field!r} in {message!r}"
            )


def approval_gate(
    id: str,
    *,
    message: str,
    choices: list[str],
    approver: Literal["human", "agent"] = "human",
    advise: bool = False,
    criteria: str | None = None,
    min_confidence: float = 0.7,
    llm: dict[str, Any] | None = None,
    result_key: str | None = None,
) -> Fragment:
    choices = list(choices)
    if not choices:
        raise ValueError("approval_gate: choices must not be empty")
    if len(set(choices)) != len(choices):
        raise ValueError(f"approval_gate: choices must be unique, got {choices}")
    if ESCALATE in choices:
        raise ValueError(f"approval_gate: '{ESCALATE}' is reserved for the advisor")
    if approver not in ("human", "agent"):
        raise ValueError(f"approval_gate: approver must be 'human' or 'agent', got {approver!r}")
    if advise and approver == "agent":
        raise ValueError("approval_gate: advise=True only applies to approver='human'")
    use_advisor = advise or approver == "agent"
    if llm and not use_advisor:
        raise ValueError("approval_gate: llm only configures the advisor; no advisor runs with approver='human' and advise=False")
    if use_advisor and not (criteria and criteria.strip()):
        raise ValueError("approval_gate: criteria is required when an advisor runs")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"approval_gate: min_confidence must be within 0..1, got {min_confidence}")
    _check_template(message)

    gate_params: dict[str, Any] = {
        "message": message,
        "choices": choices,
        "approver": approver,
        "min_confidence": min_confidence,
    }
    if result_key:
        gate_params["result_key"] = result_key

    nodes: list[NodeSpec] = []
    edges: list[EdgeSpec] = []
    if use_advisor:
        adv_id = f"{id}__advisor"
        adv_params: dict[str, Any] = {
            "gate_id": id,
            "message": message,
            "choices": choices,
            "criteria": criteria,
        }
        if llm:
            adv_params["llm"] = dict(llm)
        nodes.append(NodeSpec(id=adv_id, kind="advisor", params=adv_params))
        edges.append(EdgeSpec(source=adv_id, target=id))
        gate_params["advice_key"] = advice_key(id)
    nodes.append(NodeSpec(id=id, kind="approval", params=gate_params))

    return Fragment(
        nodes=tuple(nodes), edges=tuple(edges), entry=nodes[0].id, exit=id, choices=tuple(choices)
    )


__all__ = ["Fragment", "approval_gate"]
