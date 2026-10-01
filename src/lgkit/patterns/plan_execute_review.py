"""plan_execute_review — plan once, then execute and review until the reviewer
approves or the round budget is spent. Returns plain spec data (a Fragment).

    <id>__plan -> <id>__execute -> <id>__review --approve--> exit "approved"
                      ^                 |
                      |               revise
                      |                 v
                      +---again--- <id>__limit --exhausted--> exit "exhausted"

Running out of rounds is NOT an approval: the run leaves through "exhausted"
and scratch["<id>__exhausted"] is True.
"""

from __future__ import annotations

from typing import Any

from lgkit.nodes.loop_limit import AGAIN, EXHAUSTED
from lgkit.patterns._fragment import Fragment
from lgkit.spec import EdgeSpec, NodeSpec

APPROVE = "approve"
REVISE = "revise"


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"plan_execute_review: {name} must be a non-empty prompt")
    return value


def plan_execute_review(
    id: str,
    *,
    planner: str,
    executor: str | NodeSpec,
    reviewer: str,
    max_rounds: int = 3,
    llm: dict[str, Any] | None = None,
    executor_result_key: str | None = None,
) -> Fragment:
    """``max_rounds`` is the most times the execute step may run."""
    if not isinstance(id, str) or not id.strip():
        raise ValueError(f"plan_execute_review: id must be a non-blank pattern id, got {id!r}")
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError(f"plan_execute_review: max_rounds must be a whole number >= 1, got {max_rounds!r}")
    _require_text("planner", planner)
    _require_text("reviewer", reviewer)

    plan_id, exec_id, review_id, limit_id = (f"{id}__plan", f"{id}__execute", f"{id}__review", f"{id}__limit")
    plan_key, review_key = plan_id, review_id
    llm_params: dict[str, Any] = {"llm": dict(llm)} if llm else {}

    if isinstance(executor, NodeSpec):
        if not executor_result_key:
            raise ValueError(
                "plan_execute_review: executor_result_key is required when executor is a NodeSpec "
                "(the scratch key your node writes its work to)"
            )
        if executor.id in (plan_id, review_id, limit_id):
            raise ValueError(f"plan_execute_review: executor id '{executor.id}' collides with a pattern node")
        exec_node = executor
        result_key = executor_result_key
    else:
        _require_text("executor", executor)
        if executor_result_key:
            raise ValueError(
                "plan_execute_review: executor_result_key only applies when executor is a NodeSpec"
            )
        result_key = f"{id}__result"
        exec_node = NodeSpec(
            id=exec_id,
            kind="llm",
            prompt=executor,
            params={
                **llm_params,
                "result_key": result_key,
                "message": (
                    "Task:\n{task}\n\n"
                    f"Plan:\n{{scratch.{plan_key}}}\n\n"
                    f"Your previous attempt (empty on the first round):\n{{scratch.{result_key}}}\n\n"
                    f"Reviewer feedback to address (empty on the first round):\n{{scratch.{review_key}.feedback}}"
                ),
            },
        )

    nodes = (
        NodeSpec(
            id=plan_id,
            kind="llm",
            prompt=planner,
            params={**llm_params, "result_key": plan_key, "message": "Task:\n{task}"},
        ),
        exec_node,
        NodeSpec(
            id=review_id,
            kind="llm",
            prompt=reviewer,
            params={
                **llm_params,
                "result_key": review_key,
                "message": (
                    "Task:\n{task}\n\n"
                    f"Plan:\n{{scratch.{plan_key}}}\n\n"
                    f"Work to review:\n{{scratch.{result_key}}}"
                ),
                "output_fields": [
                    {"name": "verdict", "type": "string",
                     "description": f"'{APPROVE}' only if the work meets the criteria, otherwise '{REVISE}'"},
                    {"name": "feedback", "type": "string", "required": False,
                     "description": "What must change; empty when approving"},
                ],
                "signal_field": "verdict",
                "signal_values": [APPROVE, REVISE],
            },
        ),
        NodeSpec(
            id=limit_id,
            kind="loop_limit",
            params={
                "max_rounds": max_rounds,
                "counter_key": f"{id}__round",
                "exhausted_key": f"{id}__exhausted",
            },
        ),
    )
    edges = (
        EdgeSpec(source=plan_id, target=exec_node.id),
        EdgeSpec(source=exec_node.id, target=review_id),
        EdgeSpec(source=review_id, target=limit_id, condition=REVISE),
        EdgeSpec(source=limit_id, target=exec_node.id, condition=AGAIN),
    )
    return Fragment(
        nodes=nodes,
        edges=edges,
        entry=plan_id,
        exit=review_id,
        choices=(),
        exits={"approved": (review_id, APPROVE), "exhausted": (limit_id, EXHAUSTED)},
        name=id,
    )


__all__ = ["APPROVE", "REVISE", "plan_execute_review"]
