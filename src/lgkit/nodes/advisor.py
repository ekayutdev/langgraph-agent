"""Advisor node — an LLM recommends one of an approval gate's choices.

It runs as its own node BEFORE the gate so the LLM call is checkpointed:
LangGraph re-executes an interrupted node from its first line on resume, so an
LLM call inside the gate would be paid twice and could change the advice the
human already saw. It never raises; a failure is recorded as
{"error": "..."} and the gate treats that as "ask a human".
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel, Field, create_model

from lgkit.context import current_ctx
from lgkit.nodes._template import flat_state

ESCALATE = "escalate"

SYSTEM_PROMPT = (
    "You review work before it moves on. Decide using these criteria:\n"
    "{criteria}\n\n"
    "Reply with exactly one choice from: {choices}. "
    "Reply '{escalate}' if the criteria do not let you decide. "
    "confidence is 0-1: how sure you are. reason is one or two sentences."
)


def advice_key(gate_id: str) -> str:
    return f"{gate_id}__advice"


def advice_model(choices: list[str]) -> type[BaseModel]:
    return create_model(
        "Advice",
        choice=(Literal[tuple([*choices, ESCALATE])], ...),
        confidence=(float, Field(ge=0.0, le=1.0)),
        reason=(str, ""),
    )


def _llm_for(params: dict[str, Any], ctx: Any) -> Any:
    """Precedence: ctx dry-run -> (refuse if the ambient context has a dry-run
    that did not reach us) -> per-gate llm spec -> run context llm -> lgkit
    default.

    A dry-run ALWAYS wins: it must never build a real model, even when a
    per-gate ``llm`` spec or ``ctx.llm`` is set. If a dry-run is bound in the
    ambient context but did not reach us as ``ctx``, the ctx plumbing
    regressed — refuse to build a real model rather than silently spending
    API money (mirrors lgtools' agent_node).
    """
    from lgkit.llm import build_llm

    if ctx is not None and getattr(ctx, "dry_run", None) is not None:
        return ctx.dry_run.model_for(
            params.get("__node_id") or f"{params['gate_id']}__advisor"
        )
    bound_ctx = current_ctx()
    if bound_ctx is not None and getattr(bound_ctx, "dry_run", None) is not None:
        # Checked before the per-gate llm spec: that spec builds a real model too.
        raise RuntimeError("dry-run context lost: refusing to build a real model")
    if params.get("llm"):
        return build_llm(**params["llm"])
    if ctx is not None and getattr(ctx, "llm", None) is not None:
        return ctx.llm
    return build_llm()


def _system_prompt(template: str, criteria: str, choices: list[str]) -> str:
    # str.replace, not format(): a user-supplied prompt may contain other braces.
    return (
        template.replace("{criteria}", criteria)
        .replace("{choices}", ", ".join(choices))
        .replace("{escalate}", ESCALATE)
    )


def run(state, params, prompt, ctx=None, resolved=None) -> dict[str, Any]:
    gate_id = params["gate_id"]
    choices = list(params["choices"])
    message = str(params.get("message") or "").format_map(flat_state(state))
    system = _system_prompt(prompt or SYSTEM_PROMPT, str(params.get("criteria") or ""), choices)
    schema = advice_model(choices)

    result: dict[str, Any] = {"error": "advisor made no attempt"}
    for _ in range(max(1, int(params.get("max_attempts", 2)))):
        try:
            structured = _llm_for(params, ctx).with_structured_output(schema)
            out = structured.invoke([SystemMessage(system), HumanMessage(message)])
            # A fake/scripted model returns a plain AIMessage (itself a
            # pydantic model) whose .content is the reply: parse that as
            # JSON before the BaseModel branch can misread it.
            if isinstance(out, BaseMessage):
                data = json.loads(out.content)
            elif isinstance(out, BaseModel):
                data = out.model_dump()
            else:
                data = out
            result = schema.model_validate(data).model_dump()
            break
        except GraphBubbleUp:
            raise
        except Exception as e:  # recorded, never raised: the gate escalates to a human
            result = {"error": f"{type(e).__name__}: {e}"}

    return {"scratch": {advice_key(gate_id): result}}


__all__ = ["ESCALATE", "SYSTEM_PROMPT", "advice_key", "advice_model", "run"]
