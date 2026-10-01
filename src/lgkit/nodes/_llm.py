"""Pick the model a node should call. Shared by every node that calls an LLM.

A dry-run ALWAYS wins: it must never build a real model, even when the node
carries its own ``llm`` spec or the run context has one. If a dry-run is bound
in the ambient context but did not reach the node as ``ctx``, the ctx plumbing
regressed — refuse to build a real model rather than silently spend API money.
"""

from __future__ import annotations

from typing import Any

from lgkit.context import current_ctx


def resolve_llm(params: dict[str, Any], ctx: Any, node_id: str) -> Any:
    """Precedence: ctx dry-run -> refuse on a lost dry-run -> the node's own
    ``llm`` spec -> the run context's llm -> lgkit's default."""
    from lgkit.llm import build_llm

    if ctx is not None and getattr(ctx, "dry_run", None) is not None:
        return ctx.dry_run.model_for(node_id)
    bound_ctx = current_ctx()
    if bound_ctx is not None and getattr(bound_ctx, "dry_run", None) is not None:
        # Checked before the node's llm spec: that spec builds a real model too.
        raise RuntimeError("dry-run context lost: refusing to build a real model")
    if params.get("llm"):
        return build_llm(**params["llm"])
    if ctx is not None and getattr(ctx, "llm", None) is not None:
        return ctx.llm
    return build_llm()


__all__ = ["resolve_llm"]
