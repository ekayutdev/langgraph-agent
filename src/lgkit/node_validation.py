"""Check that every node carries the params its implementation indexes.

Three node kinds read a required param straight out of ``node.params``:
``subworkflow`` needs ``ref``, ``harness`` needs ``harness``, and ``tool_node``
needs at least one tool id. When the key is missing the first two die mid-run
with a bare ``KeyError`` and the third degrades to a silent no-op — all three
long after the author left the editor. Reporting them from the spec turns a
run-time surprise into an editor-time message, the same way
``query_validation`` does for a query node's SQL.

Two entry points on purpose:

- :func:`validate_node_params` — everything worth telling the author about.
  Feeds ``node_errors`` on ``POST /workflows/validate``.
- :func:`fatal_node_param_errors` — only the problems that make compilation
  pointless (the two ``KeyError`` cases). ``build_graph`` raises on these.
  An empty ``tool_node`` is *not* fatal: it is a documented no-op, so the
  editor flags it while the graph still compiles and runs.
"""

from __future__ import annotations

# (message, fatal)
_Problem = tuple[str, bool]


def _is_blank(value) -> bool:
    """True when a param is present but carries nothing usable.

    ``ref`` accepts either a workflow name or an inline spec dict, so blankness
    has to cover the empty string, whitespace, and the empty container alike.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict | list):
        return not value
    return False


def _problems(spec) -> list[_Problem]:
    found: list[_Problem] = []
    for node in spec.nodes:
        params = node.params or {}
        where = f"{node.kind} node '{node.id}'"

        if node.kind == "subworkflow":
            if _is_blank(params.get("ref")):
                found.append(
                    (f"{where}: 'ref' is required — pick the workflow this node runs", True)
                )

        elif node.kind == "harness":
            if _is_blank(params.get("harness")):
                found.append(
                    (f"{where}: 'harness' is required — pick the harness this node runs", True)
                )

        elif node.kind == "tool_node":
            # tool_node.run accepts either name; blank in both means it will
            # execute nothing and return an empty message list.
            if _is_blank(params.get("tools")) and _is_blank(params.get("tool_ids")):
                found.append((f"{where}: no tools selected — this node will do nothing", False))

        elif node.kind == "advisor":
            # advisor.run indexes params["gate_id"] and params["choices"]
            # directly — blank ones die mid-run with a bare KeyError.
            if _is_blank(params.get("gate_id")):
                found.append((f"{where}: 'gate_id' is required — name the gate this advisor serves", True))
            if _is_blank(params.get("choices")):
                found.append((f"{where}: 'choices' is required — the gate has nothing to pick from", True))

        elif node.kind == "approval":
            if _is_blank(params.get("choices")):
                found.append((f"{where}: 'choices' is required — the gate has nothing to pick from", True))
            elif any(_is_blank(c) or c == "escalate" for c in params.get("choices") or []):
                # 'escalate' is the advisor's reserved word; a gate offering it
                # would let the human answer it and collide with escalation.
                found.append(
                    (f"{where}: 'choices' must be non-blank and must not contain 'escalate'", True)
                )

        elif node.kind == "llm":
            fields = params.get("output_fields") or []
            names = [f.get("name") for f in fields if isinstance(f, dict)]
            if len(names) != len(fields) or any(_is_blank(n) for n in names):
                found.append((f"{where}: every entry of 'output_fields' needs a 'name'", True))
            if len(set(names)) != len(names):
                found.append((f"{where}: 'output_fields' names must be unique", True))
            bad_types = [
                str(f.get("type"))
                for f in fields
                if isinstance(f, dict) and f.get("type") is not None
                and f.get("type") not in ("string", "number", "boolean", "object", "array")
            ]
            if bad_types:
                found.append((f"{where}: every 'type' in 'output_fields' must be one of string, number, boolean, object, array", True))
            signal_field = params.get("signal_field")
            if not _is_blank(signal_field):
                if signal_field not in names:
                    found.append(
                        (f"{where}: 'signal_field' must name one of 'output_fields'", True)
                    )
                values = params.get("signal_values")
                ok_values = (
                    isinstance(values, list)
                    and bool(values)
                    and all(isinstance(v, str) and v.strip() for v in values)
                )
                if not ok_values:
                    found.append(
                        (
                            f"{where}: 'signal_values' must be a list of non-blank strings when 'signal_field' is set",
                            True,
                        )
                    )
            if _is_blank(signal_field) and any(
                e.source == node.id and e.condition for e in spec.edges
            ):
                found.append(
                    (
                        f"{where}: has conditional edges but no 'signal_field' — it would route on a stale signal",
                        True,
                    )
                )

        elif node.kind == "loop_limit":
            max_rounds = params.get("max_rounds")
            if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
                found.append((f"{where}: 'max_rounds' must be a whole number >= 1", True))

    return found


def validate_node_params(spec) -> list[str]:
    """Return one message per problem found, fatal or not."""
    return [msg for msg, _ in _problems(spec)]


def fatal_node_param_errors(spec) -> list[str]:
    """Return only the problems that make the graph unrunnable."""
    return [msg for msg, fatal in _problems(spec) if fatal]


__all__ = ["validate_node_params", "fatal_node_param_errors"]
