"""Dynamic graph builder — turns a WorkflowSpec into a runnable StateGraph.

A node's ``kind`` selects the implementation from the registry; its
``prompt``/``params`` are injected at call time. Edges with a ``condition``
become conditional (router) edges keyed off the generic ``signal`` emitted
by nodes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from lgkit.context import current_ctx
from lgkit.hooks import DEFAULT_HOOKS, BuildHooks
from lgkit.node_validation import fatal_node_param_errors
from lgkit.registry import NodeDef, get_node_def
from lgkit.spec import NodeSpec, WorkflowSpec
from lgkit.state import AgentState


class GraphBuildError(RuntimeError):
    """Raised when a WorkflowSpec cannot be compiled into a graph."""


def _no_dispatch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {}


_VOLATILE_KEYS: frozenset[str] = frozenset({"events", "messages", "seq", "iteration", "signal"})

# Kinds that pause with interrupt(): they need a checkpointer to resume and must
# never be cached (a cached pausing node crashes resume inside LangGraph).
_PAUSING_KINDS: frozenset[str] = frozenset({"hitl", "harness", "approval"})
_NEEDS_CHECKPOINTER: frozenset[str] = frozenset({"approval"})

# Sources whose unmatched signal ENDs the run instead of falling through to an
# exit-key choice or the first conditional edge. For these kinds a fallthrough
# would act on a decision nobody made (auto-approve, or loop forever).
_END_ON_MISS_KINDS: frozenset[str] = frozenset({"approval", "llm"})

_NODE_CACHE: InMemoryCache | None = None


def _node_cache() -> InMemoryCache:
    global _NODE_CACHE
    if _NODE_CACHE is None:
        _NODE_CACHE = InMemoryCache()
    return _NODE_CACHE


def _retryable(exc: BaseException) -> bool:
    import asyncio

    if isinstance(exc, (GraphBubbleUp, KeyboardInterrupt, asyncio.CancelledError, SystemExit)):
        return False
    return True


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _cache_key_for(node: NodeSpec) -> Callable[[Any], str]:
    node_cfg = hashlib.sha256(
        _canon(
            {"id": node.id, "kind": node.kind, "prompt": node.prompt, "params": node.params}
        ).encode()
    ).hexdigest()

    def key(state: Any) -> str:
        state = state or {}
        task = str(state.get("task", "")) if isinstance(state, dict) else ""
        stable = (
            {
                k: v
                for k, v in state.items()
                if k not in _VOLATILE_KEYS and not str(k).startswith("__")
            }
            if isinstance(state, dict)
            else {}
        )
        return hashlib.sha256((node_cfg + "\0" + task + "\0" + _canon(stable)).encode()).hexdigest()

    return key


def _make_node_fn(
    node: NodeSpec,
    spec: WorkflowSpec,
    resolved=None,
    hooks: BuildHooks = DEFAULT_HOOKS,
) -> Callable:
    """Return a node function bound to the node's prompt/params + event hooks.

    Dispatches node.on_enter (before), node.on_exit (after success),
    node.on_error (on exception). An on_error route recovers the run by
    continuing to that node; otherwise node errors propagate. An event-script
    `route` is a dynamic goto (Command(goto=route)); it stacks with the node's
    own goto (event route wins) but, like goto mode, requires the node to have
    no outgoing plain edges (langgraph runs Command-goto and edges in parallel,
    causing InvalidUpdateError otherwise). Non-routing event hooks work on any
    node. Event state_deltas are accumulated and merged into the node's returned
    delta so they persist even when the node returns a fresh last-value channel
    value (e.g. scratch).
    """
    dispatch_event = hooks.dispatch_event or _no_dispatch

    defn: NodeDef = get_node_def(node.kind)
    effective_prompt = node.prompt or defn.default_prompt
    effective_params = {**defn.default_params, **(node.params or {}), "__node_id": node.id}

    def _node_trace(phase: str, payload: dict[str, Any]) -> None:
        # Best-effort trace: emit_event needs the langgraph stream context, which
        # exists inside astream but not under bare invoke(); never fail the run for telemetry.
        try:
            from lgkit.events import emit_event

            emit_event(f"event.script.{phase}", node.id, payload)
        except Exception:
            pass

    def _fn(state: AgentState):
        # Accumulate event state_deltas so they persist even when the node returns
        # a fresh delta (scratch etc. are last-value channels, so in-place mutations
        # alone would be overwritten by the node's returned value).
        event_delta: dict[str, Any] = {}

        ev = dispatch_event(
            "node.on_enter",
            state,
            scope=node.id,
            origin=node.id,
            node_id=node.id,
            spec=spec,
            trace=_node_trace,
        )
        _merge_into(event_delta, ev.get("state_delta") or {})

        route: str | None = None
        delta: dict[str, Any] = {}
        try:
            delta = defn.fn(state, effective_params, effective_prompt, current_ctx(), resolved)
            if not isinstance(delta, dict):
                delta = {}
            ev = dispatch_event(
                "node.on_exit",
                state,
                scope=node.id,
                origin=node.id,
                node_id=node.id,
                spec=spec,
                trace=_node_trace,
            )
            route = ev.get("route")
            _merge_into(event_delta, ev.get("state_delta") or {})
        except GraphBubbleUp:
            # An approval interrupt() (GraphInterrupt's base) is a PAUSE, not an
            # error. The harness makes interrupt() a routine path; routing it to
            # node.on_error would CONSUME the pause (an on_error route swallows
            # the __interrupt__) or fire an error hook on every routine gate
            # 'ask'. Re-raise untouched, mirroring _retryable's exclusion.
            raise
        except Exception:
            ev = dispatch_event(
                "node.on_error",
                state,
                scope=node.id,
                origin=node.id,
                node_id=node.id,
                spec=spec,
                trace=_node_trace,
            )
            route = ev.get("route")
            _merge_into(event_delta, ev.get("state_delta") or {})
            if route is None:
                raise  # no recovery -> propagate, run fails
            delta = {}  # recovery: clear the error, fall through to routing

        # Merge accumulated event deltas on top of the node's delta so event
        # mutations survive the node's returned (fresh) values.
        _merge_into(delta, event_delta)

        # Node's own goto (existing behavior).
        if delta.get("goto"):
            target = delta.pop("goto")
            targets = (node.params or {}).get("goto_targets")
            if isinstance(targets, list) and targets:
                if target not in targets:
                    target = targets[0]
            elif node.kind == "script":
                raise GraphBuildError(
                    f"Script node '{node.id}' emitted goto target '{target}' "
                    "but declares no 'goto_targets' param."
                )
            if route is None:
                route = END if target == "END" else target

        if route is not None:
            return Command(goto=END if route == "END" else route, update=delta)
        return delta

    _fn.__name__ = f"{node.kind}_node"
    return _fn


def _merge_into(base: dict[str, Any], extra: dict[str, Any]) -> None:
    """Shallow-merge event state_delta into the node's delta (in place)."""
    for k, v in extra.items():
        if isinstance(base.get(k), dict) and isinstance(v, dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v


def _signal_router(conditions: list[str], end_on_miss: bool = False) -> Callable[[dict[str, Any]], str]:
    def _route(state: dict[str, Any]) -> str:
        signal = state.get("signal") or ""
        if signal in conditions:
            return signal
        if end_on_miss:
            # An approval / llm / loop_limit source (see _END_ON_MISS_KINDS)
            # on a back-edge gets an iteration guard after it; once the budget
            # is exhausted the guard emits "max_iterations", which matches no
            # choice. Falling through to any exit-key choice
            # ("done"/"failed"/"max_iterations") or conditions[0] could
            # act on a decision nobody made (auto-approve, or loop forever) —
            # END the run instead. Other kinds keep the historical behaviour.
            return "END"
        for exit_key in ("done", "failed", "max_iterations"):
            if exit_key in conditions:
                return exit_key
        return conditions[0]

    return _route


def _iteration_guard(state: dict[str, Any]) -> dict[str, Any]:
    """Increment the iteration counter and emit max_iterations signal when
    the budget is exhausted.

    Injected after every agent node in loop-prone workflows so the graph
    halts gracefully via exit_conditions instead of hitting LangGraph's
    hard recursion_limit. The guard reads scratch['iteration'] and
    scratch['max_iterations'] (seeded by runner._initial_state).
    """
    scratch = state.get("scratch") or {}
    iteration = int(scratch.get("iteration", 0)) + 1
    max_iterations = scratch.get("max_iterations")
    out: dict[str, Any] = {"scratch": {"iteration": iteration}}
    if max_iterations is not None and iteration >= int(max_iterations):
        out["signal"] = "max_iterations"
    return out


def detect_workflow_cycle(
    spec: WorkflowSpec,
    resolver: Callable,
    agent_resolver: Callable | None = None,
    _seen: list[str] | None = None,
    _ref_stack: tuple[str, ...] = (),
    _labels: tuple[str, ...] = (),
) -> None:
    """Raise ``GraphBuildError`` if the workflow embeds itself by reference.

    Cycle identity uses TWO keys tracked in parallel:

    - ``id(spec)`` (object identity) — catches cycles through the SAME object,
      e.g. an in-memory-registered workflow that references itself. This is
      what commit 7b749a2 switched to, to avoid ``spec.name`` (which defaults
      to ``"custom"``) colliding across unrelated workflows.
    - ``_ref_stack`` (the chain of named refs followed to get here) — catches
      cycles through FRESH objects, e.g. an on-disk workflow whose
      ``WorkflowStore.get`` returns a brand-new ``model_validate_json``
      object every call, so ``id(spec)`` differs per resolve. Without this,
      a true A→B→A cycle in stored workflows would recurse until Python's
      limit instead of raising ``GraphBuildError``.

    Inline-dict subworkflows have no ref name and cannot form a reference
    cycle (pydantic rejects self-embedding dicts), so they are walked with
    ``id(spec)`` only.

    ``_labels`` is display only: the identity keys above are what decide whether
    there is a cycle, but rendering them raw produced
    "Workflow reference cycle: 4479080720 -> 4479080720", which names neither
    workflow. Each hop is labelled by the ref used to reach it (``spec.name`` for
    the root), so the message points at something the author can act on.
    """
    _seen = _seen or []
    # Use object identity (memory address) for cycle detection — spec.name is
    # often the generic source label ("custom", "preset") and collides across
    # unrelated workflows, causing false positive cycle errors.
    wid = str(id(spec))
    label = _ref_stack[-1] if _ref_stack else (spec.name or "<root>")
    if wid in _seen:
        chain = " -> ".join([*_labels, label])
        raise GraphBuildError(f"Workflow reference cycle: {chain}")
    _seen = [*_seen, wid]
    _labels = (*_labels, label)
    _walk_nodes(spec.nodes, resolver, agent_resolver, _seen, _ref_stack, _labels)


def _walk_nodes(nodes, resolver, agent_resolver, _seen, _ref_stack=(), _labels=()):
    def _recurse_named(ref: str):
        # Named refs are the stable identity for cycle detection across
        # fresh-object resolves (on-disk store). Same ref twice on the path
        # = true cycle, regardless of object identity.
        if ref in _ref_stack:
            chain = " -> ".join([*_labels, ref])
            raise GraphBuildError(f"Workflow reference cycle: {chain}")
        child = resolver(ref)
        if child is not None:
            detect_workflow_cycle(
                child, resolver, agent_resolver, _seen, (*_ref_stack, ref), _labels
            )

    for node in nodes:
        if node.kind == "subworkflow":
            ref = node.params.get("ref")
            if isinstance(ref, str):
                _recurse_named(ref)
            elif isinstance(ref, dict):
                try:
                    inline = WorkflowSpec.model_validate(ref)
                except (TypeError, ValueError):
                    continue
                _walk_nodes(inline.nodes, resolver, agent_resolver, _seen, _ref_stack, _labels)
        elif node.kind in ("agent", "react_agent", "map") and agent_resolver is not None:
            aid = node.params.get("agent")
            agent = agent_resolver(aid) if aid is not None else None
            if agent is not None and getattr(agent, "workflow_ref", None):
                _recurse_named(agent.workflow_ref)
            if agent is not None:
                for tid in agent.agent_tools:
                    if tid == agent.id:
                        continue
                    tool_agent = agent_resolver(tid)
                    ref = getattr(tool_agent, "workflow_ref", None) if tool_agent else None
                    if ref:
                        _recurse_named(ref)


def validate_parallel_branches(spec: WorkflowSpec, resolver: Callable) -> None:
    plain_targets = spec._plain_targets_by_source()
    node_by_id = {n.id: n for n in spec.nodes}
    for src, targets in plain_targets.items():
        if len(targets) < 2:
            continue
        keys: dict[str, str] = {}
        for b in targets:
            node = node_by_id.get(b)
            if node is None or node.kind not in ("agent", "react_agent"):
                continue
            try:
                agent_id = node.params["agent"]
            except KeyError as e:
                raise GraphBuildError(
                    f"Parallel branch '{b}' references an unknown agent: {e}"
                ) from e
            agent = resolver(agent_id)
            if agent is None:
                continue  # unknown agent — fails at run time
            if agent.output is not None:
                mode = getattr(agent.output, "mode", "signal")
                raise GraphBuildError(
                    f"Parallel branch '{b}' has a routing output (mode={mode}); "
                    "branch agents must not route."
                )
            key = agent.result_key or agent.id
            if key in keys:
                raise GraphBuildError(
                    f"Parallel branches '{keys[key]}' and '{b}' both write scratch key '{key}'."
                )
            keys[key] = b


def validate_map_nodes(spec: WorkflowSpec, resolver: Callable) -> None:
    for n in spec.nodes:
        if n.kind != "map":
            continue
        agent_id = (n.params or {}).get("agent")
        try:
            agent = resolver(agent_id)
        except KeyError as e:
            raise GraphBuildError(f"Map node '{n.id}' references unknown agent: {e}") from e
        if agent.output is not None:
            mode = getattr(agent.output, "mode", "signal")
            raise GraphBuildError(
                f"Map node '{n.id}' worker agent '{agent_id}' must not route (mode={mode})."
            )


def validate_goto_agents(spec: WorkflowSpec, resolver: Callable) -> None:
    ids = spec.node_ids()
    for n in spec.nodes:
        if n.kind not in ("agent", "react_agent"):
            continue
        aid = (n.params or {}).get("agent")
        agent = resolver(aid) if aid is not None else None
        if agent is None:
            continue  # unknown agent — fails at run time with a clear error
        if agent.output is None or getattr(agent.output, "mode", "signal") != "goto":
            continue
        targets = agent.output.values
        for t in targets:
            if t not in ids and t != "END":
                raise GraphBuildError(
                    f"Goto agent node '{n.id}' target '{t}' is not a node or 'END'"
                )
        out = [e for e in spec.edges if e.source == n.id]
        if out:
            raise GraphBuildError(f"Goto agent node '{n.id}' must not have outgoing edges")


def _resolve_over(scratch: dict, node_id: str, over: str):
    """Resolve a map node's ``over`` path relative to ``scratch``.

    Uses the same dotted-path semantics as ``subworkflow_node._get_path`` (which
    ``input_map`` and the ``query`` node's parameter binding already use), so an
    ``over: "customers.rows"`` resolves to ``scratch["customers"]["rows"]`` while
    a flat ``over: "items"`` keeps resolving to ``scratch["items"]`` — the
    single-segment case is unchanged.

    ``None`` (an absent key) means "no items" and returns ``None``; the caller
    routes that to the gather node for an empty result, matching the prior
    behaviour. A non-None value that is not a ``list``/``tuple`` raises
    ``GraphBuildError`` naming the node, the ``over`` path and what was found —
    otherwise a dict (truthy) would silently fan out over its keys and a ``str``
    (iterable) would fan out per character.
    """
    cur: Any = scratch
    for part in over.split("."):
        cur = (cur or {}).get(part) if isinstance(cur, dict) else None
    if cur is None:
        return None
    if not isinstance(cur, list | tuple):
        raise GraphBuildError(
            f"Map node '{node_id}' over='{over}' must point at a list, "
            f"got {type(cur).__name__}: {cur!r}"
        )
    return cur


def _expand_map_node(graph, m, successor: str, resolved=None, spec=None, hooks: BuildHooks = DEFAULT_HOOKS) -> None:
    over = m.params["over"]
    agent_id = m.params["agent"]
    result_key = m.params["result_key"]
    worker_id = f"{m.id}__worker"
    gather_id = f"{m.id}__gather"

    graph.add_node(m.id, lambda state: {})
    graph.add_node(
        worker_id,
        _make_node_fn(
            NodeSpec(
                id=worker_id,
                kind="agent",
                prompt=m.prompt,
                params={"agent": agent_id, "__map_result_key": result_key},
            ),
            spec,
            resolved,
            hooks,
        ),
    )

    def _gather(state: dict, _over: str = over, _rk: str = result_key) -> dict:
        scratch = state.get("scratch", {})
        items = _resolve_over(scratch, m.id, _over)
        n = len(items or [])
        return {"scratch": {_rk: [scratch.get(f"{_rk}__{i}") for i in range(n)]}}

    graph.add_node(gather_id, _gather)

    def _route(state, _over: str = over, _w: str = worker_id, _g: str = gather_id):
        items = _resolve_over(state.get("scratch", {}), m.id, _over)
        sends = [
            Send(_w, {"scratch": {"__map_item": it, "__map_index": i}})
            for i, it in enumerate(items or [])
        ]
        return sends or [Send(_g, {})]

    graph.add_conditional_edges(m.id, _route, [worker_id, gather_id])
    graph.add_edge(worker_id, gather_id)
    graph.add_edge(gather_id, END if successor == "END" else successor)


def _build_state_schema(spec: WorkflowSpec):
    if not spec.state_fields:
        return AgentState
    from langgraph.channels import Topic

    from lgkit.state import get_reducer

    base_annotations: dict = dict(AgentState.__annotations__)
    universal_channels = set(base_annotations.keys())
    forbidden = universal_channels | {
        "plan",
        "review",
        "current_step",
        "status",
        "test_results",
        "iteration",
        "max_iterations",
    }
    for sf in spec.state_fields:
        if sf.name in forbidden:
            raise GraphBuildError(
                f"state_field {sf.name!r} shadows a universal channel or removed field — use a different name"
            )

    _TYPE_MAP = {
        "string": str,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
        "list": list,
    }
    for sf in spec.state_fields:
        py_type = _TYPE_MAP.get(sf.type, str)
        reducer = get_reducer(sf.reducer)
        if reducer == "topic":
            base_annotations[sf.name] = Annotated[py_type, Topic(py_type)]
        elif reducer is not None:
            base_annotations[sf.name] = Annotated[py_type, reducer]
        else:
            base_annotations[sf.name] = py_type

    DynamicState = TypedDict(
        "DynamicState", {k: v for k, v in base_annotations.items()}, total=False
    )
    return DynamicState


def build_graph(
    spec: WorkflowSpec,
    checkpointer: Any = None,
    store: Any = None,
    dry_run: bool = False,
    *,
    hooks: BuildHooks = DEFAULT_HOOKS,
    resolved: Any = None,
) -> Any:
    """Compile a WorkflowSpec into a runnable LangGraph.

    ``hooks`` supplies agent/workflow resolution and event-script dispatch;
    ``resolved`` is handed to every node fn unchanged (an embedding app
    passes its resolved unit, standalone callers leave it None).
    """
    from lgkit.nodes import ensure_registered

    ensure_registered()

    if checkpointer is None:
        needing = [n.id for n in spec.nodes if n.kind in _NEEDS_CHECKPOINTER]
        if needing:
            raise GraphBuildError(
                f"Nodes {needing} can pause for a human and need a checkpointer to resume: "
                "build_graph(spec, checkpointer=InMemorySaver()) "
                "(from langgraph.checkpoint.memory import InMemorySaver)"
            )

    # Default memory nodes' namespace
    for node in spec.nodes:
        if node.kind == "memory" and not (node.params or {}).get("namespace"):
            if not isinstance(node.params, dict):
                node.params = {}
            node.params["namespace"] = list(spec.memory_scope)

    errors = spec.validate_edges()
    errors.extend(spec.validate_events())
    if errors:
        raise GraphBuildError("Invalid workflow: " + "; ".join(errors))

    if any(n.kind in ("subworkflow", "agent", "react_agent", "map") for n in spec.nodes):
        detect_workflow_cycle(spec, hooks.workflow_resolver, agent_resolver=hooks.agent_resolver)

    # Required-param check first: a subworkflow with no `ref` or a harness node
    # with no `harness` used to compile cleanly and then die mid-run with a bare
    # KeyError, which named neither the node nor the missing key.
    param_errors = fatal_node_param_errors(spec)
    if param_errors:
        raise GraphBuildError("Invalid workflow: " + "; ".join(param_errors))

    validate_parallel_branches(spec, hooks.agent_resolver)
    validate_map_nodes(spec, hooks.agent_resolver)
    validate_goto_agents(spec, hooks.agent_resolver)
    map_ids = {n.id for n in spec.nodes if n.kind == "map"}
    plain_targets = spec._plain_targets_by_source()
    node_by_id = {n.id: n for n in spec.nodes}

    graph = StateGraph(_build_state_schema(spec))

    from langgraph.types import CachePolicy, RetryPolicy

    wants_cache = False
    for node in spec.nodes:
        if node.kind == "map":
            continue
        kwargs: dict[str, Any] = {}
        pol = node.policy
        if pol is not None and not dry_run:
            if pol.retry is not None:
                kwargs["retry_policy"] = RetryPolicy(
                    max_attempts=pol.retry.max_attempts,
                    initial_interval=pol.retry.initial_interval_s,
                    backoff_factor=pol.retry.backoff_factor,
                    max_interval=pol.retry.max_interval_s,
                    retry_on=_retryable,
                )
            # Caching is unsafe for nodes that PAUSE (hitl) or have external
            # side effects (harness):
            # hitl — a cached hitl node with an empty write set crashes resume
            #   inside LangGraph (deque index out of range on INTERRUPT).
            # harness — the approval pause becomes un-resumable the same way,
            #   and a replayed cache entry skips the agent entirely, bypassing
            #   every permission-gate evaluation and all side effects. _NODE_CACHE
            #   is process-global, so this would leak across runs (NEW-1).
            if pol.cache is not None and node.kind not in _PAUSING_KINDS:
                kwargs["cache_policy"] = CachePolicy(
                    key_func=_cache_key_for(node), ttl=pol.cache.ttl_s
                )
                wants_cache = True
        graph.add_node(
            node.id,
            _make_node_fn(node, spec, resolved, hooks),
            **kwargs,
        )
    for node in spec.nodes:
        if node.kind == "map":
            _expand_map_node(graph, node, plain_targets[node.id][0], resolved, spec, hooks)

    graph.add_edge(START, spec.entry)

    # Detect back-edges: source nodes with self-loops or edges to earlier
    # nodes. These are loop-prone and need an iteration guard injected after
    # them so the graph halts at max_iterations instead of crashing into
    # LangGraph's hard recursion_limit.
    node_order = [n.id for n in spec.nodes if n.id not in map_ids]
    node_index = {nid: i for i, nid in enumerate(node_order)}
    guard_sources: set[str] = set()
    for edge in spec.edges:
        if edge.source in map_ids:
            continue
        if edge.target == edge.source:
            guard_sources.add(edge.source)
        elif edge.target in node_index and edge.source in node_index:
            if node_index[edge.target] <= node_index[edge.source]:
                guard_sources.add(edge.source)

    guard_id_map: dict[str, str] = {}
    for src in guard_sources:
        guard_id = f"{src}__guard"
        guard_id_map[src] = guard_id
        graph.add_node(guard_id, _iteration_guard)

    by_source: dict[str, list] = {}
    for edge in spec.edges:
        by_source.setdefault(edge.source, []).append(edge)

    for source, edges in by_source.items():
        if source in map_ids:
            continue
        conditional = [e for e in edges if e.condition]
        plain = [e for e in edges if not e.condition]

        # If this source has a guard, edges originate from the guard node,
        # and we add a plain edge from source → guard.
        edge_source = guard_id_map.get(source, source)
        if source in guard_id_map:
            graph.add_edge(source, edge_source)

        for e in plain:
            target = END if e.target == "END" else e.target
            graph.add_edge(edge_source, target)

        if conditional:
            conditions = [e.condition for e in conditional if e.condition]
            mapping: dict[str, str] = {}
            for e in conditional:
                assert e.condition is not None
                target = END if e.target == "END" else e.target
                mapping[e.condition] = target
            # Scoped router fix: when an approval / llm / loop_limit source
            # (see _END_ON_MISS_KINDS) emits a signal that matches no
            # condition, END the run instead of falling through to the first
            # conditional edge — acting on a decision nobody made
            # (auto-approving work the human chose to revise, or looping
            # forever). Other kinds (hitl etc.) keep the historical
            # conditions[0] fallback.
            src_node = node_by_id.get(source)
            end_on_miss = src_node is not None and src_node.kind in _END_ON_MISS_KINDS
            router = _signal_router(conditions, end_on_miss=end_on_miss)
            if end_on_miss:
                mapping.setdefault("END", END)
            graph.add_conditional_edges(edge_source, router, mapping)

    compile_kwargs: dict[str, Any] = {"checkpointer": checkpointer}
    if wants_cache and not dry_run:
        compile_kwargs["cache"] = _node_cache()
    if store is not None and not dry_run:
        compile_kwargs["store"] = store
    return graph.compile(**compile_kwargs)


__all__ = [
    "build_graph",
    "GraphBuildError",
    "detect_workflow_cycle",
    "validate_goto_agents",
    "validate_map_nodes",
    "validate_parallel_branches",
]
