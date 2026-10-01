"""Workflow and agent spec models.

A workflow is a list of nodes + edges; :func:`lgkit.builder.build_graph` turns a
:class:`WorkflowSpec` into a runnable LangGraph ``StateGraph``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

NodeKind = Literal[
    "custom",
    "agent",
    "react_agent",
    "script",
    "query",
    "subworkflow",
    "map",
    "hitl",
    "memory",
    "tool_node",
    "swarm",
    "supervisor",
    "harness",
    "advisor",
    "approval",
]


def namespace_from_scope(scope: list[str]) -> tuple[str, ...]:
    """Convert a ``memory_scope`` list to a Store namespace tuple.

    Validates the langgraph Store namespace constraints early (at schema/build
    time) so an invalid namespace fails before reaching the Store.
    """
    if not scope:
        raise ValueError("memory_scope must be non-empty")
    ns = tuple(scope)
    if ns[0] == "langgraph":
        raise ValueError('memory_scope root label cannot be "langgraph" (reserved by langgraph)')
    for label in ns:
        if not isinstance(label, str) or not label:
            raise ValueError(f"memory_scope labels must be non-empty strings; got {label!r}")
        if "." in label:
            raise ValueError(f"memory_scope labels cannot contain dots; got {label!r}")
    return ns


Provider = Literal[
    "openai",
    "anthropic",
    "google",
    "deepseek",
    "xai",
    "meta",
    "mistral",
    "cohere",
    "zai",
    "litellm",
]


class RetryConfig(BaseModel):
    """Per-node retry policy (maps to langgraph RetryPolicy)."""

    max_attempts: int = Field(2, ge=1, description="Total attempts incl. the first")
    backoff_factor: float = Field(2.0, gt=0)
    initial_interval_s: float = Field(0.5, gt=0)
    max_interval_s: float = Field(8.0, gt=0)


class CacheConfig(BaseModel):
    """Per-node cache policy. Presence enables caching for the node."""

    ttl_s: float | None = Field(None, gt=0, description="None = lives for the session")


class NodePolicy(BaseModel):
    """Optional per-node execution controls. All fields default to off."""

    retry: RetryConfig | None = None
    cache: CacheConfig | None = None
    interrupt: bool | None = Field(
        None,
        description="When True, always pause before this node. When False, never pause. "
        "None = follow step_mode.",
    )


class ScriptRef(BaseModel):
    """User Python for an event script. Exactly one of code/file (mirrors script_node params)."""

    code: str | None = None
    file: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> ScriptRef:
        if bool(self.code) == bool(self.file):
            raise ValueError("ScriptRef requires exactly one of 'code' or 'file'")
        return self


class EventScript(BaseModel):
    """A script subscribed to one event type at one scope (node id, 'workflow', or '*')."""

    id: str
    name: str | None = None
    on: str
    scope: str = "*"
    script: ScriptRef
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def _id_is_identifier(cls, v: str) -> str:
        if not v or not v.isidentifier():
            raise ValueError("EventScript id must be a valid identifier (no spaces/dashes)")
        return v

    @field_validator("on")
    @classmethod
    def _on_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("EventScript 'on' must not be empty")
        return v


class NodeSpec(BaseModel):
    """A node in the workflow graph.

    ``kind`` selects the implementation from the node registry; ``prompt``
    overrides the default system prompt for that node; ``params`` carries
    node-specific config.
    """

    id: str = Field(..., description="Unique node id, e.g. 'planner'")
    kind: NodeKind = Field(..., description="Implementation to use")
    label: str | None = Field(None, description="Display label (defaults to kind)")
    prompt: str | None = Field(None, description="Optional system prompt override")
    params: dict[str, Any] = Field(default_factory=dict)
    policy: NodePolicy | None = None
    events: list[EventScript] = Field(
        default_factory=list,
        description="Scripts attached to this node's lifecycle (scope implicit = this node).",
    )

    @field_validator("id")
    @classmethod
    def id_no_spaces(cls, v: str) -> str:
        if not v or not v.isidentifier():
            raise ValueError("Node id must be a valid identifier (no spaces/dashes)")
        return v


class EdgeSpec(BaseModel):
    """An edge between two nodes. Edges with a ``condition`` become
    conditional (router) edges; without one they are plain transitions."""

    source: str
    target: str
    condition: str | None = Field(
        None,
        description="Router key, e.g. 'needs_fix'. If None, unconditional edge.",
    )


class IOField(BaseModel):
    """Input or output field definition for workflows and agents."""

    name: str
    type: Literal["string", "number", "boolean", "object", "array", "image", "file"]
    required: bool = True
    default: Any = None
    description: str | None = None


class AgentOutput(BaseModel):
    """Defines how an agent signals its decision via a routing key.

    ``mode`` selects the routing style:
    - ``"signal"`` (default): ``signal_field`` is read from the JSON reply and
      mapped to a declared edge ``condition``.
    - ``"goto"``: ``signal_field`` is read from the JSON reply and used directly
      as the ``Command(goto=...)`` target.
    """

    mode: Literal["signal", "goto"] = "signal"
    signal_field: str
    values: list[str]
    default: str

    @model_validator(mode="after")
    def _require_complete_routing(self) -> AgentOutput:
        self.signal_field = self.signal_field.strip()
        if not self.signal_field:
            raise ValueError("signal_field must not be empty")
        self.values = [v.strip() for v in self.values if v.strip()]
        if not self.values:
            raise ValueError("values must contain at least one non-empty signal")
        self.default = self.default.strip()
        if not self.default:
            raise ValueError("default must not be empty")
        return self


class SkillSpec(BaseModel):
    """A reusable block of instructions injected into an agent's system prompt."""

    id: str
    name: str
    description: str | None = None
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v


class HitlAnswer(BaseModel):
    """One scripted answer for a waiting hitl gate during a test-case run."""

    choice: str = ""
    comment: str = Field("", max_length=10_000)


class TestAssertion(BaseModel):
    """One pass/fail check evaluated against a finished test-case run.

    - ``node_count``: ``node`` ran exactly ``count`` times.
    - ``path``: ``sequence`` appears in node.start order.
    - ``state``: dotted ``path`` into the final state satisfies ``op``/``value``.
    """

    __test__ = False

    kind: Literal["node_count", "path", "state"]
    node: str | None = None
    count: int | None = None
    sequence: list[str] | None = None
    path: str | None = None
    op: Literal["exists", "equals", "contains"] | None = None
    value: str | None = None


class TestCaseSpec(BaseModel):
    """A saved dry-run scenario: inputs plus the expected outcome."""

    __test__ = False

    id: str
    name: str = ""
    task: str = ""
    mock_scripts: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Per-case dry-run replies; used INSTEAD of the spec-level map",
    )
    hitl_responses: dict[str, list[HitlAnswer]] = Field(
        default_factory=dict,
        description="Scripted gate answers per hitl node id, in ask order",
    )
    expect_status: str = "done"
    assertions: list[TestAssertion] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def id_no_spaces(cls, v: str) -> str:
        if not v or not v.isidentifier():
            raise ValueError("Test case id must be a valid identifier (no spaces/dashes)")
        return v


class StructuredOutputField(BaseModel):
    """One field in a structured output schema for an agent."""

    name: str
    type: Literal["string", "number", "boolean", "object", "array"] = "string"
    required: bool = True
    description: str | None = None


class StateField(BaseModel):
    """A custom state field declared in a WorkflowSpec.

    Allows workflows to extend AgentState with typed fields and custom
    reducers. Compiled into a dynamic TypedDict at build time.
    """

    name: str
    type: Literal["string", "number", "boolean", "object", "array", "list"] = "string"
    reducer: Literal[
        "default", "append", "merge", "union", "intersection", "last_value", "topic"
    ] = "default"
    default: Any = None
    description: str | None = None


class AgentSpec(BaseModel):
    """Specification for an agent node in the workflow."""

    id: str
    role: str = ""
    model: str | None = None
    provider: str | None = None
    temperature: float | None = Field(None, ge=0, le=2)
    max_tokens: int | None = Field(None, gt=0, le=200000)
    input_template: str = "{task}"
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    memory_scope: list[str] = Field(default_factory=lambda: ["global"])
    max_tool_turns: int = 8
    output: AgentOutput | None = None
    structured_output: list[StructuredOutputField] | None = Field(
        None,
        description="When set, the agent uses with_structured_output() to enforce "
        "a typed JSON response.",
    )
    result_key: str | None = None
    description: str | None = None
    workflow_ref: str | None = None
    agent_tools: list[str] = Field(default_factory=list)
    corpus_ids: list[str] = Field(default_factory=list)
    datasource_ids: list[str] = Field(default_factory=list)

    def gen_kwargs(self) -> dict:
        """Return only the generation params that are set, for build_llm(**kwargs).

        None values are omitted so the agent inherits the LangChain default.
        """
        kw: dict = {}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.max_tokens is not None:
            kw["max_tokens"] = self.max_tokens
        return kw

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v_lower = v.lower()
        from lgkit.llm import SUPPORTED_PROVIDERS

        if v_lower not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unknown provider '{v}'. Supported: {list(SUPPORTED_PROVIDERS)}")
        return v_lower


def agent_spec_warnings(spec: AgentSpec) -> list[str]:
    """Draft-tolerant lint for agents: list facade ignored fields and self-references."""
    warnings: list[str] = []
    if spec.workflow_ref:
        ignored = [
            name
            for name, off in (
                ("role", spec.role != ""),
                ("model", spec.model is not None),
                ("provider", spec.provider is not None),
                ("tools", bool(spec.tools)),
                ("skills", bool(spec.skills)),
                ("max_tool_turns", spec.max_tool_turns != 8),
                ("agent_tools", bool(spec.agent_tools)),
                ("corpus_ids", bool(spec.corpus_ids)),
                ("datasource_ids", bool(spec.datasource_ids)),
            )
            if off
        ]
        if ignored:
            warnings.append(f"workflow_ref is set; these fields are ignored: {', '.join(ignored)}")
    if not spec.workflow_ref and spec.id in spec.agent_tools:
        warnings.append(
            "agent_tools contains this agent itself; the self-reference is skipped at run time"
        )
    # Model not in the static catalog (non-litellm only). litellm models are probed, no catalog.
    if spec.provider and spec.model and spec.provider != "litellm":
        from lgkit.llm import DEFAULT_MODELS

        catalog = DEFAULT_MODELS.get(spec.provider, [])
        if spec.model not in catalog:
            warnings.append(
                f"model '{spec.model}' is not in the catalog for provider "
                f"'{spec.provider}' ({len(catalog)} known); it may be invalid."
            )
    return warnings


def agent_tool_warnings(spec: AgentSpec, resolver) -> list[str]:
    """Resolving lint for supervisors: nested agent_tools are ignored (depth 1)."""
    if spec.workflow_ref or not spec.agent_tools:
        return []
    nested = [
        tid
        for tid in spec.agent_tools
        if tid != spec.id and (sub := resolver(tid)) is not None and sub.agent_tools
    ]
    if not nested:
        return []
    return [
        f"agents called as tools run without their own agent_tools "
        f"(depth 1); affected: {', '.join(nested)}"
    ]


class WorkflowSpec(BaseModel):
    """A complete workflow definition sent from the frontend editor."""

    nodes: list[NodeSpec] = Field(..., min_length=1)
    edges: list[EdgeSpec] = Field(default_factory=list)
    entry: str = Field(..., description="Node id where execution starts")
    name: str = "custom"
    description: str | None = None
    inputs: list[IOField] = Field(default_factory=list)
    outputs: list[IOField] = Field(default_factory=list)
    memory_scope: list[str] = Field(
        default_factory=lambda: ["global"],
        description="Store namespace for memory nodes.",
    )
    mock_scripts: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Dry-run replies per node id. Ignored by the graph builder.",
    )
    test_cases: list[TestCaseSpec] = Field(
        default_factory=list,
        description="Saved dry-run test scenarios. Ignored by the graph builder.",
    )
    state_fields: list[StateField] = Field(
        default_factory=list,
        description="Custom state fields that extend AgentState.",
    )
    events: list[EventScript] = Field(
        default_factory=list,
        description="Workflow-scoped lifecycle hooks + global custom-event subscriptions.",
    )

    @field_validator("nodes")
    @classmethod
    def unique_ids(cls, nodes: list[NodeSpec]) -> list[NodeSpec]:
        ids = [n.id for n in nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Node ids must be unique")
        return nodes

    @field_validator("entry")
    @classmethod
    def entry_exists(cls, entry: str, info) -> str:
        nodes = info.data.get("nodes", [])
        ids = {n.id for n in nodes}
        if entry not in ids:
            raise ValueError(f"Entry node '{entry}' not in nodes")
        return entry

    def validate_inputs(self, values: dict) -> dict:
        """Apply declared input defaults and check required inputs are present."""
        out: dict = {}
        missing: list[str] = []
        for f in self.inputs:
            if f.name in values:
                out[f.name] = values[f.name]
            elif not f.required:
                out[f.name] = f.default
            else:
                missing.append(f.name)
        if missing:
            raise ValueError(f"Missing required inputs: {', '.join(missing)}")
        for k, v in values.items():
            out.setdefault(k, v)
        return out

    def node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}

    def _plain_targets_by_source(self) -> dict[str, list[str]]:
        """Map each source node id to its unconditional (plain) edge targets."""
        out: dict[str, list[str]] = {}
        for e in self.edges:
            if e.condition is None:
                out.setdefault(e.source, []).append(e.target)
        return out

    def _goto_targets(self, node: NodeSpec) -> list[str]:
        """Possible goto targets for a node in goto mode, else []."""
        p = node.params or {}
        if node.kind == "script":
            t = p.get("goto_targets")
            return list(t) if isinstance(t, list) else []
        if node.kind == "hitl" and p.get("route_mode") == "goto":
            c = p.get("choices")
            return list(c) if isinstance(c, list) else []
        return []

    def validate_edges(self) -> list[str]:
        """Return a list of error messages (empty if OK)."""
        ids = self.node_ids()
        errors: list[str] = []
        for e in self.edges:
            if e.source not in ids:
                errors.append(f"Edge source '{e.source}' is not a node")
            if e.target not in ids and e.target != "END":
                errors.append(f"Edge target '{e.target}' is not a node or 'END'")

        # --- Parallel fan-out: fork -> agent branches -> shared join ---
        plain_targets = self._plain_targets_by_source()
        node_kind = {n.id: n.kind for n in self.nodes}
        for src, targets in plain_targets.items():
            if len(targets) < 2:
                continue
            joins: set[str] = set()
            for b in targets:
                if b == "END":
                    errors.append(f"Fork '{src}' cannot branch directly to END")
                    continue
                if node_kind.get(b) not in ("agent", "react_agent"):
                    errors.append(
                        f"Parallel branch '{b}' must be an 'agent' or 'react_agent' node; "
                        f"got '{node_kind.get(b)}'"
                    )
                branch_out = [e for e in self.edges if e.source == b]
                if any(e.condition for e in branch_out):
                    errors.append(f"Parallel branch '{b}' must not use conditional edges")
                branch_plain = [e for e in branch_out if e.condition is None]
                if len(branch_plain) != 1:
                    errors.append(
                        f"Parallel branch '{b}' must have exactly one outgoing edge "
                        f"to the join node (has {len(branch_plain)})"
                    )
                    continue
                joins.add(branch_plain[0].target)
            if len(joins) > 1:
                errors.append(
                    f"Fork '{src}' branches must converge on one join node; found {sorted(joins)}"
                )
            if "END" in joins:
                errors.append(f"Fork '{src}' branches must converge on a join node, not END")

        # --- Dynamic map node ---
        for n in self.nodes:
            if n.kind != "map":
                continue
            p = n.params or {}
            for req in ("over", "agent", "result_key"):
                if not isinstance(p.get(req), str) or not p.get(req):
                    errors.append(f"Map node '{n.id}' requires a non-empty '{req}' param")
            out_edges = [e for e in self.edges if e.source == n.id]
            if any(e.condition for e in out_edges):
                errors.append(f"Map node '{n.id}' must not have conditional outgoing edges")
            plain = [e for e in out_edges if e.condition is None]
            if len(plain) != 1:
                errors.append(
                    f"Map node '{n.id}' must have exactly one outgoing edge (has {len(plain)})"
                )

        # --- HITL gates ---
        for n in self.nodes:
            if n.kind != "hitl":
                continue
            params = n.params or {}
            if params.get("route_mode") == "goto":
                continue
            msg = params.get("message")
            if not isinstance(msg, str) or not msg.strip():
                errors.append(f"HITL node '{n.id}' requires a non-empty 'message'")
            choices = params.get("choices")
            ok_list = (
                isinstance(choices, list)
                and bool(choices)
                and all(isinstance(c, str) and c.strip() for c in choices)
            )
            if not ok_list:
                errors.append(f"HITL node '{n.id}' requires a non-empty 'choices' list of strings")
            elif len(set(choices)) != len(choices):
                errors.append(f"HITL node '{n.id}' has duplicate choices")
            out = [e for e in self.edges if e.source == n.id]
            if not out:
                errors.append(f"HITL node '{n.id}' needs at least one outgoing edge")
            if ok_list:
                for e in out:
                    if e.condition and e.condition not in choices:
                        errors.append(
                            f"HITL node '{n.id}' edge condition '{e.condition}' is not one of its choices"
                        )

        # --- Goto-mode nodes ---
        for n in self.nodes:
            goto_targets = self._goto_targets(n)
            if not goto_targets:
                continue
            for t in goto_targets:
                if t not in ids and t != "END":
                    errors.append(
                        f"Goto node '{n.id}' possible target '{t}' is not a node or 'END'"
                    )
            out = [e for e in self.edges if e.source == n.id]
            if out:
                errors.append(
                    f"Goto node '{n.id}' must not have outgoing edges "
                    f"(it routes dynamically); remove its {len(out)} edge(s)"
                )

        # --- Memory nodes ---
        for n in self.nodes:
            if n.kind != "memory":
                continue
            p = n.params or {}
            mode = p.get("mode")
            if mode not in ("read", "write"):
                errors.append(
                    f"Memory node '{n.id}' requires 'mode' in ('read','write'); got {mode!r}"
                )
            key = p.get("key")
            if not isinstance(key, str) or not key:
                errors.append(f"Memory node '{n.id}' requires a non-empty 'key' param")
            ns = p.get("namespace")
            if not isinstance(ns, list) or not ns:
                errors.append(f"Memory node '{n.id}' requires a non-empty 'namespace' list")
            else:
                try:
                    namespace_from_scope(ns)
                except ValueError as exc:
                    errors.append(f"Memory node '{n.id}' has invalid namespace: {exc}")
            if mode == "read":
                rk = p.get("result_key")
                if not isinstance(rk, str) or not rk:
                    errors.append(
                        f"Memory node '{n.id}' (read mode) requires a non-empty 'result_key'"
                    )
            elif mode == "write":
                vp = p.get("value_path")
                if not isinstance(vp, str) or not vp:
                    errors.append(
                        f"Memory node '{n.id}' (write mode) requires a non-empty 'value_path'"
                    )
        return errors

    def validate_events(self) -> list[str]:
        """Hard errors for event scripts (empty list = OK).

        Custom-event subscriptions to types that nobody emits are NOT errors
        (statically unknowable).
        """
        ids = self.node_ids()
        errors: list[str] = []
        node_lifecycle = {"node.on_enter", "node.on_exit", "node.on_error"}
        wf_lifecycle = {"workflow.on_start", "workflow.on_complete", "workflow.on_error"}

        for n in self.nodes:
            seen: set[str] = set()
            for es in n.events:
                if es.id in seen:
                    errors.append(f"Node '{n.id}' has duplicate event id '{es.id}'")
                seen.add(es.id)
                if es.on not in node_lifecycle:
                    errors.append(
                        f"Node '{n.id}' event '{es.id}' on='{es.on}' is not a node lifecycle type "
                        f"(allowed: {sorted(node_lifecycle)})"
                    )

        seen_wf: set[str] = set()
        for es in self.events:
            if es.id in seen_wf:
                errors.append(f"Workflow has duplicate event id '{es.id}'")
            seen_wf.add(es.id)
            if es.on in wf_lifecycle and es.scope != "workflow":
                errors.append(
                    f"Workflow event '{es.id}' on='{es.on}' requires scope='workflow' "
                    f"(got '{es.scope}')"
                )
            if es.scope not in ("workflow", "*") and es.scope not in ids:
                errors.append(
                    f"Workflow event '{es.id}' scope='{es.scope}' is not a known node id, "
                    f"'*', or 'workflow'"
                )
        return errors

    def _reachable_subagent_ids(self, agent_resolver) -> set[str]:
        """Sub-agent ids reachable one level via agent_tools of any agent."""
        if agent_resolver is None:
            return set()
        ids: set[str] = set()
        for n in self.nodes:
            if n.kind not in ("agent", "react_agent", "map"):
                continue
            aid = (n.params or {}).get("agent")
            if not isinstance(aid, str):
                continue
            spec = agent_resolver(aid)
            if spec is None:
                continue
            ids.update(spec.agent_tools)
        return ids

    def mock_script_warnings(self, agent_resolver=None) -> list[str]:
        known = self.node_ids() | self._reachable_subagent_ids(agent_resolver)
        return [
            f"mock_scripts references unknown node '{k}'"
            for k in sorted(self.mock_scripts)
            if k not in known
        ]

    def hitl_choice_warnings(self) -> list[str]:
        warnings: list[str] = []
        for n in self.nodes:
            if n.kind != "hitl":
                continue
            choices = (n.params or {}).get("choices") or []
            if not isinstance(choices, list):
                continue
            if any(e.source == n.id and not e.condition for e in self.edges):
                continue
            conds = {e.condition for e in self.edges if e.source == n.id and e.condition}
            for c in choices:
                if isinstance(c, str) and c not in conds:
                    warnings.append(f"HITL node '{n.id}' choice '{c}' has no matching edge")
        return warnings

    def test_case_warnings(self, agent_resolver=None) -> list[str]:
        ids = self.node_ids()
        known_script_keys = ids | self._reachable_subagent_ids(agent_resolver)
        gate_choices: dict[str, list] = {}
        for n in self.nodes:
            if n.kind not in {"hitl", "approval"}:
                continue
            choices = (n.params or {}).get("choices") or []
            gate_choices[n.id] = choices if isinstance(choices, list) else []
        warnings: list[str] = []
        seen: set[str] = set()
        for case in self.test_cases:
            if case.id in seen:
                warnings.append(f"duplicate test case id '{case.id}'")
            seen.add(case.id)
            label = f"test case '{case.id}'"
            if not case.task.strip():
                warnings.append(f"{label} has no task input")
            if case.expect_status not in {"done", "failed", "max_iterations", "waiting"}:
                warnings.append(
                    f"{label} expect_status '{case.expect_status}' is not a known status"
                )
            for k in sorted(case.mock_scripts):
                if k not in known_script_keys:
                    warnings.append(f"{label} mock_scripts references unknown node '{k}'")
            for gate, answers in sorted(case.hitl_responses.items()):
                if gate not in gate_choices:
                    warnings.append(f"{label} hitl_responses references unknown gate node '{gate}'")
                    continue
                for a in answers:
                    if a.choice not in gate_choices[gate]:
                        warnings.append(
                            f"{label} answer '{a.choice}' is not a choice of gate node '{gate}'"
                        )
            for a in case.assertions:
                if a.kind == "node_count":
                    if not a.node or a.count is None or a.count < 0:
                        warnings.append(
                            f"{label} assertion is incomplete (node_count needs node and count >= 0)"
                        )
                    elif a.node not in ids:
                        warnings.append(f"{label} assertion references unknown node '{a.node}'")
                elif a.kind == "path":
                    if not a.sequence:
                        warnings.append(
                            f"{label} assertion is incomplete (path needs a node sequence)"
                        )
                    else:
                        for n in a.sequence:
                            if n not in ids:
                                warnings.append(f"{label} assertion references unknown node '{n}'")
                else:  # state
                    if not a.path or a.op is None:
                        warnings.append(
                            f"{label} assertion is incomplete (state needs path and op)"
                        )
                    elif a.op in {"equals", "contains"} and a.value is None:
                        warnings.append(f"{label} assertion is incomplete ('{a.op}' needs a value)")
        return warnings


__all__ = [
    "NodeKind",
    "Provider",
    "namespace_from_scope",
    "RetryConfig",
    "CacheConfig",
    "NodePolicy",
    "ScriptRef",
    "EventScript",
    "NodeSpec",
    "EdgeSpec",
    "IOField",
    "AgentOutput",
    "SkillSpec",
    "HitlAnswer",
    "TestAssertion",
    "TestCaseSpec",
    "StructuredOutputField",
    "StateField",
    "AgentSpec",
    "agent_spec_warnings",
    "agent_tool_warnings",
    "WorkflowSpec",
]
