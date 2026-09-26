# lgkit Extraction + `approval_gate` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move lgtools' spec→StateGraph kernel into a UI-free library `lgkit` (in `langgraph-agent`), make lgtools import it back through shims, and ship the first pattern `approval_gate` (human / human+advice / agent-with-escalation).

**Architecture:** `lgkit` owns spec models, node registry, state, run context, events, the graph builder and an LLM factory. Everything the builder used to import from lgtools (agent/workflow resolvers, event-script dispatch, the resolved unit) becomes a `BuildHooks` value or a `resolved=` argument. lgtools keeps its old module paths as shims — pure modules are aliased through `sys.modules` so private state (`_REGISTRY`, `_trace_path`) and monkeypatch targets stay the same object. `approval_gate()` returns plain spec data (a `Fragment`), compiled by the same builder.

**Tech Stack:** Python ≥3.12 (dev runs 3.14.7), LangGraph 1.2.x, langchain-core ≥1.0, Pydantic 2, uv, pytest, hatchling.

**Spec:** `docs/superpowers/specs/2026-09-26-lgkit-approval-gate-design.md`

**Paths used below:**
- `$LGA` = `/Users/ekayut/Project/product/langgraph-agent` (this repo, package `lgkit`)
- `$LGT` = `/Users/ekayut/Project/product/laggraph-hermes/backend` (lgtools; git root is `$LGT/..`)

## Deviations from the spec (decided while planning)

1. **Fake LLM:** `GenericFakeChatModel` has no `with_structured_output`, so tests and the demo use `lgkit.testing.ScriptedLLM` (Task 5).
2. **Advisor retry:** the advisor never raises (spec §3.3), so LangGraph `NodePolicy.retry` would never fire. The advisor retries internally with a `max_attempts` param (default 2).
3. **`BuildHooks` has no `resolve` field:** lgtools' shim computes `resolved` itself and passes `build_graph(..., resolved=...)`. Standalone lgkit passes `None`.
4. **Built-in kinds register lazily:** `build_graph` calls `lgkit.nodes.ensure_registered()`, which only adds `hitl`/`advisor`/`approval` if absent, so lgtools' own `hitl` registration (whose `fn.__module__` the frontend contract test filters on) is never overridden.
5. **lgtools `hitl_node.run` stays a thin wrapper** defined in `lgtools.templates.primitives.hitl_node` for the same contract-test reason.

## Global Constraints

- `requires-python = ">=3.12"` for `langgraph-agent`; lgtools already declares `>=3.12`.
- `lgkit` runtime deps: `langgraph>=1.2.0`, `langchain-core>=1.0`, `pydantic>=2.7` — nothing else. Provider SDKs are optional extras.
- `lgkit` must never import `lgtools` (check: `grep -rn lgtools $LGA/src` returns nothing).
- No module-level mutable config in lgkit (no `configure()`); hooks are passed per `build_graph` call.
- Approval semantics: when unsure, ask a human — never auto-approve on error, low confidence, `escalate`, or invalid LLM output.
- Nothing with side effects may run before `interrupt()` inside a node.
- Every lgtools test result after migration must equal the baseline recorded in Task 9.
- Commits end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb
  ```
- lgtools work happens on branch `feat/use-lgkit` in `$LGT/..`; `langgraph-agent` work on branch `feat/lgkit` in `$LGA`.

## Review Focus

1. **Monkeypatching lgtools paths after the move** — `monkeypatch.setattr("lgtools.kernel.event_dispatch.dispatch_event", ...)` or patching `lgtools.kernel.registry._REGISTRY` must still affect graph builds. Expectation: identical behavior. Pinned by `test_lgkit_shims.py` in Task 11 (hooks resolve their targets at call time; aliased modules are identical objects).
2. **Human resumes with a bare string** — `Command(resume="approve")` instead of `{"choice": "approve"}`. Expectation: accepted as the choice. Pinned in Task 6.
3. **Malformed message template** — `message="Approve {draft"` (unbalanced brace). Expectation: `ValueError` from `approval_gate()` at build time, not a crash mid-run. Pinned in Task 7.
4. **Confidence exactly at the threshold** — `confidence == min_confidence`. Expectation: the agent decides (`>=`). Pinned in Task 6.
5. **Gate reached twice in a revise loop** — second pass must use fresh advice, not the first round's. Pinned in Task 7.

---

## Part A — `lgkit` in `langgraph-agent`

### Task 1: Package scaffold

**Files:**
- Modify: `$LGA/pyproject.toml`, `$LGA/.gitignore`, `$LGA/README.md`
- Create: `$LGA/src/lgkit/__init__.py`, `$LGA/tests/test_smoke.py`

**Interfaces:**
- Produces: importable package `lgkit` with `lgkit.__version__ == "0.1.0"`; `uv run pytest` works from `$LGA`.

- [ ] **Step 1: Commit the existing untracked skeleton as-is, then branch**

```bash
cd $LGA
git add .gitignore .python-version README.md main.py pyproject.toml uv.lock
git commit -m "chore: initial langgraph-agent skeleton (as found)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
git switch -c feat/lgkit
```

- [ ] **Step 2: Write the failing smoke test** — `tests/test_smoke.py`

```python
def test_package_imports():
    import lgkit

    assert lgkit.__version__ == "0.1.0"
```

- [ ] **Step 3: Replace `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "langgraph-agent"
version = "0.1.0"
description = "lgkit: reusable LangGraph workflow kernel (spec -> StateGraph) and workflow patterns"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=1.2.0",
    "langchain-core>=1.0",
    "pydantic>=2.7",
]

[project.optional-dependencies]
openai = ["langchain-openai>=0.2.0"]
anthropic = ["langchain-anthropic>=0.2.0"]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.8.0",
    "langchain-openai>=0.2.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/lgkit"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP"]
ignore = ["E501"]
```

- [ ] **Step 4: Create `src/lgkit/__init__.py`**

```python
"""lgkit — build LangGraph workflows from plain spec data, plus ready-made patterns."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Append to `.gitignore`**

```
# Local env / OS
.env
.DS_Store
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 6: Sync and run**

Run: `cd $LGA && uv sync && uv run pytest -q`
Expected: `1 passed`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/lgkit/__init__.py tests/test_smoke.py
git commit -m "build: turn langgraph-agent into the lgkit package

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 2: Pure kernel modules (state, registry, context, events, node_validation)

**Files:**
- Create: `$LGA/src/lgkit/{state,registry,context,events,node_validation}.py` (copied from `$LGT/src/lgtools/kernel/`)
- Create: `$LGA/tests/conftest.py`
- Test: `$LGA/tests/{test_registry,test_kernel_state,test_context}.py` (ported)

**Interfaces:**
- Produces (same names/signatures as lgtools):
  - `lgkit.state`: `AgentState`, `add_events`, `merge_scratch`, `union_reducer`, `intersection_reducer`, `append_unique`, `get_reducer(name)`
  - `lgkit.registry`: `NodeDef`, `register(kind, def_)`, `get_node_def(kind)`, `all_node_defs()`, `node_palette()`, private `_REGISTRY`
  - `lgkit.context`: `Budget(max_iterations, recursion_limit)`, `RunContext(session_id, workspace, llm, budget, dry_run=None)`, `current_ctx()`, `use_ctx(ctx) -> Token`, `clear_ctx(token)`, `get_resolved_workflow`, `get_resolved_agent`
  - `lgkit.events`: `emit_event(event_type, node=None, payload=None, path=None)`, `push_trace_frame(node_id)`, `_trace_path`
  - `lgkit.node_validation`: `validate_node_params(spec)`, `fatal_node_param_errors(spec)`

- [ ] **Step 1: Port the tests first**

```bash
cd $LGA
for f in test_registry test_kernel_state test_context; do
  sed -e 's/lgtools\.kernel\./lgkit./g' $LGT/tests/$f.py > tests/$f.py
done
grep -n lgtools tests/test_registry.py tests/test_kernel_state.py tests/test_context.py
```
Expected: the grep prints nothing.

- [ ] **Step 2: Add the registry-isolation fixture** — `tests/conftest.py`

```python
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
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_registry.py tests/test_kernel_state.py tests/test_context.py -q`
Expected: collection errors, `ModuleNotFoundError: No module named 'lgkit.registry'` (and siblings).

- [ ] **Step 4: Copy the modules and rewrite imports**

```bash
for m in state registry context events node_validation; do
  sed -e 's/lgtools\.kernel\.schemas/lgkit.spec/g' -e 's/lgtools\.kernel\./lgkit./g' \
    $LGT/src/lgtools/kernel/$m.py > src/lgkit/$m.py
done
grep -rn lgtools src/lgkit
```
Expected: the grep prints nothing. (`context.py`'s `TYPE_CHECKING` import now points at `lgkit.spec`, created in Task 3; it is type-only so nothing fails at runtime.)

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`
Expected: all pass (1 smoke + 5 registry + 8 state + 5 context = 19).

- [ ] **Step 6: Commit**

```bash
git add src/lgkit tests
git commit -m "feat(lgkit): move pure kernel modules from lgtools

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 3: Spec models and LLM factory

**Files:**
- Create: `$LGA/src/lgkit/spec.py` (lines 1–780 of `$LGT/src/lgtools/kernel/schemas.py`, edited)
- Create: `$LGA/src/lgkit/llm.py`
- Test: `$LGA/tests/test_schemas.py`, `$LGA/tests/test_event_schemas.py` (ported), `$LGA/tests/test_llm.py` (new)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `lgkit.spec`: every model/function lgtools had above the "API request/response models" divider — `NodeKind` (now incl. `"advisor"`, `"approval"`), `Provider`, `namespace_from_scope`, `RetryConfig`, `CacheConfig`, `NodePolicy`, `ScriptRef`, `EventScript`, `NodeSpec`, `EdgeSpec`, `IOField`, `AgentOutput`, `SkillSpec`, `HitlAnswer`, `TestAssertion`, `TestCaseSpec`, `StructuredOutputField`, `StateField`, `AgentSpec`, `agent_spec_warnings`, `agent_tool_warnings`, `WorkflowSpec`; `__all__` lists them all.
  - `lgkit.llm`: `SUPPORTED_PROVIDERS: tuple[str, ...]`, `PROVIDER_ENV_KEYS: dict[str, str]`, `DEFAULT_MODELS: dict[str, list[str]]`, `LLMConfigError`, `build_llm(provider=None, model=None, *, api_key=<env>, base_url=None, **kwargs) -> BaseChatModel`.

- [ ] **Step 1: Port the spec tests and write the LLM tests**

```bash
cd $LGA
for f in test_schemas test_event_schemas; do
  sed -e 's/lgtools\.kernel\.schemas/lgkit.spec/g' $LGT/tests/$f.py > tests/$f.py
done
```

`tests/test_llm.py`:

```python
from typing import get_args

import pytest

from lgkit.llm import DEFAULT_MODELS, SUPPORTED_PROVIDERS, LLMConfigError, build_llm
from lgkit.spec import NodeSpec, Provider


def test_supported_providers_match_spec_literal():
    assert set(get_args(Provider)) == set(SUPPORTED_PROVIDERS)


def test_unknown_provider_raises():
    with pytest.raises(LLMConfigError, match="Unknown provider"):
        build_llm(provider="nope_xyz")


def test_missing_key_from_env_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        build_llm(provider="openai")


def test_explicit_none_key_does_not_fall_back_to_env(monkeypatch):
    # lgtools passes its Settings' key explicitly (None when a placeholder was
    # rejected); lgkit must not silently pick the raw env var up instead.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        build_llm(provider="openai", api_key=None)


def test_default_model_and_base_url():
    llm = build_llm(provider="openai", api_key="sk-test", base_url="http://local:8000/v1")
    assert llm.model_name == DEFAULT_MODELS["openai"][0]
    assert llm.openai_api_base == "http://local:8000/v1"


def test_litellm_defaults():
    llm = build_llm(provider="litellm", model="GLM-5.2", api_key=None)
    assert llm.disable_streaming == "tool_calling"
    assert llm.openai_api_base == "http://localhost:4000"


def test_new_node_kinds_accepted():
    assert NodeSpec(id="g", kind="approval").kind == "approval"
    assert NodeSpec(id="a", kind="advisor").kind == "advisor"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_schemas.py tests/test_event_schemas.py tests/test_llm.py -q`
Expected: collection errors, `No module named 'lgkit.spec'` / `'lgkit.llm'`.

- [ ] **Step 3: Create `src/lgkit/llm.py`**

```python
"""LLM factory supporting multiple providers.

Providers are imported lazily so only the chosen provider's package needs to
be installed. Keys come from the ``api_key`` argument when given (even ``None``)
and from the provider's environment variable only when it is omitted.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

SUPPORTED_PROVIDERS: tuple[str, ...] = (
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
)

PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "meta": "META_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "zai": "ZAI_API_KEY",
    "litellm": "LITELLM_API_KEY",
}

DEFAULT_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "o3-mini", "o3", "o4-mini"],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
    "google": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "xai": ["grok-3", "grok-3-mini"],
    "meta": ["llama-4-maverick", "llama-4-scout", "llama-3.3-70b"],
    "mistral": ["mistral-large", "mistral-medium", "mistral-small", "codestral"],
    "cohere": ["command-r-plus", "command-r"],
    "zai": ["glm-4.5-air", "glm-4.5", "glm-4-plus"],
    "litellm": [],
}


class LLMConfigError(RuntimeError):
    """Raised when the LLM cannot be constructed (missing key/package)."""


_ENV = object()  # sentinel: "api_key not passed, read the provider's env var"


def _require_key(provider: str, api_key: str | None) -> str:
    if not api_key:
        raise LLMConfigError(f"{PROVIDER_ENV_KEYS[provider]} is not set")
    return api_key


def _chat_openai(**kwargs: Any) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise LLMConfigError(
            "langchain-openai is not installed. Run: uv add langchain-openai"
        ) from e
    return ChatOpenAI(**kwargs)


def _openai_compatible(provider: str, default_base_url: str | None):
    def build(model: str, api_key: str | None, base_url: str | None, **kw: Any) -> BaseChatModel:
        url = base_url or default_base_url
        extra = {"base_url": url} if url else {}
        return _chat_openai(model=model, api_key=_require_key(provider, api_key), **extra, **kw)

    return build


def _build_anthropic(model: str, api_key: str | None, base_url: str | None, **kw: Any) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise LLMConfigError(
            "langchain-anthropic is not installed. Run: uv add langchain-anthropic"
        ) from e
    return ChatAnthropic(model=model, api_key=_require_key("anthropic", api_key), **kw)


def _build_google(model: str, api_key: str | None, base_url: str | None, **kw: Any) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise LLMConfigError("langchain-google-genai is not installed.") from e
    return ChatGoogleGenerativeAI(model=model, api_key=_require_key("google", api_key), **kw)


def _build_mistral(model: str, api_key: str | None, base_url: str | None, **kw: Any) -> BaseChatModel:
    try:
        from langchain_mistralai import ChatMistralAI
    except ImportError as e:
        raise LLMConfigError("langchain-mistralai is not installed") from e
    return ChatMistralAI(model=model, api_key=_require_key("mistral", api_key), **kw)


def _build_cohere(model: str, api_key: str | None, base_url: str | None, **kw: Any) -> BaseChatModel:
    try:
        from langchain_cohere import ChatCohere
    except ImportError as e:
        raise LLMConfigError("langchain-cohere is not installed") from e
    return ChatCohere(model=model, api_key=_require_key("cohere", api_key), **kw)


def _build_litellm(model: str, api_key: str | None, base_url: str | None, **kw: Any) -> BaseChatModel:
    """LiteLLM proxy (OpenAI-compatible).

    Defaults kept from lgtools: ``enable_thinking=False`` so reasoning models
    (e.g. GLM-5.2) return ``content`` instead of only ``reasoning_content``, and
    ``disable_streaming="tool_calling"`` because that gateway drops tool calls
    from streamed responses. Callers can override both.
    """
    if not model:
        raise LLMConfigError("LiteLLM requires a model name")
    extra_body = kw.pop("extra_body", None) or {"chat_template_kwargs": {"enable_thinking": False}}
    kw.setdefault("disable_streaming", "tool_calling")
    return _chat_openai(
        model=model,
        api_key=api_key or "sk-anything",
        base_url=base_url or "http://localhost:4000",
        extra_body=extra_body,
        **kw,
    )


_BUILDERS: dict[str, Any] = {
    "openai": _openai_compatible("openai", None),
    "anthropic": _build_anthropic,
    "google": _build_google,
    "gemini": _build_google,
    "deepseek": _openai_compatible("deepseek", "https://api.deepseek.com/v1"),
    "xai": _openai_compatible("xai", "https://api.x.ai/v1"),
    "meta": _openai_compatible("meta", "https://api.together.xyz/v1"),
    "mistral": _build_mistral,
    "cohere": _build_cohere,
    "zai": _openai_compatible("zai", "https://api.z.ai/api/paas/v4"),
    "litellm": _build_litellm,
}


def build_llm(
    provider: str | None = None,
    model: str | None = None,
    *,
    api_key: Any = _ENV,
    base_url: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Build a chat model. ``provider`` defaults to ``$LGKIT_PROVIDER`` or openai."""
    provider = (provider or os.environ.get("LGKIT_PROVIDER") or "openai").lower()
    builder = _BUILDERS.get(provider)
    if builder is None:
        raise LLMConfigError(f"Unknown provider '{provider}'. Supported: {list(_BUILDERS)}")
    if api_key is _ENV:
        api_key = os.environ.get(PROVIDER_ENV_KEYS.get(provider, ""))
    catalog = DEFAULT_MODELS.get(provider) or [""]
    return builder(model or catalog[0], api_key, base_url, **kwargs)


__all__ = [
    "SUPPORTED_PROVIDERS",
    "PROVIDER_ENV_KEYS",
    "DEFAULT_MODELS",
    "LLMConfigError",
    "build_llm",
]
```

- [ ] **Step 4: Create `src/lgkit/spec.py` from lgtools' schemas**

```bash
sed -n '1,780p' $LGT/src/lgtools/kernel/schemas.py > src/lgkit/spec.py
```

Then make exactly these edits in `src/lgkit/spec.py`:

a) Replace the module docstring (lines 1–6) with:
```python
"""Workflow and agent spec models.

A workflow is a list of nodes + edges; :func:`lgkit.builder.build_graph` turns a
:class:`WorkflowSpec` into a runnable LangGraph ``StateGraph``.
"""
```
b) Delete the line `from lgtools.api_models import StrictRequest`.
c) In `NodeKind`, after `"harness",` add:
```python
    "advisor",
    "approval",
```
d) Replace `from lgtools.config import SUPPORTED_PROVIDERS` with `from lgkit.llm import SUPPORTED_PROVIDERS`.
e) Replace `from lgtools.llm import DEFAULT_MODELS` with `from lgkit.llm import DEFAULT_MODELS`.
f) Append:
```python


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
```

Check: `grep -n "lgtools\|StrictRequest" src/lgkit/spec.py` prints nothing.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`
Expected: all pass (19 + 16 schemas + 8 event_schemas + 7 llm = 50).

- [ ] **Step 6: Commit**

```bash
git add src/lgkit/spec.py src/lgkit/llm.py tests/test_schemas.py tests/test_event_schemas.py tests/test_llm.py
git commit -m "feat(lgkit): spec models and provider-agnostic LLM factory

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 4: `BuildHooks` and the graph builder

**Files:**
- Create: `$LGA/src/lgkit/hooks.py`
- Create: `$LGA/src/lgkit/builder.py` (copy of `$LGT/src/lgtools/kernel/graph_builder.py`, edited)
- Test: `$LGA/tests/test_graph_builder.py` (ported), `$LGA/tests/test_builder_hooks.py` (new)

**Interfaces:**
- Consumes: `lgkit.spec.{NodeSpec, WorkflowSpec}`, `lgkit.registry.{NodeDef, get_node_def}`, `lgkit.state.{AgentState, get_reducer}`, `lgkit.context.current_ctx`, `lgkit.events.emit_event`, `lgkit.node_validation.fatal_node_param_errors`.
- Produces:
  - `lgkit.hooks.BuildHooks(agent_resolver, workflow_resolver, dispatch_event)` (frozen dataclass), `lgkit.hooks.DEFAULT_HOOKS`
  - `lgkit.builder.build_graph(spec: WorkflowSpec, checkpointer=None, store=None, dry_run=False, *, hooks: BuildHooks = DEFAULT_HOOKS, resolved: Any = None)` → compiled graph
  - `lgkit.builder.{GraphBuildError, detect_workflow_cycle, validate_goto_agents, validate_map_nodes, validate_parallel_branches, _iteration_guard}` — same behavior as lgtools.
  - Node fns keep the signature `fn(state, params, prompt, ctx, resolved) -> dict`.

- [ ] **Step 1: Port the builder test and write the hooks tests**

```bash
cd $LGA
sed -e 's/lgtools\.kernel\.schemas/lgkit.spec/g' -e 's/lgtools\.kernel\.graph_builder/lgkit.builder/g' \
    -e 's/lgtools\.kernel\./lgkit./g' $LGT/tests/test_graph_builder.py > tests/test_graph_builder.py
```

`tests/test_builder_hooks.py`:

```python
import pytest

from lgkit.builder import GraphBuildError, build_graph
from lgkit.hooks import BuildHooks
from lgkit.registry import NodeDef, register
from lgkit.spec import EventScript, NodeSpec, ScriptRef, WorkflowSpec

SEEN: dict = {}


def _probe(state, params, prompt, ctx=None, resolved=None):
    SEEN["resolved"] = resolved
    return {"signal": "done"}


@pytest.fixture(autouse=True)
def _probe_kind():
    SEEN.clear()
    register("custom", NodeDef(kind="custom", fn=_probe, default_prompt="", description="probe"))


def _one_node(**node_kw) -> WorkflowSpec:
    return WorkflowSpec(nodes=[NodeSpec(id="n", kind="custom", **node_kw)], entry="n")


def test_default_hooks_skip_event_scripts():
    # A valid on_enter script would need lgtools' subprocess runtime; with the
    # default hooks it must simply not run.
    ev = EventScript(id="e1", on="node.on_enter", script=ScriptRef(code="raise SystemExit(1)"))
    app = build_graph(_one_node(events=[ev]))
    assert app.invoke({"task": "t"})["signal"] == "done"


def test_dispatch_hook_sees_lifecycle_events():
    calls: list[str] = []

    def dispatch(event_type, state, **kw):
        calls.append(event_type)
        return {}

    app = build_graph(_one_node(), hooks=BuildHooks(dispatch_event=dispatch))
    app.invoke({"task": "t"})
    assert calls == ["node.on_enter", "node.on_exit"]


def test_dispatch_hook_route_becomes_goto():
    register("custom", NodeDef(kind="custom", fn=_probe, default_prompt="", description="p"))
    spec = WorkflowSpec(
        nodes=[NodeSpec(id="a", kind="custom"), NodeSpec(id="b", kind="custom")],
        entry="a",
    )

    def dispatch(event_type, state, **kw):
        return {"route": "b"} if (event_type == "node.on_exit" and kw["node_id"] == "a") else {}

    app = build_graph(spec, hooks=BuildHooks(dispatch_event=dispatch))
    app.invoke({"task": "t"})  # would raise if goto 'b' were not honored as a node


def test_resolved_is_passed_to_node_fns():
    marker = object()
    build_graph(_one_node(), resolved=marker).invoke({"task": "t"})
    assert SEEN["resolved"] is marker


def test_workflow_resolver_drives_cycle_detection():
    looping = WorkflowSpec(
        nodes=[NodeSpec(id="s", kind="subworkflow", params={"ref": "a"})], entry="s"
    )
    hooks = BuildHooks(workflow_resolver=lambda ref: looping if ref == "a" else None)
    with pytest.raises(GraphBuildError, match="cycle"):
        build_graph(looping, hooks=hooks)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_graph_builder.py tests/test_builder_hooks.py -q`
Expected: `No module named 'lgkit.builder'`.

- [ ] **Step 3: Create `src/lgkit/hooks.py`**

```python
"""Hooks the graph builder calls instead of importing an app's stores.

lgkit knows nothing about where agents/workflows are stored or how event
scripts run. An app (lgtools) passes its own resolvers; the defaults resolve
nothing and dispatch nothing, which is what a standalone workflow needs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def _resolve_nothing(_ref: Any) -> None:
    return None


@dataclass(frozen=True)
class BuildHooks:
    agent_resolver: Callable[[Any], Any] = _resolve_nothing
    """agent id -> AgentSpec | None (must not raise for unknown ids)."""

    workflow_resolver: Callable[[str], Any] = _resolve_nothing
    """workflow ref -> WorkflowSpec | None."""

    dispatch_event: Callable[..., dict[str, Any]] | None = None
    """dispatch_event(event_type, state, *, scope, origin, node_id, spec, trace)
    -> {"state_delta": dict, "route": str | None}. None = event scripts do not run."""


DEFAULT_HOOKS = BuildHooks()

__all__ = ["BuildHooks", "DEFAULT_HOOKS"]
```

- [ ] **Step 4: Create `src/lgkit/builder.py`**

```bash
cp $LGT/src/lgtools/kernel/graph_builder.py src/lgkit/builder.py
```

Apply these edits in order:

a) Replace the whole import block from `import hashlib` down to `from lgtools.kernel.state import AgentState` with:
```python
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
```

b) Right after the `GraphBuildError` class add:
```python


def _no_dispatch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {}
```

c) Change `_make_node_fn`'s signature and its first lines:
```python
def _make_node_fn(
    node: NodeSpec,
    spec: WorkflowSpec,
    resolved=None,
    hooks: BuildHooks = DEFAULT_HOOKS,
) -> Callable:
```
and replace `    from lgtools.kernel.event_dispatch import dispatch_event` with
```python
    dispatch_event = hooks.dispatch_event or _no_dispatch
```

d) Inside `_node_trace`, replace `from lgtools.kernel.events import emit_event` with `from lgkit.events import emit_event`.

e) `_expand_map_node`: change the signature to `def _expand_map_node(graph, m, successor: str, resolved=None, spec=None, hooks: BuildHooks = DEFAULT_HOOKS) -> None:` and pass `hooks` as the fourth argument of its inner `_make_node_fn(NodeSpec(...), spec, resolved, hooks)` call.

f) Delete the whole `_safe_resolve_agent` function.

g) In `_build_state_schema`, replace `from lgtools.kernel.state import get_reducer` with `from lgkit.state import get_reducer`.

h) Replace the `build_graph` signature and its opening through the `resolved = Resolver.resolve(spec)` branch with:
```python
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
    ``resolved`` is handed to every node fn unchanged (lgtools passes its
    resolved unit, standalone callers leave it None).
    """
    from lgkit.nodes import ensure_registered

    ensure_registered()
```
(Task 5 creates `lgkit.nodes`; until then add a temporary `src/lgkit/nodes/__init__.py` containing `def ensure_registered() -> None: ...` so this task's tests run — Task 5 replaces it.)

i) In the rest of `build_graph` replace:
   - `detect_workflow_cycle(spec, resolve_workflow, agent_resolver=_safe_resolve_agent)` → `detect_workflow_cycle(spec, hooks.workflow_resolver, agent_resolver=hooks.agent_resolver)`
   - the three `validate_*(spec, _safe_resolve_agent)` calls → `validate_*(spec, hooks.agent_resolver)`
   - `_make_node_fn(node, spec, resolved)` → `_make_node_fn(node, spec, resolved, hooks)`
   - `_expand_map_node(graph, node, plain_targets[node.id][0], resolved, spec)` → `_expand_map_node(graph, node, plain_targets[node.id][0], resolved, spec, hooks)`

j) Replace `__all__` with:
```python
__all__ = [
    "build_graph",
    "GraphBuildError",
    "detect_workflow_cycle",
    "validate_goto_agents",
    "validate_map_nodes",
    "validate_parallel_branches",
]
```

Check: `grep -n "lgtools\|Resolver\|resolve_workflow\b\|_safe_resolve_agent" src/lgkit/builder.py` prints nothing.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`
Expected: all pass (50 + 8 graph_builder + 5 hooks = 63).

- [ ] **Step 6: Commit**

```bash
git add src/lgkit/hooks.py src/lgkit/builder.py src/lgkit/nodes/__init__.py tests/test_graph_builder.py tests/test_builder_hooks.py
git commit -m "feat(lgkit): graph builder with injectable BuildHooks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 5: Built-in `hitl` node, template helpers, `ScriptedLLM`

**Files:**
- Create: `$LGA/src/lgkit/nodes/_template.py`, `$LGA/src/lgkit/nodes/hitl.py`, `$LGA/src/lgkit/testing.py`
- Modify: `$LGA/src/lgkit/nodes/__init__.py` (replace the Task 4 placeholder)
- Test: `$LGA/tests/test_hitl.py`, `$LGA/tests/test_testing.py`

**Interfaces:**
- Consumes: `lgkit.builder.build_graph`, `lgkit.events.emit_event`, `lgkit.context.{RunContext, Budget, use_ctx, clear_ctx}`.
- Produces:
  - `lgkit.nodes._template.flat_state(state: dict) -> dict` (missing keys render `""`, `{scratch.a.b}` works)
  - `lgkit.nodes.hitl.run(state, params, prompt, ctx=None, resolved=None) -> dict` (same behavior as lgtools' hitl)
  - `lgkit.nodes.ensure_registered() -> None` (adds `hitl`; Task 6 adds `advisor`, `approval`; never overwrites an existing kind)
  - `lgkit.testing.ScriptedLLM(replies: list[dict | BaseException])` with `.calls: int` and `.with_structured_output(schema).invoke(messages)`; `lgkit.testing.using_llm(llm)` context manager that sets a `RunContext` whose `.llm` is `llm`.

- [ ] **Step 1: Write the failing tests**

`tests/test_testing.py`:

```python
import pytest
from pydantic import BaseModel

from lgkit.context import current_ctx
from lgkit.testing import ScriptedLLM, using_llm


class Out(BaseModel):
    x: int


def test_scripted_llm_returns_replies_in_order_and_counts_calls():
    llm = ScriptedLLM([{"x": 1}, {"x": 2}])
    s = llm.with_structured_output(Out)
    assert s.invoke([]).x == 1
    assert s.invoke([]).x == 2
    assert llm.calls == 2


def test_scripted_llm_raises_scripted_exceptions_and_when_empty():
    llm = ScriptedLLM([RuntimeError("boom")])
    s = llm.with_structured_output(Out)
    with pytest.raises(RuntimeError, match="boom"):
        s.invoke([])
    with pytest.raises(RuntimeError, match="no replies left"):
        s.invoke([])


def test_using_llm_sets_and_clears_context():
    llm = ScriptedLLM([])
    with using_llm(llm):
        assert current_ctx().llm is llm
    assert current_ctx() is None
```

`tests/test_hitl.py`:

```python
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from lgkit.builder import build_graph
from lgkit.nodes._template import flat_state
from lgkit.registry import NodeDef, all_node_defs, register
from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec


def _gate_spec() -> WorkflowSpec:
    return WorkflowSpec(
        nodes=[
            NodeSpec(
                id="gate",
                kind="hitl",
                params={"message": "Ship {scratch.item}?", "choices": ["yes", "no"]},
            )
        ],
        edges=[
            EdgeSpec(source="gate", target="END", condition="yes"),
            EdgeSpec(source="gate", target="END", condition="no"),
        ],
        entry="gate",
    )


def test_flat_state_tolerates_missing_and_nested_keys():
    flat = flat_state({"task": "t", "scratch": {"a": {"b": "deep"}}})
    assert "{task}|{nope}|{scratch.a.b}|{scratch.zz}".format_map(flat) == "t||deep|"


def test_hitl_pauses_then_resumes_with_choice():
    app = build_graph(_gate_spec(), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "h1"}}
    app.invoke({"task": "t", "scratch": {"item": "v2"}}, cfg)
    ask = app.get_state(cfg).interrupts[0].value
    assert ask == {"node": "gate", "message": "Ship v2?", "choices": ["yes", "no"]}

    out = app.invoke(Command(resume={"choice": "yes", "comment": "go"}), cfg)
    assert out["signal"] == "yes"
    assert out["scratch"]["gate"] == {"choice": "yes", "comment": "go"}


def test_hitl_rejects_unknown_choice():
    app = build_graph(_gate_spec(), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "h2"}}
    app.invoke({"task": "t"}, cfg)
    with pytest.raises(ValueError, match="not in"):
        app.invoke(Command(resume={"choice": "maybe"}), cfg)


def test_ensure_registered_never_overrides_an_app_kind():
    def app_hitl(state, params, prompt, ctx=None, resolved=None):
        return {}

    register("hitl", NodeDef(kind="hitl", fn=app_hitl, default_prompt="", description="app"))
    build_graph(_gate_spec(), checkpointer=InMemorySaver())
    assert all_node_defs()["hitl"].fn is app_hitl
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_testing.py tests/test_hitl.py -q`
Expected: `No module named 'lgkit.testing'` / `'lgkit.nodes._template'`.

- [ ] **Step 3: Create `src/lgkit/nodes/_template.py`**

```python
"""Render node message templates against graph state.

``{scratch.key}`` works because format_map looks up ``scratch`` and then does
attribute access; _AttrDict makes that attribute access succeed, and missing
keys render as "" instead of raising.
"""

from __future__ import annotations


class _DefaultDict(dict):
    def __missing__(self, key):
        return ""


class _AttrDict(dict):
    def __missing__(self, key: str):
        return ""

    def __getattr__(self, key: str):
        # Ephemeral wrapper — must not be persisted into graph state/checkpoint.
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        value = self.get(key, "")
        return _AttrDict(value) if isinstance(value, dict) else value


def flat_state(state: dict) -> dict:
    flat = _DefaultDict(state)
    flat["scratch"] = _AttrDict(state.get("scratch") or {})
    return flat


__all__ = ["flat_state"]
```

- [ ] **Step 4: Create `src/lgkit/nodes/hitl.py`** (lgtools' `hitl_node.py` with lgkit imports)

```python
"""HITL node — parks the run with interrupt() until a human answers.

IMPORTANT: code before interrupt() re-executes when the run resumes
(LangGraph replay semantics), so NOTHING is emitted before that line.
After resume the node emits one normal node.start/node.end pair.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.types import interrupt

from lgkit.events import emit_event
from lgkit.nodes._template import flat_state


def run(state, params, prompt, ctx=None, resolved=None) -> dict[str, Any]:
    node_id = params.get("__node_id") or "hitl"
    choices = list(params.get("choices") or [])
    message = str(params.get("message") or "").format_map(flat_state(state))
    started = time.perf_counter()

    answer = interrupt({"node": node_id, "message": message, "choices": choices})

    choice = (answer or {}).get("choice")
    comment = (answer or {}).get("comment", "")
    if choice not in choices:
        raise ValueError(f"hitl '{node_id}': resume choice {choice!r} not in {choices}")

    emit_event(
        "node.start",
        node_id,
        {
            "input": {"message": message},
            "iteration": (state.get("scratch") or {}).get("iteration", 0),
        },
    )
    result = {"choice": choice, "comment": comment}
    key = params.get("result_key") or node_id
    scratch_out = {**state.get("scratch", {}), key: result}
    goto_mode = params.get("route_mode") == "goto"
    route_key = "goto" if goto_mode else "signal"
    emit_event(
        "node.end",
        node_id,
        {
            "result": result,
            "signal": choice if not goto_mode else None,
            "goto": choice if goto_mode else None,
            "duration_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return {
        route_key: choice,
        "scratch": scratch_out,
        "events": [{"node": node_id, "result": result, "tools": []}],
    }


__all__ = ["run"]
```

- [ ] **Step 5: Replace `src/lgkit/nodes/__init__.py`**

```python
"""Node kinds lgkit ships. Registered lazily by build_graph, never overriding
a kind an app registered first (lgtools registers its own ``hitl``)."""

from __future__ import annotations

from lgkit.nodes import hitl
from lgkit.registry import NodeDef, all_node_defs, register


def _builtin_defs() -> list[NodeDef]:
    return [
        NodeDef(
            kind="hitl",
            fn=hitl.run,
            default_prompt="",
            description="Waits for a human decision — choices + optional comment; the choice becomes the routing signal.",
            default_params={"message": "", "choices": ["approve", "reject"], "result_key": ""},
            is_router=True,
        ),
    ]


def ensure_registered() -> None:
    existing = all_node_defs()
    for d in _builtin_defs():
        if d.kind not in existing:
            register(d.kind, d)


__all__ = ["ensure_registered"]
```

- [ ] **Step 6: Create `src/lgkit/testing.py`**

```python
"""Test/demo helpers: a scripted fake LLM and a context manager that hands it
to nodes the same way an app's runner does (RunContext.llm)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lgkit.context import Budget, RunContext, clear_ctx, use_ctx


class ScriptedLLM:
    """Returns scripted replies from ``with_structured_output(schema).invoke()``.

    A dict reply is validated into ``schema`` (so an invalid reply raises the
    same ValidationError a real model's bad output would); an exception reply
    is raised. ``calls`` counts invoke() calls across all structured views.
    """

    def __init__(self, replies: list[Any]):
        self._replies = list(replies)
        self.calls = 0

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> _Structured:
        return _Structured(self, schema)

    def _next(self) -> Any:
        self.calls += 1
        if not self._replies:
            raise RuntimeError("ScriptedLLM: no replies left")
        return self._replies.pop(0)


class _Structured:
    def __init__(self, llm: ScriptedLLM, schema: Any):
        self._llm = llm
        self._schema = schema

    def invoke(self, _messages: Any, **_kwargs: Any) -> Any:
        reply = self._llm._next()
        if isinstance(reply, BaseException):
            raise reply
        return self._schema.model_validate(reply)


@contextmanager
def using_llm(llm: Any) -> Iterator[None]:
    token = use_ctx(
        RunContext(
            session_id="lgkit-local",
            workspace=Path("."),
            llm=llm,
            budget=Budget(max_iterations=25, recursion_limit=100),
        )
    )
    try:
        yield
    finally:
        clear_ctx(token)


__all__ = ["ScriptedLLM", "using_llm"]
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest -q`
Expected: all pass (63 + 3 testing + 4 hitl = 70).

- [ ] **Step 8: Commit**

```bash
git add src/lgkit/nodes src/lgkit/testing.py tests/test_hitl.py tests/test_testing.py
git commit -m "feat(lgkit): built-in hitl node, template helper, ScriptedLLM

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 6: `advisor` and `approval` nodes + builder guards

**Files:**
- Create: `$LGA/src/lgkit/nodes/advisor.py`, `$LGA/src/lgkit/nodes/approval.py`
- Modify: `$LGA/src/lgkit/nodes/__init__.py` (register both), `$LGA/src/lgkit/builder.py` (checkpointer guard, cache exclusion)
- Test: `$LGA/tests/test_approval_nodes.py`

**Interfaces:**
- Consumes: `flat_state`, `emit_event`, `lgkit.llm.build_llm`, `ScriptedLLM`/`using_llm` (tests).
- Produces:
  - `lgkit.nodes.advisor.ESCALATE = "escalate"`, `advisor.SYSTEM_PROMPT: str`, `advisor.advice_key(gate_id: str) -> str` (= `f"{gate_id}__advice"`), `advisor.advice_model(choices) -> type[BaseModel]`, `advisor.run(...)`.
    Advisor params: `gate_id: str`, `message: str`, `choices: list[str]`, `criteria: str`, optional `llm: {"provider", "model", ...}`, optional `max_attempts: int = 2`.
    Writes `scratch[advice_key(gate_id)]` = `{"choice", "confidence", "reason"}` or `{"error": "<Type>: <msg>"}`. Never raises (except LangGraph control-flow).
  - `lgkit.nodes.approval.run(...)`. Params: `message`, `choices`, `approver: "human"|"agent"`, `min_confidence: float`, optional `advice_key`, optional `result_key`.
    Returns `{"signal": choice, "scratch": {result_key or node_id: {"choice", "comment", "by", "advice"}}, "events": [...]}`.
    Interrupt payload: `{"node", "message", "choices", "advice", "advice_error"}`. Resume accepts `{"choice", "comment"?}` or a bare `str`.
  - `build_graph` raises `GraphBuildError` if the spec has an `approval` node and `checkpointer is None`; `approval` nodes are never cached.

- [ ] **Step 1: Write the failing tests** — `tests/test_approval_nodes.py`

```python
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from lgkit.builder import GraphBuildError, build_graph
from lgkit.nodes.advisor import advice_key
from lgkit.spec import CacheConfig, EdgeSpec, NodePolicy, NodeSpec, WorkflowSpec
from lgkit.testing import ScriptedLLM, using_llm

CHOICES = ["approve", "revise"]


def _spec(approver: str, with_advisor: bool, min_confidence: float = 0.7) -> WorkflowSpec:
    nodes = []
    edges = []
    gate_params = {
        "message": "Publish {scratch.draft}?",
        "choices": CHOICES,
        "approver": approver,
        "min_confidence": min_confidence,
    }
    if with_advisor:
        nodes.append(
            NodeSpec(
                id="g__advisor",
                kind="advisor",
                params={
                    "gate_id": "g",
                    "message": "Publish {scratch.draft}?",
                    "choices": CHOICES,
                    "criteria": "complete and polite",
                },
            )
        )
        edges.append(EdgeSpec(source="g__advisor", target="g"))
        gate_params["advice_key"] = advice_key("g")
    nodes.append(NodeSpec(id="g", kind="approval", params=gate_params))
    edges += [EdgeSpec(source="g", target="END", condition=c) for c in CHOICES]
    return WorkflowSpec(nodes=nodes, edges=edges, entry=nodes[0].id)


def _run(spec, llm, thread="t"):
    app = build_graph(spec, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": thread}}
    with using_llm(llm):
        out = app.invoke({"task": "t", "scratch": {"draft": "notes"}}, cfg)
    return app, cfg, out


def _ask(app, cfg):
    snap = app.get_state(cfg)
    return snap.interrupts[0].value if snap.interrupts else None


GOOD = {"choice": "approve", "confidence": 0.9, "reason": "fine"}


def test_human_mode_interrupts_without_advice():
    app, cfg, _ = _run(_spec("human", False), ScriptedLLM([]))
    assert _ask(app, cfg) == {
        "node": "g",
        "message": "Publish notes?",
        "choices": CHOICES,
        "advice": None,
        "advice_error": None,
    }
    out = app.invoke(Command(resume={"choice": "revise", "comment": "typo"}), cfg)
    assert out["signal"] == "revise"
    assert out["scratch"]["g"] == {"choice": "revise", "comment": "typo", "by": "human", "advice": None}


def test_human_mode_accepts_bare_string_resume():
    app, cfg, _ = _run(_spec("human", False), ScriptedLLM([]))
    out = app.invoke(Command(resume="approve"), cfg)
    assert out["scratch"]["g"]["choice"] == "approve"
    assert out["scratch"]["g"]["comment"] == ""


def test_human_mode_rejects_unknown_choice():
    app, cfg, _ = _run(_spec("human", False), ScriptedLLM([]))
    with pytest.raises(ValueError, match="not in"):
        app.invoke(Command(resume={"choice": "maybe"}), cfg)


def test_human_with_advice_shows_advice_and_calls_llm_once_across_resume():
    llm = ScriptedLLM([GOOD])
    app, cfg, _ = _run(_spec("human", True), llm)
    assert _ask(app, cfg)["advice"] == GOOD
    with using_llm(llm):
        out = app.invoke(Command(resume={"choice": "approve"}), cfg)
    assert llm.calls == 1  # regression: advisor must not re-run on resume
    assert out["scratch"]["g"]["by"] == "human"
    assert out["scratch"]["g"]["advice"] == GOOD


def test_agent_confident_decides_without_interrupt():
    app, cfg, out = _run(_spec("agent", True), ScriptedLLM([GOOD]))
    assert _ask(app, cfg) is None
    assert out["signal"] == "approve"
    assert out["scratch"]["g"] == {"choice": "approve", "comment": "fine", "by": "agent", "advice": GOOD}


def test_agent_at_exact_threshold_decides():
    reply = {"choice": "approve", "confidence": 0.7, "reason": "ok"}
    app, cfg, out = _run(_spec("agent", True, min_confidence=0.7), ScriptedLLM([reply]))
    assert _ask(app, cfg) is None
    assert out["scratch"]["g"]["by"] == "agent"


@pytest.mark.parametrize(
    "reply",
    [
        {"choice": "approve", "confidence": 0.3, "reason": "unsure"},
        {"choice": "escalate", "confidence": 0.9, "reason": "policy question"},
    ],
)
def test_agent_low_confidence_or_escalate_asks_human(reply):
    app, cfg, _ = _run(_spec("agent", True), ScriptedLLM([reply]))
    ask = _ask(app, cfg)
    assert ask["advice"] == reply
    out = app.invoke(Command(resume={"choice": "revise"}), cfg)
    assert out["scratch"]["g"]["by"] == "human"


def test_advisor_failure_is_retried_then_escalated():
    llm = ScriptedLLM([RuntimeError("timeout"), RuntimeError("timeout again")])
    app, cfg, _ = _run(_spec("agent", True), llm)
    ask = _ask(app, cfg)
    assert llm.calls == 2
    assert ask["advice"] is None
    assert ask["advice_error"] == "RuntimeError: timeout again"


def test_advisor_invalid_choice_counts_as_failure():
    bad = {"choice": "maybe", "confidence": 0.99, "reason": "?"}
    app, cfg, _ = _run(_spec("agent", True), ScriptedLLM([bad, bad]))
    ask = _ask(app, cfg)
    assert ask["advice"] is None
    assert ask["advice_error"].startswith("ValidationError")


def test_advisor_retry_recovers_on_second_attempt():
    app, cfg, out = _run(_spec("agent", True), ScriptedLLM([RuntimeError("blip"), GOOD]))
    assert _ask(app, cfg) is None
    assert out["scratch"]["g"]["by"] == "agent"


def test_approval_requires_checkpointer():
    with pytest.raises(GraphBuildError, match="checkpointer"):
        build_graph(_spec("human", False))


def test_approval_is_never_cached():
    spec = _spec("human", False)
    spec.nodes[-1].policy = NodePolicy(cache=CacheConfig())
    app = build_graph(spec, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "c"}}
    app.invoke({"task": "t"}, cfg)
    out = app.invoke(Command(resume={"choice": "approve"}), cfg)  # a cached gate crashes resume
    assert out["signal"] == "approve"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_approval_nodes.py -q`
Expected: `No module named 'lgkit.nodes.advisor'`.

- [ ] **Step 3: Create `src/lgkit/nodes/advisor.py`**

```python
"""Advisor node — an LLM recommends one of an approval gate's choices.

It runs as its own node BEFORE the gate so the LLM call is checkpointed:
LangGraph re-executes an interrupted node from its first line on resume, so an
LLM call inside the gate would be paid twice and could change the advice the
human already saw. It never raises; a failure is recorded as
{"error": "..."} and the gate treats that as "ask a human".
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel, Field, create_model

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
    """Precedence: per-gate llm spec -> run context llm -> lgkit default."""
    from lgkit.llm import build_llm

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
            data = out.model_dump() if isinstance(out, BaseModel) else out
            result = schema.model_validate(data).model_dump()
            break
        except GraphBubbleUp:
            raise
        except Exception as e:  # recorded, never raised: the gate escalates to a human
            result = {"error": f"{type(e).__name__}: {e}"}

    return {"scratch": {advice_key(gate_id): result}}


__all__ = ["ESCALATE", "SYSTEM_PROMPT", "advice_key", "advice_model", "run"]
```

- [ ] **Step 4: Create `src/lgkit/nodes/approval.py`**

```python
"""Approval gate — a human or an agent picks one of ``choices``.

approver="agent": the advisor's answer is used when it is valid, not
"escalate" and confidence >= min_confidence; otherwise a human is asked.
approver="human": a human is always asked; advice (if an advisor ran) is
shown alongside.

Nothing with side effects runs before interrupt(): on resume LangGraph
re-executes this node from the top, so only pure reads happen before it.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.types import interrupt

from lgkit.events import emit_event
from lgkit.nodes._template import flat_state


def _usable(advice: dict[str, Any] | None) -> dict[str, Any] | None:
    return advice if advice and "error" not in advice else None


def _agent_can_decide(advice: dict[str, Any] | None, choices: list[str], min_confidence: float) -> bool:
    return (
        advice is not None
        and advice.get("choice") in choices
        and float(advice.get("confidence", 0.0)) >= min_confidence
    )


def _parse_answer(answer: Any) -> tuple[Any, str]:
    if isinstance(answer, str):
        return answer, ""
    answer = answer or {}
    return answer.get("choice"), str(answer.get("comment") or "")


def run(state, params, prompt, ctx=None, resolved=None) -> dict[str, Any]:
    node_id = params.get("__node_id") or "approval"
    choices = list(params.get("choices") or [])
    approver = params.get("approver", "human")
    min_confidence = float(params.get("min_confidence", 0.7))
    advice_key = params.get("advice_key")
    raw_advice = (state.get("scratch") or {}).get(advice_key) if advice_key else None
    advice = _usable(raw_advice)
    message = str(params.get("message") or "").format_map(flat_state(state))
    started = time.perf_counter()

    if approver == "agent" and _agent_can_decide(advice, choices, min_confidence):
        choice, comment, by = advice["choice"], str(advice.get("reason") or ""), "agent"
    else:
        answer = interrupt(
            {
                "node": node_id,
                "message": message,
                "choices": choices,
                "advice": advice,
                "advice_error": (raw_advice or {}).get("error"),
            }
        )
        choice, comment = _parse_answer(answer)
        if choice not in choices:
            raise ValueError(f"approval '{node_id}': resume choice {choice!r} not in {choices}")
        by = "human"

    result = {"choice": choice, "comment": comment, "by": by, "advice": advice}
    emit_event("node.start", node_id, {"input": {"message": message}})
    emit_event(
        "node.end",
        node_id,
        {
            "result": result,
            "signal": choice,
            "duration_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return {
        "signal": choice,
        "scratch": {params.get("result_key") or node_id: result},
        "events": [{"node": node_id, "result": result, "tools": []}],
    }


__all__ = ["run"]
```

- [ ] **Step 5: Register both kinds** — in `src/lgkit/nodes/__init__.py` change the import to `from lgkit.nodes import advisor, approval, hitl` and append to the list returned by `_builtin_defs()`:

```python
        NodeDef(
            kind="advisor",
            fn=advisor.run,
            default_prompt=advisor.SYSTEM_PROMPT,
            description="LLM recommends one of an approval gate's choices (or escalates).",
            default_params={"max_attempts": 2},
            uses_node_prompt=True,
        ),
        NodeDef(
            kind="approval",
            fn=approval.run,
            default_prompt="",
            description="Approval gate — a human or an agent picks a choice; the choice becomes the routing signal.",
            default_params={"approver": "human", "min_confidence": 0.7},
            is_router=True,
        ),
```

- [ ] **Step 6: Builder guards** — in `src/lgkit/builder.py`:

a) Below `_VOLATILE_KEYS` add:
```python
# Kinds that pause with interrupt(): they need a checkpointer to resume and must
# never be cached (a cached pausing node crashes resume inside LangGraph).
_PAUSING_KINDS: frozenset[str] = frozenset({"hitl", "harness", "approval"})
_NEEDS_CHECKPOINTER: frozenset[str] = frozenset({"approval"})
```
b) In `build_graph`, right after `ensure_registered()`:
```python
    if checkpointer is None:
        needing = [n.id for n in spec.nodes if n.kind in _NEEDS_CHECKPOINTER]
        if needing:
            raise GraphBuildError(
                f"Nodes {needing} can pause for a human and need a checkpointer to resume: "
                "build_graph(spec, checkpointer=InMemorySaver()) "
                "(from langgraph.checkpoint.memory import InMemorySaver)"
            )
```
(`hitl` is deliberately not in `_NEEDS_CHECKPOINTER`: lgtools builds hitl graphs without one in some paths and that behavior must not change.)
c) Replace `if pol.cache is not None and node.kind not in ("hitl", "harness"):` with `if pol.cache is not None and node.kind not in _PAUSING_KINDS:`.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest -q`
Expected: all pass (70 + 13 = 83).

- [ ] **Step 8: Commit**

```bash
git add src/lgkit/nodes src/lgkit/builder.py tests/test_approval_nodes.py
git commit -m "feat(lgkit): advisor + approval nodes with human escalation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 7: `approval_gate()` pattern

**Files:**
- Create: `$LGA/src/lgkit/patterns/__init__.py`, `$LGA/src/lgkit/patterns/approval_gate.py`
- Test: `$LGA/tests/test_approval_gate.py`

**Interfaces:**
- Consumes: `NodeSpec`, `EdgeSpec`, `WorkflowSpec`, `advisor.advice_key`, `advisor.ESCALATE`, `build_graph`, `ScriptedLLM`, `using_llm`.
- Produces:
  - `lgkit.patterns.Fragment` (frozen dataclass): `nodes: tuple[NodeSpec, ...]`, `edges: tuple[EdgeSpec, ...]`, `entry: str`, `exit: str`, `choices: tuple[str, ...]`; `.to_workflow(name="approval_gate") -> WorkflowSpec` (adds `exit --choice--> END` for every choice).
  - `lgkit.patterns.approval_gate(id, *, message, choices, approver="human", advise=False, criteria=None, min_confidence=0.7, llm=None, result_key=None) -> Fragment`.

- [ ] **Step 1: Write the failing tests** — `tests/test_approval_gate.py`

```python
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from lgkit.builder import build_graph
from lgkit.patterns import Fragment, approval_gate
from lgkit.registry import NodeDef, register
from lgkit.spec import EdgeSpec, NodeSpec, WorkflowSpec
from lgkit.testing import ScriptedLLM, using_llm


def test_human_gate_is_a_single_node():
    frag = approval_gate("g", message="Ok?", choices=["yes", "no"])
    assert isinstance(frag, Fragment)
    assert [n.kind for n in frag.nodes] == ["approval"]
    assert frag.entry == frag.exit == "g"
    assert frag.edges == ()


def test_agent_gate_puts_advisor_first():
    frag = approval_gate(
        "g", message="Ok?", choices=["yes", "no"], approver="agent", criteria="be sure",
        llm={"provider": "openai", "model": "gpt-4o-mini"},
    )
    adv, gate = frag.nodes
    assert (adv.id, adv.kind, gate.kind) == ("g__advisor", "advisor", "approval")
    assert adv.params["llm"] == {"provider": "openai", "model": "gpt-4o-mini"}
    assert gate.params["advice_key"] == "g__advice"
    assert frag.edges == (EdgeSpec(source="g__advisor", target="g"),)
    assert (frag.entry, frag.exit) == ("g__advisor", "g")


def test_to_workflow_routes_every_choice_to_end():
    wf = approval_gate("g", message="Ok?", choices=["yes", "no"]).to_workflow(name="w")
    assert wf.name == "w" and wf.entry == "g"
    assert {(e.source, e.target, e.condition) for e in wf.edges} == {
        ("g", "END", "yes"),
        ("g", "END", "no"),
    }


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"choices": []}, "choices"),
        ({"choices": ["a", "a"]}, "unique"),
        ({"choices": ["a", "escalate"]}, "escalate"),
        ({"approver": "robot"}, "approver"),
        ({"approver": "agent"}, "criteria"),
        ({"advise": True}, "criteria"),
        ({"approver": "agent", "advise": True, "criteria": "x"}, "advise"),
        ({"min_confidence": 1.5}, "min_confidence"),
        ({"message": "Approve {draft"}, "message"),
    ],
)
def test_invalid_arguments_fail_at_build_time(kwargs, match):
    args = {"message": "Ok?", "choices": ["yes", "no"], **kwargs}
    with pytest.raises(ValueError, match=match):
        approval_gate("g", **args)


def test_end_to_end_human_gate():
    frag = approval_gate("g", message="Ship {scratch.v}?", choices=["yes", "no"])
    app = build_graph(frag.to_workflow(), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "e1"}}
    app.invoke({"task": "t", "scratch": {"v": "1.2"}}, cfg)
    assert app.get_state(cfg).interrupts[0].value["message"] == "Ship 1.2?"
    out = app.invoke(Command(resume={"choice": "yes"}), cfg)
    assert out["scratch"]["g"]["by"] == "human"


def _drafter(state, params, prompt, ctx=None, resolved=None):
    n = int((state.get("scratch") or {}).get("round", 0)) + 1
    return {"scratch": {"round": n, "draft": f"draft v{n}"}}


def test_revise_loop_uses_fresh_advice_each_round():
    register("custom", NodeDef(kind="custom", fn=_drafter, default_prompt="", description="drafter"))
    frag = approval_gate(
        "g", message="{scratch.draft}", choices=["approve", "revise"],
        approver="agent", criteria="approve v2 only",
    )
    wf = WorkflowSpec(
        nodes=[NodeSpec(id="draft", kind="custom"), *frag.nodes],
        edges=[
            EdgeSpec(source="draft", target=frag.entry),
            *frag.edges,
            EdgeSpec(source="g", target="draft", condition="revise"),
            EdgeSpec(source="g", target="END", condition="approve"),
        ],
        entry="draft",
    )
    llm = ScriptedLLM(
        [
            {"choice": "revise", "confidence": 0.9, "reason": "v1 incomplete"},
            {"choice": "approve", "confidence": 0.9, "reason": "v2 good"},
        ]
    )
    app = build_graph(wf, checkpointer=InMemorySaver())
    with using_llm(llm):
        out = app.invoke({"task": "t", "scratch": {"max_iterations": 10}}, {"configurable": {"thread_id": "loop"}})
    assert llm.calls == 2
    assert out["scratch"]["round"] == 2
    assert out["scratch"]["g"]["comment"] == "v2 good"
    assert out["signal"] == "approve"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_approval_gate.py -q`
Expected: `No module named 'lgkit.patterns'`.

- [ ] **Step 3: Create `src/lgkit/patterns/approval_gate.py`**

```python
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
        list(string.Formatter().parse(message))
    except ValueError as e:
        raise ValueError(f"approval_gate: message is not a valid template ({e}): {message!r}") from e


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
```

- [ ] **Step 4: Create `src/lgkit/patterns/__init__.py`**

```python
"""Ready-made workflow patterns. Each returns spec data, not a runtime."""

from lgkit.patterns.approval_gate import Fragment, approval_gate

__all__ = ["Fragment", "approval_gate"]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`
Expected: all pass (83 + 14 = 97). If `test_revise_loop_uses_fresh_advice_each_round` stops early with `signal == "max_iterations"`, the builder's iteration guard is firing: confirm `max_iterations` in the initial scratch is 10 as written.

- [ ] **Step 6: Commit**

```bash
git add src/lgkit/patterns tests/test_approval_gate.py
git commit -m "feat(lgkit): approval_gate pattern returning a spec Fragment

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 8: Examples and README

**Files:**
- Move: `$LGA/main.py` → `$LGA/examples/plan_execute_review.py` (rewritten)
- Create: `$LGA/examples/approval_demo.py`
- Modify: `$LGA/README.md`
- Test: `$LGA/tests/test_examples.py`

**Interfaces:**
- Consumes: `approval_gate`, `build_graph`, `ScriptedLLM`, `using_llm`, `lgkit.llm.build_llm`.
- Produces: `examples/plan_execute_review.py` with `MAX_ROUNDS = 3`, `route(state) -> str`, `run(task: str) -> dict`; `examples/approval_demo.py` with `main() -> dict`.

- [ ] **Step 1: Write the failing tests** — `tests/test_examples.py`

```python
import importlib.util
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plan_execute_review_loops_until_approved():
    mod = _load("plan_execute_review")
    out = mod.run("Create API documentation")
    assert out["approved"] is True
    assert out["attempts"] == 2


def test_plan_execute_review_stops_at_max_rounds():
    mod = _load("plan_execute_review")
    assert mod.route({"approved": False, "attempts": mod.MAX_ROUNDS}) == "end"
    assert mod.route({"approved": False, "attempts": 1}) == "executor"


def test_approval_demo_escalates_and_records_human(monkeypatch):
    mod = _load("approval_demo")
    monkeypatch.delenv("LGKIT_REAL_LLM", raising=False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "approve")
    result = mod.main()
    assert result["by"] == "human"
    assert result["choice"] == "approve"
    assert result["advice"]["confidence"] == 0.55
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_examples.py -q`
Expected: FAIL — `examples/plan_execute_review.py` does not exist.

- [ ] **Step 3: Move and rewrite the old demo**

```bash
mkdir -p examples && git mv main.py examples/plan_execute_review.py
```

Replace the contents of `examples/plan_execute_review.py`:

```python
"""Plain-LangGraph plan -> execute -> review loop (no lgkit), kept as a reference.

The first draft is always rejected so the loop is exercised; MAX_ROUNDS stops
it even if the reviewer never approves.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

MAX_ROUNDS = 3


class AgentState(TypedDict, total=False):
    task: str
    plan: str
    result: str
    approved: bool
    attempts: int


def planner(state: AgentState) -> AgentState:
    return {"plan": f"Plan for: {state['task']}", "attempts": 0}


def executor(state: AgentState) -> AgentState:
    attempts = state.get("attempts", 0) + 1
    suffix = " (draft)" if attempts == 1 else ""
    return {"result": f"Executed: {state['plan']}{suffix}", "attempts": attempts}


def reviewer(state: AgentState) -> AgentState:
    return {"approved": not state["result"].endswith("(draft)")}


def route(state: AgentState) -> str:
    if state["approved"] or state["attempts"] >= MAX_ROUNDS:
        return "end"
    return "executor"


def build():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("reviewer", reviewer)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges("reviewer", route, {"executor": "executor", "end": END})
    return graph.compile()


def run(task: str) -> dict:
    return build().invoke({"task": task})


if __name__ == "__main__":
    print(run("Create API documentation"))
```

- [ ] **Step 4: Create `examples/approval_demo.py`**

```python
"""approval_gate demo: an agent reviews a draft and escalates to you when unsure.

    uv run python examples/approval_demo.py                  # scripted LLM, no API key
    LGKIT_REAL_LLM=1 OPENAI_API_KEY=sk-... uv run python examples/approval_demo.py
"""

import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from lgkit.builder import build_graph
from lgkit.patterns import approval_gate
from lgkit.testing import ScriptedLLM, using_llm


def make_llm():
    if os.environ.get("LGKIT_REAL_LLM"):
        from lgkit.llm import build_llm

        return build_llm()
    return ScriptedLLM([{"choice": "approve", "confidence": 0.55, "reason": "Fine but terse."}])


def main() -> dict:
    gate = approval_gate(
        "publish_check",
        message="Publish this announcement?\n\n{scratch.draft}",
        choices=["approve", "revise"],
        approver="agent",
        criteria="Approve only if the announcement is complete, accurate and polite.",
        min_confidence=0.8,
    )
    app = build_graph(gate.to_workflow(), checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "demo"}}

    with using_llm(make_llm()):
        app.invoke({"task": "publish", "scratch": {"draft": "Release moves to Friday."}}, cfg)
        snap = app.get_state(cfg)
        if snap.interrupts:
            ask = snap.interrupts[0].value
            print(f"Agent escalated. Advice: {ask['advice']}  Error: {ask['advice_error']}")
            answer = input(f"{ask['message']}\nChoose {ask['choices']}: ").strip() or "approve"
            app.invoke(Command(resume={"choice": answer, "comment": "via demo"}), cfg)

    result = app.get_state(cfg).values["scratch"]["publish_check"]
    print(result)
    return result


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Replace `README.md`**

````markdown
# langgraph-agent (`lgkit`)

Build LangGraph workflows from plain spec data, plus ready-made patterns.
`lgtools` (laggraph-hermes) uses this as its kernel.

## Install

```bash
uv sync                      # core + dev tools
uv add 'langgraph-agent[openai]' # in another project, with the OpenAI provider
```

## Approval gate

```python
from langgraph.checkpoint.memory import InMemorySaver
from lgkit.builder import build_graph
from lgkit.patterns import approval_gate

gate = approval_gate(
    "publish_check",
    message="Publish?\n\n{scratch.draft}",
    choices=["approve", "revise"],
    approver="agent",              # "human" (default) | "agent"
    criteria="Approve only if complete and polite.",
    min_confidence=0.8,            # below this, or on any error, a human is asked
)
app = build_graph(gate.to_workflow(), checkpointer=InMemorySaver())
```

| Setting | Who decides |
|---|---|
| `approver="human"` | you, via `interrupt()` |
| `approver="human", advise=True` | you, with the agent's recommendation shown |
| `approver="agent"` | the agent; escalates to you on `escalate`, low confidence or failure |

Every mode writes `scratch[<id>] = {"choice", "comment", "by", "advice"}` and routes on `signal = choice`.

Run the offline demo: `uv run python examples/approval_demo.py`

## Tests

```bash
uv run pytest -q
```
````

- [ ] **Step 6: Run everything**

Run: `uv run pytest -q && uv run python examples/plan_execute_review.py`
Expected: all tests pass (97 + 3 = 100); the script prints a dict with `'approved': True, 'attempts': 2`.

- [ ] **Step 7: Commit**

```bash
git add examples README.md tests/test_examples.py
git commit -m "docs(lgkit): runnable examples and README; fix never-looping review demo

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

## Part B — lgtools switches to `lgkit`

### Task 9: Baseline and dependency

**Files:**
- Modify: `$LGT/pyproject.toml`, `$LGT/uv.lock` (via `uv add`)
- Create (untracked, deleted in Task 11): `$LGT/.pytest-baseline.txt`

**Interfaces:**
- Produces: `.pytest-baseline.txt` — one line per test: `<OUTCOME> <nodeid>`, sorted; `import lgkit` works inside lgtools' venv.

- [ ] **Step 1: Branch and record the baseline BEFORE any change**

```bash
cd $LGT/.. && git status --short   # must be empty; stop and ask if not
git switch -c feat/use-lgkit
cd $LGT
uv run pytest -q -rA -p no:cacheprovider 2>&1 | tee /tmp/lgtools-baseline-full.txt \
  | grep -E '^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) ' | sed -E 's/ - .*//' | sort > .pytest-baseline.txt
tail -3 /tmp/lgtools-baseline-full.txt
wc -l .pytest-baseline.txt
```
Expected: a pytest summary line; the baseline file has one line per test. Note the counts in the task report.

- [ ] **Step 2: Add lgkit as an editable dependency**

```bash
uv add --editable ../../langgraph-agent
uv run python -c "import lgkit, lgkit.builder, lgkit.patterns; print(lgkit.__version__)"
```
Expected: `0.1.0`

- [ ] **Step 3: Re-run the suite — nothing imports lgkit yet, results must be identical**

```bash
uv run pytest -q -rA -p no:cacheprovider 2>&1 | grep -E '^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) ' \
  | sed -E 's/ - .*//' | sort > /tmp/lgtools-after.txt
diff .pytest-baseline.txt /tmp/lgtools-after.txt && echo SAME
```
Expected: `SAME`. If the resolver changed shared versions (langgraph/langchain-core) and results differ, stop and report.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: depend on lgkit (editable, ../../langgraph-agent)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 10: Shims for pure modules, schemas, llm, config, hitl

**Files:**
- Modify: `$LGT/src/lgtools/kernel/{context,events,state,registry,node_validation}.py` (become aliases)
- Modify: `$LGT/src/lgtools/kernel/schemas.py` (lines 1–780 replaced by a re-export header)
- Modify: `$LGT/src/lgtools/llm.py` (wrapper over `lgkit.llm`)
- Modify: `$LGT/src/lgtools/config.py:18-29` (`SUPPORTED_PROVIDERS` from lgkit)
- Modify: `$LGT/src/lgtools/templates/primitives/hitl_node.py` (thin wrapper)
- Test: `$LGT/tests/test_lgkit_shims.py` (new)

**Interfaces:**
- Consumes: everything `lgkit` produced in Tasks 2–6.
- Produces: `lgtools.kernel.<m> is lgkit.<m>` for the five aliased modules; `lgtools.kernel.schemas` exports every `lgkit.spec` name plus the request/response models; `lgtools.llm.build_llm(settings=None, *, provider=None, model=None, **kw)` unchanged signature.

- [ ] **Step 1: Write the failing test** — `$LGT/tests/test_lgkit_shims.py`

```python
"""lgtools' kernel modules are lgkit's: same objects, so private state and
monkeypatch targets keep working after the move."""

import importlib

import pytest


@pytest.mark.parametrize("name", ["context", "events", "state", "registry", "node_validation"])
def test_kernel_module_is_the_lgkit_module(name):
    assert importlib.import_module(f"lgtools.kernel.{name}") is importlib.import_module(f"lgkit.{name}")


def test_schemas_reexports_lgkit_spec_models():
    import lgkit.spec
    from lgtools.kernel import schemas

    assert schemas.WorkflowSpec is lgkit.spec.WorkflowSpec
    assert schemas.NodePolicy is lgkit.spec.NodePolicy
    assert hasattr(schemas, "RunRequest")


def test_llm_wrapper_uses_lgkit_catalog():
    import lgkit.llm
    from lgtools import config, llm

    assert llm.DEFAULT_MODELS is lgkit.llm.DEFAULT_MODELS
    assert llm.LLMConfigError is lgkit.llm.LLMConfigError
    assert config.SUPPORTED_PROVIDERS is lgkit.llm.SUPPORTED_PROVIDERS


def test_hitl_wrapper_keeps_lgtools_module_for_the_palette_contract():
    from lgtools.templates.primitives import hitl_node

    assert hitl_node.run.__module__ == "lgtools.templates.primitives.hitl_node"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd $LGT && uv run pytest tests/test_lgkit_shims.py -q`
Expected: FAILs (modules are not the same objects yet).

- [ ] **Step 3: Alias the five pure modules** — write each file in full, e.g. `src/lgtools/kernel/registry.py`:

```python
"""Moved to lgkit.registry. Aliased (not re-exported) so `lgtools.kernel.registry`
IS `lgkit.registry`: private state like _REGISTRY and monkeypatch targets stay shared."""

import sys

import lgkit.registry as _impl

sys.modules[__name__] = _impl
```

Do the same for `context.py`, `events.py`, `state.py`, `node_validation.py`, changing both `registry` occurrences to the module's name.

- [ ] **Step 4: Rewrite the top of `schemas.py`**

Delete lines 1–780 of `src/lgtools/kernel/schemas.py` (everything above `# ----- API request/response models -----`) and put this in their place:

```python
"""API request/response models.

Workflow/agent spec models live in :mod:`lgkit.spec` and are re-exported here
so existing ``from lgtools.kernel.schemas import WorkflowSpec`` imports keep working.
"""

from __future__ import annotations

from typing import Any, Literal

from lgkit.spec import *  # noqa: F403
from lgkit.spec import __all__ as _SPEC_ALL
from pydantic import BaseModel, Field, field_validator, model_validator

from lgtools.api_models import StrictRequest
```

Then change the `__all__ = [` at the bottom of the file to start with `__all__ = [*_SPEC_ALL,` and delete from it the names already in `_SPEC_ALL` (`NodeKind` … `WorkflowSpec`), keeping only the request/response names (`WorkflowCreateRequest` … `HealthResponse`).

Check: `uv run python -c "from lgtools.kernel.schemas import *; import lgtools.kernel.schemas as s; print(len(s.__all__))"` runs without error.

- [ ] **Step 5: `config.py`** — replace the `SUPPORTED_PROVIDERS = ( ... )` tuple (lines 18–29) with:

```python
from lgkit.llm import SUPPORTED_PROVIDERS  # noqa: E402  (single source of truth)
```

(Keep `PROVIDER_ENV_KEYS` in config — Settings uses it.)

- [ ] **Step 6: Replace `src/lgtools/llm.py`**

```python
"""LLM factory. Provider builders live in lgkit.llm; this wrapper supplies
lgtools' Settings (keys, default provider/model, LiteLLM endpoint, LangSmith)."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from lgkit.llm import DEFAULT_MODELS, PROVIDER_ENV_KEYS, LLMConfigError
from lgkit.llm import build_llm as _build_llm

from lgtools.config import Settings, get_settings


def build_llm(
    settings: Settings | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    settings = settings or get_settings()
    settings.activate_langsmith()
    provider = (provider or settings.provider).lower()
    model = model or settings.model or None
    extra: dict[str, Any] = {}
    if provider == "litellm":
        extra["base_url"] = settings.litellm_api_base or None
    # api_key is passed explicitly (even None) so lgkit never falls back to the
    # raw env var — Settings has already rejected placeholder keys. Unknown
    # providers skip api_key_for (it would hit the keychain) and lgkit raises
    # "Unknown provider" exactly as before.
    api_key = settings.api_key_for(provider) if provider in PROVIDER_ENV_KEYS else None
    return _build_llm(provider, model, api_key=api_key, **extra, **kwargs)


__all__ = ["build_llm", "LLMConfigError", "DEFAULT_MODELS"]
```

- [ ] **Step 7: Thin `hitl_node.py` wrapper** — replace the file:

```python
"""HITL node. Implementation lives in lgkit.nodes.hitl; this wrapper keeps the
fn's __module__ under lgtools.templates so the frontend palette contract test
still sees `hitl` as a kind lgtools ships."""

from __future__ import annotations

from typing import Any

from lgkit.nodes import hitl as _impl


def run(state, params, prompt, ctx=None, resolved=None) -> dict[str, Any]:
    return _impl.run(state, params, prompt, ctx, resolved)


__all__ = ["run"]
```

- [ ] **Step 8: Run the new test, then the whole suite against the baseline**

```bash
uv run pytest tests/test_lgkit_shims.py -q
uv run pytest -q -rA -p no:cacheprovider 2>&1 | grep -E '^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) ' \
  | sed -E 's/ - .*//' | grep -v 'test_lgkit_shims' | sort > /tmp/lgtools-after.txt
diff .pytest-baseline.txt /tmp/lgtools-after.txt && echo SAME
```
Expected: shim tests `8 passed`; then `SAME`. Any diff line is a regression to fix before committing — do not edit existing lgtools tests to make them pass.

- [ ] **Step 9: Commit**

```bash
git add src/lgtools tests/test_lgkit_shims.py
git commit -m "refactor: lgtools kernel modules, schemas, llm and hitl come from lgkit

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

### Task 11: `graph_builder` shim with lgtools hooks

**Files:**
- Modify: `$LGT/src/lgtools/kernel/graph_builder.py` (replace whole file)
- Modify: `$LGT/tests/test_lgkit_shims.py` (add hook tests)
- Delete: `$LGT/.pytest-baseline.txt` (after the final comparison)

**Interfaces:**
- Consumes: `lgkit.builder.{build_graph, GraphBuildError, detect_workflow_cycle, validate_goto_agents, validate_map_nodes, validate_parallel_branches, _iteration_guard}`, `lgkit.hooks.BuildHooks`.
- Produces: `lgtools.kernel.graph_builder.build_graph(spec: WorkflowSpec | ResolvedSpec, checkpointer=None, store=None, dry_run=False)` — same signature as before; `LGTOOLS_HOOKS`.

- [ ] **Step 1: Add the failing tests** — append to `tests/test_lgkit_shims.py`

```python
def test_builder_shim_reexports():
    import lgkit.builder
    from lgtools.kernel import graph_builder

    assert graph_builder.GraphBuildError is lgkit.builder.GraphBuildError
    assert graph_builder._iteration_guard is lgkit.builder._iteration_guard


def test_dispatch_hook_resolves_at_call_time(monkeypatch):
    """Patching lgtools.kernel.event_dispatch.dispatch_event after import must
    still be what the builder calls (the hook looks it up per call)."""
    from lgtools.kernel.graph_builder import build_graph
    from lgtools.kernel.registry import NodeDef, register
    from lgtools.kernel.schemas import NodeSpec, WorkflowSpec

    seen: list[str] = []
    monkeypatch.setattr(
        "lgtools.kernel.event_dispatch.dispatch_event",
        lambda event_type, state, **kw: seen.append(event_type) or {},
    )
    register(
        "custom",
        NodeDef(kind="custom", fn=lambda s, p, pr, c=None, r=None: {}, default_prompt="", description="t"),
    )
    build_graph(WorkflowSpec(nodes=[NodeSpec(id="n", kind="custom")], entry="n")).invoke({"task": "t"})
    assert seen == ["node.on_enter", "node.on_exit"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_lgkit_shims.py -q`
Expected: `test_builder_shim_reexports` FAILS (`GraphBuildError` is still lgtools' own class).

- [ ] **Step 3: Replace `src/lgtools/kernel/graph_builder.py`**

```python
"""Graph builder shim. The builder lives in lgkit.builder; this module binds
lgtools' stores and event scripts to it through BuildHooks and keeps the old
``build_graph(spec_or_resolved, checkpointer, store, dry_run)`` signature.

Each hook imports its target at call time so monkeypatching the lgtools
function (tests do) is honored."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lgkit.builder import (
    GraphBuildError,
    _iteration_guard,
    detect_workflow_cycle,
    validate_goto_agents,
    validate_map_nodes,
    validate_parallel_branches,
)
from lgkit.builder import build_graph as _lgkit_build_graph
from lgkit.hooks import BuildHooks

if TYPE_CHECKING:
    from lgtools.kernel.resolved import ResolvedSpec
    from lgtools.kernel.schemas import WorkflowSpec


def _agent_resolver(aid: Any):
    from lgtools.kernel.agents import resolve_agent

    try:
        return resolve_agent(aid)
    except (KeyError, TypeError, ValueError):
        return None


def _workflow_resolver(ref: str):
    from lgtools.kernel.workflows import resolve_workflow

    return resolve_workflow(ref)


def _dispatch_event(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from lgtools.kernel import event_dispatch

    return event_dispatch.dispatch_event(*args, **kwargs)


LGTOOLS_HOOKS = BuildHooks(
    agent_resolver=_agent_resolver,
    workflow_resolver=_workflow_resolver,
    dispatch_event=_dispatch_event,
)


def build_graph(
    spec: WorkflowSpec | ResolvedSpec,
    checkpointer: Any = None,
    store: Any = None,
    dry_run: bool = False,
) -> Any:
    """Compile a WorkflowSpec (or an already-resolved unit) into a runnable LangGraph."""
    from lgtools.kernel.resolved import ResolvedSpec, Resolver

    if isinstance(spec, ResolvedSpec):
        resolved, spec = spec, spec.root
    else:
        resolved = Resolver.resolve(spec)
    return _lgkit_build_graph(
        spec, checkpointer, store, dry_run, hooks=LGTOOLS_HOOKS, resolved=resolved
    )


__all__ = [
    "build_graph",
    "GraphBuildError",
    "detect_workflow_cycle",
    "validate_goto_agents",
    "validate_map_nodes",
    "validate_parallel_branches",
    "LGTOOLS_HOOKS",
]
```

(`_dispatch_event` looks up `event_dispatch.dispatch_event` as a module attribute on every call — that is what makes the monkeypatch test pass.)

- [ ] **Step 4: Run the new tests, then the full suite against the baseline**

```bash
uv run pytest tests/test_lgkit_shims.py -q
uv run pytest -q -rA -p no:cacheprovider 2>&1 | grep -E '^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) ' \
  | sed -E 's/ - .*//' | grep -v 'test_lgkit_shims' | sort > /tmp/lgtools-after.txt
diff .pytest-baseline.txt /tmp/lgtools-after.txt && echo SAME
```
Expected: shim tests `10 passed`; `SAME`.

- [ ] **Step 5: Confirm lgkit is still standalone and lgtools no longer duplicates the kernel**

```bash
grep -rn "lgtools" $LGA/src && echo "LEAK" || echo "lgkit clean"
wc -l src/lgtools/kernel/graph_builder.py src/lgtools/kernel/schemas.py src/lgtools/llm.py
cd $LGA && uv run pytest -q
```
Expected: `lgkit clean`; graph_builder ≈ 90 lines, schemas ≈ 270 lines, llm ≈ 35 lines; lgkit suite passes (100).

- [ ] **Step 6: Remove the baseline file and commit**

```bash
cd $LGT && rm .pytest-baseline.txt
git add src/lgtools/kernel/graph_builder.py tests/test_lgkit_shims.py
git commit -m "refactor: graph_builder delegates to lgkit via LGTOOLS_HOOKS

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017F1keyvotGL1g5aRamQaeb"
```

---

## Follow-ups (not in this plan)

- lgtools' node palette will list `advisor` and `approval` after the first build that registers them, but the frontend has no `NodeKind`/`NODE_COLORS`/Inspector entries for them yet (they render grey). Add frontend support, or filter lgkit kinds out of `node_palette()`, before using them from the editor.
- `agent_node._flat` in lgtools duplicates `lgkit.nodes._template.flat_state`; switch it over once lgtools' agent nodes move.
- Next patterns from the self-audit: plan-execute-review with `max_rounds`, keyword→LLM router, fan-out→synthesize (committee).
