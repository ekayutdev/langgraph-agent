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
