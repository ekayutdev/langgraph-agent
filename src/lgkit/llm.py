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

    Defaults carried over from the original implementation: ``enable_thinking=False`` so reasoning models
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
