"""Render node message templates against graph state.

``{scratch.key}`` works because format_map looks up ``scratch`` and then does
attribute access; _AttrDict makes that attribute access succeed, and missing
keys render as "" instead of raising.
"""

from __future__ import annotations


class _Blank(str):
    """What a missing key renders as: "" that also tolerates further
    attribute access, so ``{scratch.review.feedback}`` is empty (not an
    AttributeError) before any review exists."""

    def __getattr__(self, key: str):
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        return self


_BLANK = _Blank("")


class _DefaultDict(dict):
    def __missing__(self, key):
        return _BLANK


class _AttrDict(dict):
    def __missing__(self, key: str):
        return _BLANK

    def __getattr__(self, key: str):
        # Ephemeral wrapper — must not be persisted into graph state/checkpoint.
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        value = self.get(key, _BLANK)
        if value is None:
            # Present but None renders like a missing key: "None" leaking into
            # a template reads like real content (e.g. reviewer feedback).
            return _BLANK
        return _AttrDict(value) if isinstance(value, dict) else value


def flat_state(state: dict) -> dict:
    flat = _DefaultDict(state)
    for key, value in state.items():
        if value is None:
            flat[key] = _BLANK
    flat["scratch"] = _AttrDict(state.get("scratch") or {})
    return flat


__all__ = ["flat_state"]
