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
