from __future__ import annotations

from typing import Any


def validate_permission_name(permission_name: str) -> str:
    if not permission_name or not permission_name.strip():
        raise ValueError("Permission name cannot be empty.")
    if permission_name.strip() != permission_name:
        raise ValueError("Permission name must not include leading or trailing spaces.")
    return permission_name


def validate_mode(mode: str) -> str:
    if mode != "simulated":
        raise ValueError("Only simulated mode is supported in Fase 1.")
    return mode


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and any(token in key.lower() for token in ("secret", "token", "password", "api_key", "credential", "biometric")):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value
