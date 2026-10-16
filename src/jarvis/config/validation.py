from __future__ import annotations

import re
from typing import Any

_PERMISSION_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\Z")


def validate_permission_name(permission_name: str) -> str:
    if type(permission_name) is not str:
        raise TypeError("Permission name must be a string.")
    if not permission_name or not permission_name.strip():
        raise ValueError("Permission name cannot be empty.")
    if not _PERMISSION_NAME_PATTERN.fullmatch(permission_name):
        raise ValueError("Permission name has an invalid format.")
    return permission_name


def validate_mode(mode: str) -> str:
    if mode != "simulated":
        raise ValueError("Only simulated mode is supported in Fase 1.")
    return mode


def normalize_audit_identifier(value: Any, *, pattern: str, fallback: str) -> str:
    if type(value) is str and re.fullmatch(pattern, value):
        return value
    return fallback


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
