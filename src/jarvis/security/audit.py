from __future__ import annotations

from copy import deepcopy
import re
import uuid
from typing import Any

from jarvis.config.validation import normalize_audit_identifier


_OUTCOMES = frozenset({"allowed", "denied", "error"})
_SESSION_ID_PATTERN = r"session-[0-9a-f]{8}"
_ACTION_REF_PATTERN = r"(?:empty_input|echo:[0-9a-f]{12})"
_DETAIL_VALUES = {
    "reason": frozenset({"tool_not_registered", "authorization_failed", "invalid_classification", "invalid_proposal", "provider_failure"}),
    "authorization": frozenset({"not_attempted", "not_verified", "denied", "granted"}),
    "execution_status": frozenset({"denied", "blocked", "error", "completed"}),
    "error_code": frozenset({"tool_execution_failed", "provider_failure", "internal_failure"}),
    "result_code": frozenset({"simulated_ok", "ok"}),
    "identifier_status": frozenset({"invalid_session", "invalid_action_ref", "invalid_both"}),
}
_TOOL_NAME_PATTERN = r"[a-z][a-z0-9_]{0,63}\Z"


class AuditLogger:
    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    @property
    def entries(self) -> list[dict[str, Any]]:
        return deepcopy(self._entries)

    @staticmethod
    def _sanitize_details(details: dict[str, Any]) -> dict[str, Any]:
        if type(details) is not dict:
            return {}

        cleaned: dict[str, str] = {}
        for key, allowed_values in _DETAIL_VALUES.items():
            value = details.get(key)
            if type(value) is str and value in allowed_values:
                cleaned[key] = value

        tool_name = details.get("tool_name")
        if (
            type(tool_name) is str
            and len(tool_name) <= 64
            and re.fullmatch(_TOOL_NAME_PATTERN, tool_name)
        ):
            cleaned["tool_name"] = tool_name
        return cleaned

    def log(self, *, session_id: str, action_ref: str, outcome: str, details: dict[str, Any]) -> dict[str, Any]:
        safe_outcome = outcome if type(outcome) is str and outcome in _OUTCOMES else "error"
        safe_session_id = normalize_audit_identifier(
            session_id,
            pattern=_SESSION_ID_PATTERN,
            fallback="invalid-session",
        )
        safe_action_ref = normalize_audit_identifier(
            action_ref,
            pattern=_ACTION_REF_PATTERN,
            fallback="invalid-action",
        )
        safe_details = self._sanitize_details(details)
        if safe_session_id == "invalid-session" or safe_action_ref == "invalid-action":
            if safe_session_id == "invalid-session" and safe_action_ref == "invalid-action":
                safe_details["identifier_status"] = "invalid_both"
            elif safe_session_id == "invalid-session":
                safe_details["identifier_status"] = "invalid_session"
            else:
                safe_details["identifier_status"] = "invalid_action_ref"
        entry = {
            "audit_id": uuid.uuid4().hex[:12],
            "session_id": safe_session_id,
            "action_ref": safe_action_ref,
            "outcome": safe_outcome,
            "details": safe_details,
        }
        self._entries.append(deepcopy(entry))
        return deepcopy(entry)
