from __future__ import annotations

import uuid
from typing import Any

from jarvis.config.validation import redact_sensitive


class AuditLogger:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def log(self, *, session_id: str, action_ref: str, outcome: str, details: dict[str, Any]) -> dict[str, Any]:
        clean_details = redact_sensitive(details)
        entry = {
            "audit_id": uuid.uuid4().hex[:12],
            "session_id": session_id,
            "action_ref": action_ref,
            "outcome": outcome,
            "details": clean_details,
        }
        self.entries.append(entry)
        return entry
