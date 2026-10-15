from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class UserInput:
    text: str


@dataclass(slots=True)
class ToolProposal:
    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    permission: str = ""
    risk_level: str = "low"
    requires_confirmation: bool = False
    confirmation_token: str | None = None
    action_ref: str = ""


@dataclass(slots=True)
class ConversationResult:
    user_text: str
    intent: str
    proposal: ToolProposal | None
    authorized: bool
    success: bool
    tool_result: dict[str, Any] | None = None
    error: str | None = None
    audit_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_text": self.user_text,
            "intent": self.intent,
            "proposal": self.proposal.__dict__ if self.proposal else None,
            "authorized": self.authorized,
            "success": self.success,
            "tool_result": self.tool_result,
            "error": self.error,
            "audit_id": self.audit_id,
        }
