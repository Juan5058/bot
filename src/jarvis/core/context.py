from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass(slots=True)
class ConversationContext:
    session_id: str
    user_text: str
    mode: str = "simulated"
    claims: dict[str, Any] = field(default_factory=dict)
    confirmed_actions: set[str] = field(default_factory=set)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, user_text: str, claims: dict[str, Any] | None = None) -> "ConversationContext":
        return cls(
            session_id=f"session-{uuid.uuid4().hex[:8]}",
            user_text=user_text,
            mode="simulated",
            claims=claims or {},
            confirmed_actions=set(),
        )
