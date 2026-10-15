from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from jarvis.conversation.models import ToolProposal
from jarvis.core.context import ConversationContext


class AIProvider(ABC):
    @abstractmethod
    def classify_intent(self, user_text: str) -> dict[str, Any]:
        """Return a typed intent structure."""

    @abstractmethod
    def propose_action(self, user_text: str, context: ConversationContext) -> ToolProposal:
        """Return a proposed action without execution authority."""

    @abstractmethod
    def respond(self, user_text: str, context: ConversationContext) -> str:
        """Return a final natural-language response."""


class ToolDefinition(ABC):
    name: str
    description: str
    required_permission: str
    risk_level: str = "low"
    requires_confirmation: bool = False
    execution_mode: str = "simulated"

    @abstractmethod
    def execute(self, context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        """Execute a simulated tool only."""
