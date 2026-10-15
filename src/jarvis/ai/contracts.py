from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from jarvis.conversation.models import ToolProposal
from jarvis.core.context import ConversationContext


class AIProviderContract(ABC):
    @abstractmethod
    def classify_intent(self, user_text: str) -> dict[str, Any]:
        """Classify a user request."""

    @abstractmethod
    def propose_action(self, user_text: str, context: ConversationContext) -> ToolProposal:
        """Return a structured proposal, never an execution command."""

    @abstractmethod
    def respond(self, user_text: str, context: ConversationContext) -> str:
        """Return a response to the user."""
