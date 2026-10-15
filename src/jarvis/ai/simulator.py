from __future__ import annotations

from typing import Any

from jarvis.ai.contracts import AIProviderContract
from jarvis.conversation.models import ToolProposal
from jarvis.core.context import ConversationContext


class SimulatedAIProvider(AIProviderContract):
    def classify_intent(self, user_text: str) -> dict[str, Any]:
        text = (user_text or "").strip()
        return {
            "intent": "text_request",
            "confidence": 1.0,
            "normalized_text": text,
        }

    def propose_action(self, user_text: str, context: ConversationContext) -> ToolProposal:
        text = (user_text or "").strip()
        if not text:
            return ToolProposal(
                tool_name="echo_message",
                parameters={"message": ""},
                permission="tool.echo_message",
                risk_level="low",
                requires_confirmation=False,
                action_ref="empty_input",
            )
        return ToolProposal(
            tool_name="echo_message",
            parameters={"message": text},
            permission="tool.echo_message",
            risk_level="low",
            requires_confirmation=False,
            action_ref=f"echo:{context.trace_id}",
        )

    def respond(self, user_text: str, context: ConversationContext) -> str:
        return f"Respuesta simulada para: {user_text}"
