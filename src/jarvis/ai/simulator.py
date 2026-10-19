from __future__ import annotations

from typing import Any

from jarvis.ai.contracts import (
    AIProviderContract,
    ActionSuggestion,
    InformationalRequest,
    InformationalResponse,
)


class SimulatedAIProvider(AIProviderContract):
    def classify_intent(self, user_text: str) -> dict[str, Any]:
        text = (user_text or "").strip()
        return {
            "intent": "text_request",
            "confidence": 1.0,
            "normalized_text": text,
        }

    def propose_action(self, user_text: str) -> ActionSuggestion:
        text = (user_text or "").strip()
        return ActionSuggestion(
            tool_name="echo_message",
            parameters={"message": text},
        )

    def respond(self, user_text: str) -> str:
        return f"Respuesta simulada para: {user_text}"

    def respond_informational(
        self,
        request: InformationalRequest,
    ) -> InformationalResponse:
        if type(request) is not InformationalRequest:
            raise TypeError("The simulated provider requires an informational request.")
        return InformationalResponse(
            text=f"Respuesta simulada para: {request.content}"
        )
