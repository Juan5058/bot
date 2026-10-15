from __future__ import annotations

from typing import Any

from jarvis.conversation.models import ToolProposal
from jarvis.core.context import ConversationContext
from jarvis.core.errors import AuthorizationError, SimulatedModeError
from jarvis.security.authorization import AuthorizationService
from jarvis.tools.registry import ToolRegistry


class SimulatedToolRunner:
    def __init__(self, registry: ToolRegistry, authorization_service: AuthorizationService, mode: str = "simulated") -> None:
        self.registry = registry
        self.authorization_service = authorization_service
        self.mode = mode

    def execute(self, proposal: ToolProposal, context: ConversationContext) -> dict[str, Any]:
        if self.mode != "simulated":
            raise SimulatedModeError("Only simulated mode is allowed in Fase 1.")
        if not self.authorization_service.authorize(proposal, context):
            raise AuthorizationError(f"Authorization denied for tool '{proposal.tool_name}'.")
        tool = self.registry.get(proposal.tool_name)
        if tool.execution_mode != "simulated":
            raise SimulatedModeError("Real execution path is disabled.")
        result = tool.execute(context, **proposal.parameters)
        return {
            "tool_name": proposal.tool_name,
            "status": "simulated_ok",
            "result": result,
        }
