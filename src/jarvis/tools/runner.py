from __future__ import annotations

from typing import Any

from jarvis.conversation.models import ToolProposal, validate_tool_parameters
from jarvis.core.context import ConversationContext
from jarvis.core.errors import (
    AuthorizationError,
    SimulatedModeError,
    ToolExecutionError,
    ToolNotRegisteredError,
)
from jarvis.security.authorization import AuthorizationService
from jarvis.tools.registry import ToolRegistry


class SimulatedToolRunner:
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_registry", "_authorization_service"} and hasattr(self, name):
            raise AttributeError(f"{name} is immutable after runner construction.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in {"_registry", "_authorization_service"}:
            raise AttributeError(f"{name} cannot be deleted from a tool runner.")
        super().__delattr__(name)

    def __init__(self, registry: ToolRegistry, authorization_service: AuthorizationService, mode: str = "simulated") -> None:
        if type(registry) is not ToolRegistry:
            raise TypeError("Runner requires the core tool registry.")
        if registry.is_frozen is not True:
            raise ValueError("Runner requires a frozen tool registry.")
        if type(authorization_service) is not AuthorizationService:
            raise TypeError("Runner requires the core authorization service.")
        if registry is not authorization_service.registry:
            raise ValueError("Runner and authorization service must share the same tool registry.")
        self._registry = registry
        self._authorization_service = authorization_service
        self.mode = mode

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def authorization_service(self) -> AuthorizationService:
        return self._authorization_service

    def execute(self, proposal: ToolProposal, context: ConversationContext) -> dict[str, Any]:
        if (
            type(self.registry) is not ToolRegistry
            or self.registry.is_frozen is not True
            or type(self.authorization_service) is not AuthorizationService
            or self.registry is not self.authorization_service.registry
        ):
            raise AuthorizationError("Authorization could not be verified.")
        if type(self.mode) is not str or self.mode != "simulated":
            raise SimulatedModeError("Only simulated mode is allowed in Fase 1.")
        if type(proposal) is not ToolProposal or type(proposal.tool_name) is not str:
            raise AuthorizationError("Authorization denied.")
        try:
            validate_tool_parameters(proposal.parameters)
        except Exception:
            raise AuthorizationError("Authorization denied.") from None
        try:
            tool = self.registry.get(proposal.tool_name)
        except ToolNotRegisteredError:
            raise
        if type(tool.execution_mode) is not str or tool.execution_mode != "simulated":
            raise SimulatedModeError("Real execution path is disabled.")
        try:
            authorized = self.authorization_service.authorize(proposal, context)
        except ToolNotRegisteredError:
            raise
        except Exception:
            raise AuthorizationError("Authorization could not be verified.") from None
        if authorized is not True:
            raise AuthorizationError("Authorization denied.")
        try:
            result = tool.execute(context, **proposal.parameters)
        except Exception:
            raise ToolExecutionError("Simulated tool execution failed.") from None
        return {
            "tool_name": proposal.tool_name,
            "status": "simulated_ok",
            "result": result,
        }
