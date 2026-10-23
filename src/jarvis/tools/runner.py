from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any

from jarvis.conversation.models import ToolProposal
from jarvis.core.context import ConversationContext
from jarvis.core.errors import (
    AuthorizationError,
    PreExecutionAuditError,
    SimulatedModeError,
    ToolExecutionError,
    ToolNotRegisteredError,
)
from jarvis.security.audit import AuditLogger
from jarvis.security.authorization import AuthorizationService
from jarvis.tools.registry import ToolRegistry, _CORE_TRUSTED_CATALOG


def _execute_echo_message(
    context: ConversationContext,
    **kwargs: Any,
) -> dict[str, Any]:
    return {
        "kind": "simulated_echo",
        "message": kwargs["message"],
        "status": "ok",
        "execution_mode": "simulated",
    }


def _dispatch_echo_message(
    context: ConversationContext,
    **kwargs: Any,
) -> dict[str, Any]:
    return _execute_echo_message(context, **kwargs)


_TRUSTED_EXECUTORS: Mapping[str, Callable[..., dict[str, Any]]] = (
    MappingProxyType({"echo_message": _dispatch_echo_message})
)


class _SimulatedToolDispatcher:
    __slots__ = (
        "_registry",
        "_authorization_service",
        "_executors",
        "_audit_logger",
        "mode",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {
            "_registry",
            "_authorization_service",
            "_executors",
            "_audit_logger",
            "mode",
        } and hasattr(self, name):
            raise AttributeError(f"{name} is immutable after runner construction.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in {
            "_registry",
            "_authorization_service",
            "_executors",
            "_audit_logger",
            "mode",
        }:
            raise AttributeError(f"{name} cannot be deleted from a tool runner.")
        super().__delattr__(name)

    def __init__(
        self,
        registry: ToolRegistry,
        authorization_service: AuthorizationService,
        *,
        mode: str,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        if type(registry) is not ToolRegistry:
            raise TypeError("Runner requires the core tool registry.")
        if registry.is_frozen is not True:
            raise ValueError("Runner requires a frozen tool registry.")
        if type(authorization_service) is not AuthorizationService:
            raise TypeError("Runner requires the core authorization service.")
        if registry is not authorization_service.registry:
            raise ValueError("Runner and authorization service must share the same tool registry.")
        if registry.list_names() != sorted(_CORE_TRUSTED_CATALOG):
            raise ValueError("Dispatcher only supports the core tool catalog.")
        for tool_name, trusted_definition in _CORE_TRUSTED_CATALOG.items():
            if registry.get_trusted_definition(tool_name) != trusted_definition:
                raise ValueError("Dispatcher only supports trusted core definitions.")
        if audit_logger is not None and type(audit_logger) is not AuditLogger:
            raise TypeError("Runner requires the core audit logger.")
        self._registry = registry
        self._authorization_service = authorization_service
        self._executors = _TRUSTED_EXECUTORS
        self._audit_logger = audit_logger if audit_logger is not None else AuditLogger()
        self.mode = mode

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def authorization_service(self) -> AuthorizationService:
        return self._authorization_service

    @property
    def audit_logger(self) -> AuditLogger:
        return self._audit_logger

    def execute(self, proposal: ToolProposal, context: ConversationContext) -> dict[str, Any]:
        if (
            isinstance(self, SimulatedToolRunner)
            and self._executors is not _TRUSTED_EXECUTORS
        ):
            raise AuthorizationError("Authorization could not be verified.")
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
            self.registry.validate_tool_arguments(
                proposal.tool_name,
                proposal.parameters,
            )
            execution_proposal = ToolProposal(
                tool_name=proposal.tool_name,
                parameters=deepcopy(proposal.parameters),
                permission=proposal.permission,
                risk_level=proposal.risk_level,
                requires_confirmation=proposal.requires_confirmation,
                confirmation_token=proposal.confirmation_token,
                action_ref=proposal.action_ref,
            )
            self.registry.validate_tool_arguments(
                execution_proposal.tool_name,
                execution_proposal.parameters,
            )
        except ToolNotRegisteredError:
            raise
        except Exception:
            raise AuthorizationError("Authorization denied.") from None
        try:
            trusted_definition = self.registry.get_trusted_definition(
                execution_proposal.tool_name
            )
        except ToolNotRegisteredError:
            raise
        if trusted_definition.execution_mode != "simulated":
            raise SimulatedModeError("Real execution path is disabled.")
        try:
            authorized = self.authorization_service.authorize(execution_proposal, context)
        except ToolNotRegisteredError:
            raise
        except Exception:
            raise AuthorizationError("Authorization could not be verified.") from None
        if authorized is not True:
            raise AuthorizationError("Authorization denied.")
        try:
            self.audit_logger.log(
                session_id=context.session_id,
                action_ref=execution_proposal.action_ref,
                outcome="allowed",
                details={
                    "authorization": "granted",
                    "execution_status": "authorized",
                    "tool_name": execution_proposal.tool_name,
                },
            )
        except Exception:
            raise PreExecutionAuditError() from None
        if (
            isinstance(self, SimulatedToolRunner)
            and self._executors is not _TRUSTED_EXECUTORS
        ):
            raise AuthorizationError("Authorization could not be verified.")
        try:
            executor = self._executors[execution_proposal.tool_name]
            result = executor(context, **execution_proposal.parameters)
        except Exception:
            raise ToolExecutionError("Simulated tool execution failed.") from None
        return {
            "tool_name": execution_proposal.tool_name,
            "status": "simulated_ok",
            "result": result,
        }


class SimulatedToolRunner(_SimulatedToolDispatcher):
    __slots__ = ()

    def __init__(
        self,
        registry: ToolRegistry,
        authorization_service: AuthorizationService,
        mode: str = "simulated",
        *,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        if type(mode) is not str or mode != "simulated":
            raise SimulatedModeError("Only simulated mode is allowed in Fase 1.")
        if type(registry) is not ToolRegistry:
            raise TypeError("Runner requires the core tool registry.")
        if registry.is_frozen is not True:
            raise ValueError("Runner requires a frozen tool registry.")
        if type(authorization_service) is not AuthorizationService:
            raise TypeError("Runner requires the core authorization service.")
        if registry is not authorization_service.registry:
            raise ValueError("Runner and authorization service must share the same tool registry.")
        if registry.list_names() != sorted(_CORE_TRUSTED_CATALOG):
            raise ValueError("Production runner only supports the core tool catalog.")
        for tool_name, trusted_definition in _CORE_TRUSTED_CATALOG.items():
            if registry.get_trusted_definition(tool_name) != trusted_definition:
                raise ValueError("Production runner only supports trusted core definitions.")
        super().__init__(
            registry,
            authorization_service,
            mode=mode,
            audit_logger=audit_logger,
        )
