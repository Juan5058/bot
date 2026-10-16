from __future__ import annotations

import re

from jarvis.config.validation import validate_permission_name
from jarvis.conversation.models import ToolProposal, validate_tool_parameters
from jarvis.core.context import ConversationContext
from jarvis.core.errors import ToolNotRegisteredError
from jarvis.security.policy import SecurityPolicy
from jarvis.tools.registry import ToolRegistry


_SESSION_ID_PATTERN = re.compile(r"session-[0-9a-f]{8}\Z")
_ACTION_REF_PATTERN = re.compile(r"(?:|[A-Za-z0-9_.:-]{1,128})\Z")
_TOOL_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_RISK_LEVELS = frozenset({"low", "medium", "high"})


class AuthorizationService:
    def __setattr__(self, name: str, value: object) -> None:
        if name in {"policy", "_registry"} and hasattr(self, name):
            raise AttributeError(f"{name} is immutable after authorization service construction.")
        if name == "authorize":
            raise AttributeError("The authorization decision method cannot be replaced on an instance.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in {"policy", "_registry", "authorize"}:
            raise AttributeError(f"{name} cannot be deleted from an authorization service.")
        super().__delattr__(name)

    def __init__(self, policy: SecurityPolicy, registry: ToolRegistry) -> None:
        if type(policy) is not SecurityPolicy:
            raise TypeError("Authorization service requires the core security policy.")
        if type(registry) is not ToolRegistry:
            raise TypeError("Authorization service requires the core tool registry.")
        if registry.is_frozen is not True:
            raise ValueError("Authorization service requires a frozen tool registry.")
        self.policy = policy
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def authorize(self, proposal: ToolProposal, context: ConversationContext) -> bool:
        registry = self.registry
        if type(registry) is not ToolRegistry or registry.is_frozen is not True:
            return False
        if type(proposal) is not ToolProposal or type(context) is not ConversationContext:
            return False
        if (
            type(context.mode) is not str
            or context.mode != "simulated"
            or type(context.session_id) is not str
            or not _SESSION_ID_PATTERN.fullmatch(context.session_id)
        ):
            return False
        if (
            type(proposal.tool_name) is not str
            or not _TOOL_NAME_PATTERN.fullmatch(proposal.tool_name)
            or type(proposal.permission) is not str
            or type(proposal.parameters) is not dict
            or any(type(key) is not str for key in proposal.parameters)
            or type(proposal.risk_level) is not str
            or proposal.risk_level not in _RISK_LEVELS
            or type(proposal.requires_confirmation) is not bool
            or (proposal.confirmation_token is not None and type(proposal.confirmation_token) is not str)
            or type(proposal.action_ref) is not str
            or not _ACTION_REF_PATTERN.fullmatch(proposal.action_ref)
        ):
            return False

        try:
            validate_tool_parameters(proposal.parameters)
        except (TypeError, ValueError):
            return False

        try:
            permission = validate_permission_name(proposal.permission)
            tool = self.registry.get(proposal.tool_name)
            trusted_definition = self.registry.get_trusted_definition(proposal.tool_name)
        except ToolNotRegisteredError:
            raise
        except (TypeError, ValueError, KeyError):
            return False
        except Exception:
            return False

        if (
            type(tool) is not trusted_definition.implementation_type
            or type(tool.name) is not str
            or tool.name != proposal.tool_name
            or type(trusted_definition.required_permission) is not str
            or type(trusted_definition.execution_mode) is not str
            or trusted_definition.execution_mode != "simulated"
            or trusted_definition.risk_level not in _RISK_LEVELS
            or tool.required_permission != trusted_definition.required_permission
            or tool.execution_mode != trusted_definition.execution_mode
            or tool.risk_level != trusted_definition.risk_level
            or tool.requires_confirmation != trusted_definition.requires_confirmation
            or proposal.risk_level != trusted_definition.risk_level
            or proposal.requires_confirmation != trusted_definition.requires_confirmation
            or type(tool._frozen) is not bool
            or not tool._frozen
        ):
            return False

        if (
            trusted_definition.identity_required
            or trusted_definition.requires_confirmation
            or proposal.confirmation_token is not None
        ):
            return False

        return (
            permission == trusted_definition.required_permission
            and self.policy.validate_required_inputs(permission, context)
            and self.policy.can_execute_tool(trusted_definition.name, permission)
            and self.policy.has_rule_for(permission)
        )
