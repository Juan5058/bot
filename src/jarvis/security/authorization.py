from __future__ import annotations

from jarvis.conversation.models import ToolProposal
from jarvis.core.context import ConversationContext
from jarvis.core.errors import AuthorizationError, ToolNotRegisteredError
from jarvis.security.policy import SecurityPolicy
from jarvis.tools.registry import ToolRegistry


class AuthorizationService:
    def __init__(self, policy: SecurityPolicy, registry: ToolRegistry) -> None:
        self.policy = policy
        self.registry = registry

    def authorize(self, proposal: ToolProposal, context: ConversationContext) -> bool:
        if not isinstance(proposal, ToolProposal):
            raise AuthorizationError("Invalid proposal object.")
        if context.mode != "simulated":
            return False
        if not self.registry.has(proposal.tool_name):
            raise ToolNotRegisteredError(f"Tool '{proposal.tool_name}' is not registered.")

        if not proposal.permission:
            return False
        if not self.policy.validate_required_inputs(proposal.permission, context):
            return False
        if not self.policy.can_execute(proposal.permission):
            return False
        if not self.policy.has_rule_for(proposal.permission):
            return False

        if proposal.requires_confirmation:
            if not proposal.confirmation_token:
                return False
            if proposal.action_ref not in context.confirmed_actions:
                return False

        required_tool = self.registry.get(proposal.tool_name)
        if required_tool.required_permission != proposal.permission:
            return False

        return True
