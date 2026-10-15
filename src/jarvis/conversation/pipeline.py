from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.ai.contracts import AIProviderContract
from jarvis.conversation.models import ConversationResult, UserInput
from jarvis.core.context import ConversationContext
from jarvis.security.audit import AuditLogger
from jarvis.security.authorization import AuthorizationService
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.runner import SimulatedToolRunner


@dataclass(slots=True)
class ConversationPipeline:
    ai_provider: AIProviderContract
    authorizer: AuthorizationService
    registry: ToolRegistry
    tool_runner: SimulatedToolRunner
    audit_logger: AuditLogger
    settings: Any

    def process(self, user_text: str) -> ConversationResult:
        user_input = UserInput(text=user_text)
        context = ConversationContext.create(user_text=user_input.text)
        intent = self.ai_provider.classify_intent(user_input.text)
        proposal = self.ai_provider.propose_action(user_input.text, context)

        if not self.registry.has(proposal.tool_name):
            audit_entry = self.audit_logger.log(
                session_id=context.session_id,
                action_ref=proposal.action_ref,
                outcome="denied",
                details={"reason": "tool_not_registered", "tool_name": proposal.tool_name},
            )
            return ConversationResult(
                user_text=user_input.text,
                intent=str(intent.get("intent", "unknown")),
                proposal=proposal,
                authorized=False,
                success=False,
                error="Tool is not registered.",
                audit_id=audit_entry["audit_id"],
            )

        if not self.authorizer.authorize(proposal, context):
            audit_entry = self.audit_logger.log(
                session_id=context.session_id,
                action_ref=proposal.action_ref,
                outcome="denied",
                details={"reason": "authorization_failed", "tool": proposal.tool_name},
            )
            return ConversationResult(
                user_text=user_input.text,
                intent=str(intent.get("intent", "unknown")),
                proposal=proposal,
                authorized=False,
                success=False,
                error="Authorization failed.",
                audit_id=audit_entry["audit_id"],
            )

        result = self.tool_runner.execute(proposal, context)
        audit_entry = self.audit_logger.log(
            session_id=context.session_id,
            action_ref=proposal.action_ref,
            outcome="allowed",
            details={"tool": proposal.tool_name, "result": result},
        )
        return ConversationResult(
            user_text=user_input.text,
            intent=str(intent.get("intent", "unknown")),
            proposal=proposal,
            authorized=True,
            success=True,
            tool_result=result,
            audit_id=audit_entry["audit_id"],
        )
