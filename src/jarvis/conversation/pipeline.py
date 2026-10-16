from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.ai.contracts import AIProviderContract
from jarvis.conversation.models import (
    MAX_USER_INPUT_LENGTH,
    ConversationResult,
    ToolProposal,
    UserInput,
)
from jarvis.core.context import ConversationContext
from jarvis.core.errors import (
    AuthorizationError,
    InvalidInputError,
    SimulatedModeError,
    ToolExecutionError,
    ToolNotRegisteredError,
)
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

    def _audit_tool_name(self, proposal: ToolProposal) -> str | None:
        if type(proposal.tool_name) is str and self.registry.has(proposal.tool_name):
            return proposal.tool_name
        return None

    def process(self, user_text: str) -> ConversationResult:
        if type(user_text) is not str:
            raise InvalidInputError("User input must be text.")
        if len(user_text) > MAX_USER_INPUT_LENGTH:
            raise InvalidInputError(
                f"User input cannot exceed {MAX_USER_INPUT_LENGTH} characters."
            )
        if not user_text.strip():
            raise InvalidInputError("User input cannot be empty.")

        user_input = UserInput(text=user_text)
        context = ConversationContext.create(user_text=user_input.text)
        session_id = context.session_id
        trace_id = context.trace_id

        def restore_context() -> None:
            context.session_id = session_id
            context.trace_id = trace_id
            context.user_text = user_input.text
            context.mode = "simulated"
            context.claims = {}
            context.confirmed_actions = set()
            context.metadata = {}

        def context_was_modified() -> bool:
            return (
                type(context.session_id) is not str
                or context.session_id != session_id
                or type(context.trace_id) is not str
                or context.trace_id != trace_id
                or type(context.user_text) is not str
                or context.user_text != user_input.text
                or type(context.mode) is not str
                or context.mode != "simulated"
                or type(context.claims) is not dict
                or bool(context.claims)
                or type(context.confirmed_actions) is not set
                or bool(context.confirmed_actions)
                or type(context.metadata) is not dict
                or bool(context.metadata)
            )

        try:
            intent_data = self.ai_provider.classify_intent(user_input.text)
        except Exception:
            audit_entry = self.audit_logger.log(
                session_id=session_id,
                action_ref="empty_input",
                outcome="error",
                details={
                    "reason": "provider_failure",
                    "authorization": "not_attempted",
                    "execution_status": "error",
                    "error_code": "provider_failure",
                },
            )
            return ConversationResult(
                user_text=user_input.text,
                intent="unknown",
                proposal=None,
                authorized=False,
                success=False,
                error="Request could not be processed.",
                audit_id=audit_entry["audit_id"],
            )

        if (
            type(intent_data) is not dict
            or any(type(key) is not str for key in intent_data)
            or type(intent_data.get("intent")) is not str
            or intent_data["intent"] != "text_request"
        ):
            audit_entry = self.audit_logger.log(
                session_id=session_id,
                action_ref="empty_input",
                outcome="denied",
                details={
                    "reason": "invalid_classification",
                    "authorization": "not_attempted",
                    "execution_status": "blocked",
                },
            )
            return ConversationResult(
                user_text=user_input.text,
                intent="unknown",
                proposal=None,
                authorized=False,
                success=False,
                error="Request was denied.",
                audit_id=audit_entry["audit_id"],
            )

        try:
            proposal = self.ai_provider.propose_action(user_input.text, context)
        except Exception:
            restore_context()
            audit_entry = self.audit_logger.log(
                session_id=session_id,
                action_ref="empty_input",
                outcome="error",
                details={
                    "reason": "provider_failure",
                    "authorization": "not_attempted",
                    "execution_status": "error",
                    "error_code": "provider_failure",
                },
            )
            return ConversationResult(
                user_text=user_input.text,
                intent="unknown",
                proposal=None,
                authorized=False,
                success=False,
                error="Request could not be processed.",
                audit_id=audit_entry["audit_id"],
            )

        if context_was_modified():
            restore_context()
            audit_entry = self.audit_logger.log(
                session_id=session_id,
                action_ref=f"echo:{trace_id}",
                outcome="denied",
                details={
                    "reason": "invalid_proposal",
                    "authorization": "not_attempted",
                    "execution_status": "blocked",
                },
            )
            return ConversationResult(
                user_text=user_input.text,
                intent="unknown",
                proposal=None,
                authorized=False,
                success=False,
                error="Request was denied.",
                audit_id=audit_entry["audit_id"],
            )

        intent = intent_data["intent"]
        if type(proposal) is not ToolProposal:
            audit_entry = self.audit_logger.log(
                session_id=context.session_id,
                action_ref="empty_input",
                outcome="denied",
                details={
                    "reason": "invalid_proposal",
                    "authorization": "not_attempted",
                    "execution_status": "blocked",
                },
            )
            return ConversationResult(
                user_text=user_input.text,
                intent=intent,
                proposal=None,
                authorized=False,
                success=False,
                error="Request was denied.",
                audit_id=audit_entry["audit_id"],
            )

        try:
            result = self.tool_runner.execute(proposal, context)
        except (AuthorizationError, SimulatedModeError, ToolNotRegisteredError):
            details = {
                "reason": "authorization_failed",
                "authorization": "denied",
                "execution_status": "blocked",
            }
            tool_name = self._audit_tool_name(proposal)
            if tool_name is not None:
                details["tool_name"] = tool_name
            audit_entry = self.audit_logger.log(
                session_id=context.session_id,
                action_ref=proposal.action_ref,
                outcome="denied",
                details=details,
            )
            return ConversationResult(
                user_text=user_input.text,
                intent=intent,
                proposal=proposal,
                authorized=False,
                success=False,
                error="Request was denied.",
                audit_id=audit_entry["audit_id"],
            )
        except ToolExecutionError:
            details = {
                "authorization": "granted",
                "execution_status": "error",
                "error_code": "tool_execution_failed",
            }
            tool_name = self._audit_tool_name(proposal)
            if tool_name is not None:
                details["tool_name"] = tool_name
            audit_entry = self.audit_logger.log(
                session_id=context.session_id,
                action_ref=proposal.action_ref,
                outcome="error",
                details=details,
            )
            return ConversationResult(
                user_text=user_input.text,
                intent=intent,
                proposal=proposal,
                authorized=True,
                success=False,
                error="Tool execution failed.",
                audit_id=audit_entry["audit_id"],
            )
        except Exception:
            details = {
                "authorization": "not_verified",
                "execution_status": "blocked",
                "error_code": "internal_failure",
            }
            tool_name = self._audit_tool_name(proposal)
            if tool_name is not None:
                details["tool_name"] = tool_name
            audit_entry = self.audit_logger.log(
                session_id=context.session_id,
                action_ref=proposal.action_ref,
                outcome="error",
                details=details,
            )
            return ConversationResult(
                user_text=user_input.text,
                intent=intent,
                proposal=proposal,
                authorized=False,
                success=False,
                error="Request could not be processed.",
                audit_id=audit_entry["audit_id"],
            )

        audit_entry = self.audit_logger.log(
            session_id=context.session_id,
            action_ref=proposal.action_ref,
            outcome="allowed",
            details={
                "tool_name": self._audit_tool_name(proposal),
                "authorization": "granted",
                "execution_status": "completed",
                "result_code": result.get("status") if isinstance(result, dict) else None,
            },
        )
        return ConversationResult(
            user_text=user_input.text,
            intent=intent,
            proposal=proposal,
            authorized=True,
            success=True,
            tool_result=result,
            audit_id=audit_entry["audit_id"],
        )
