from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.ai.contracts import (
    AIProviderContract,
    InformationalRequest,
    InformationalResponse,
    MAX_INFORMATIONAL_RESPONSE_LENGTH,
)
from jarvis.ai.selection import (
    AIProviderSelector,
    SelectionTarget,
)
from jarvis.conversation.models import (
    MAX_USER_INPUT_LENGTH,
    ConversationResult,
    RequestEnvelope,
    ToolProposal,
    UserInput,
)
from jarvis.conversation.router import (
    Capability,
    DeterministicRequestRouter,
    InterpretationSource,
    InterpretationStatus,
    RouteDecision,
    RouteCategory,
)
from jarvis.core.context import ConversationContext
from jarvis.core.errors import (
    AuditLoggingError,
    AuthorizationError,
    InvalidInputError,
    PostExecutionAuditError,
    PreExecutionAuditError,
    SimulatedModeError,
    ToolExecutionError,
    ToolNotRegisteredError,
)
from jarvis.security.audit import AuditLogger
from jarvis.security.authorization import AuthorizationService
from jarvis.internet import (
    InternetServiceCapability,
    InternetServiceRequest,
    InternetServiceSelection,
    InternetServiceSelector,
    InternetServiceUnavailableReason,
)
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.runner import SimulatedToolRunner


@dataclass(slots=True)
class ConversationPipeline:
    ai_provider: AIProviderContract | None
    authorizer: AuthorizationService
    registry: ToolRegistry
    tool_runner: SimulatedToolRunner
    audit_logger: AuditLogger
    settings: Any
    request_router: DeterministicRequestRouter
    provider_selector: AIProviderSelector | None = None
    internet_service_selector: InternetServiceSelector | None = None

    def __post_init__(self) -> None:
        if self.provider_selector is None:
            providers = (
                {"simulated": self.ai_provider}
                if isinstance(self.ai_provider, AIProviderContract)
                else {}
            )
            self.provider_selector = AIProviderSelector(providers)
        if self.internet_service_selector is None:
            self.internet_service_selector = InternetServiceSelector()

    def _audit_tool_name(self, proposal: ToolProposal) -> str | None:
        if type(proposal.tool_name) is str and self.registry.has(proposal.tool_name):
            return proposal.tool_name
        return None

    def _log_audit(
        self,
        *,
        session_id: str,
        action_ref: str,
        outcome: str,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return self.audit_logger.log(
                session_id=session_id,
                action_ref=action_ref,
                outcome=outcome,
                details=details,
            )
        except Exception:
            raise AuditLoggingError("Audit event could not be recorded.") from None

    def process(self, user_text: str) -> ConversationResult:
        if type(user_text) is not str:
            raise InvalidInputError("User input must be text.")
        if len(user_text) > MAX_USER_INPUT_LENGTH:
            raise InvalidInputError(
                f"User input cannot exceed {MAX_USER_INPUT_LENGTH} characters."
            )
        if not user_text.strip():
            raise InvalidInputError("User input cannot be empty.")

        request = RequestEnvelope(text=user_text)
        user_input = UserInput(text=user_text)
        context = ConversationContext.create(user_text=user_input.text)
        trace_id = context.trace_id

        decision = self.request_router.route(request)
        if (
            type(decision) is not RouteDecision
            or decision.request is not request
            or decision.source is not InterpretationSource.DETERMINISTIC_ROUTER
        ):
            return ConversationResult(
                user_text=user_input.text,
                intent="unknown",
                proposal=None,
                authorized=False,
                success=False,
                error="Request was denied.",
            )

        if self.provider_selector is None:
            raise RuntimeError("The provider selector was not initialized.")
        selection = self.provider_selector.select(
            decision,
            self.registry,
            self.settings,
        )

        if decision.status is InterpretationStatus.NEEDS_CLARIFICATION:
            if decision.capability is Capability.WEB_SEARCH:
                clarification = (
                    "Please provide search text after 'web:' or "
                    "'busca en Internet:'."
                )
            elif decision.capability in {
                Capability.ANSWER_QUESTION,
                Capability.SUMMARIZE_TEXT,
            }:
                clarification = (
                    "Please provide text after 'pregunta:' or 'resumen:'."
                )
            elif decision.capability is Capability.ECHO_MESSAGE:
                clarification = "Please provide a message after 'repite:' or 'echo:'."
            else:
                clarification = (
                    "Use 'repite: <message>' or 'echo: <message>' "
                    "for a supported request."
                )
            return ConversationResult(
                user_text=user_input.text,
                intent="unknown",
                proposal=None,
                authorized=False,
                success=False,
                error=clarification,
            )
        if decision.category is RouteCategory.EXTERNAL_SERVICE:
            if (
                decision.capability is not Capability.WEB_SEARCH
                or set(decision.parameters) != {"query"}
                or type(decision.parameters["query"]) is not str
                or not decision.parameters["query"].strip()
            ):
                return ConversationResult(
                    user_text=user_input.text,
                    intent="unknown",
                    proposal=None,
                    authorized=False,
                    success=False,
                    error="Request was denied.",
                )
            if self.internet_service_selector is None:
                raise RuntimeError("The Internet service selector was not initialized.")
            try:
                service_request = InternetServiceRequest(
                    capability=InternetServiceCapability.WEB_SEARCH,
                    query=decision.parameters["query"],
                )
            except (TypeError, ValueError):
                return ConversationResult(
                    user_text=user_input.text,
                    intent="unknown",
                    proposal=None,
                    authorized=False,
                    success=False,
                    error="Request was denied.",
                )
            try:
                service_selection = self.internet_service_selector.select(
                    service_request
                )
            except Exception:
                return ConversationResult(
                    user_text=user_input.text,
                    intent="unknown",
                    proposal=None,
                    authorized=False,
                    success=False,
                    error="Internet search is unavailable.",
                )
            if (
                type(service_selection) is not InternetServiceSelection
                or service_selection.reason
                is InternetServiceUnavailableReason.POLICY_DENIED
            ):
                return ConversationResult(
                    user_text=user_input.text,
                    intent="unknown",
                    proposal=None,
                    authorized=False,
                    success=False,
                    error="Internet search is unavailable.",
                )
            if (
                service_selection.reason
                is not InternetServiceUnavailableReason.ADAPTER_UNAVAILABLE
            ):
                return ConversationResult(
                    user_text=user_input.text,
                    intent="unknown",
                    proposal=None,
                    authorized=False,
                    success=False,
                    error="Internet search is unavailable.",
                )
            return ConversationResult(
                user_text=user_input.text,
                intent="unknown",
                proposal=None,
                authorized=False,
                success=False,
                error="Internet search is unavailable.",
            )
        if decision.category is RouteCategory.GENERATIVE:
            if (
                decision.status is not InterpretationStatus.UNDERSTOOD
                or decision.capability
                not in {Capability.ANSWER_QUESTION, Capability.SUMMARIZE_TEXT}
                or set(decision.parameters) != {"text"}
                or type(decision.parameters["text"]) is not str
                or not decision.parameters["text"].strip()
            ):
                return ConversationResult(
                    user_text=user_input.text,
                    intent="unknown",
                    proposal=None,
                    authorized=False,
                    success=False,
                    error="Request was denied.",
                )
            if selection.target is not SelectionTarget.AI_PROVIDER:
                return ConversationResult(
                    user_text=user_input.text,
                    intent="text_request",
                    proposal=None,
                    authorized=False,
                    success=False,
                    error="Informative processing is unavailable.",
                )
            provider = selection.provider
            if (
                selection.provider_name == "simulated"
                and self.ai_provider is not None
            ):
                provider = self.ai_provider
            if provider is None:
                return ConversationResult(
                    user_text=user_input.text,
                    intent="text_request",
                    proposal=None,
                    authorized=False,
                    success=False,
                    error="Informative processing is unavailable.",
                )
            try:
                provider_method = getattr(
                    provider,
                    "respond_informational",
                    None,
                )
            except Exception:
                return ConversationResult(
                    user_text=user_input.text,
                    intent="text_request",
                    proposal=None,
                    authorized=False,
                    success=False,
                    error="Informative response could not be generated.",
                )
            if not callable(provider_method):
                return ConversationResult(
                    user_text=user_input.text,
                    intent="text_request",
                    proposal=None,
                    authorized=False,
                    success=False,
                    error="Informative processing is unavailable.",
                )
            try:
                informational_request = InformationalRequest(
                    capability=decision.capability.value,
                    content=decision.parameters["text"],
                )
                provider_response = provider_method(informational_request)
                if (
                    type(provider_response) is not InformationalResponse
                    or type(provider_response.text) is not str
                    or not provider_response.text.strip()
                    or len(provider_response.text)
                    > MAX_INFORMATIONAL_RESPONSE_LENGTH
                    or any(
                        ord(character) < 32 and character not in "\t\n\r"
                        for character in provider_response.text
                    )
                    or "\x7f" in provider_response.text
                ):
                    raise ValueError("Provider response does not match the contract.")
                validated_response = InformationalResponse(
                    text=provider_response.text
                )
            except Exception:
                return ConversationResult(
                    user_text=user_input.text,
                    intent="text_request",
                    proposal=None,
                    authorized=False,
                    success=False,
                    error="Informative response could not be generated.",
                )
            return ConversationResult(
                user_text=user_input.text,
                intent="text_request",
                proposal=None,
                authorized=False,
                success=True,
                informational_response=validated_response.text,
            )
        if (
            decision.category is RouteCategory.REJECT
            or decision.status is InterpretationStatus.UNSUPPORTED
        ):
            return ConversationResult(
                user_text=user_input.text,
                intent="unknown",
                proposal=None,
                authorized=False,
                success=False,
                error="Request was denied.",
            )
        if (
            decision.category is RouteCategory.SPECIALIZED
            and selection.target is not SelectionTarget.DETERMINISTIC_TOOL
        ):
            return ConversationResult(
                user_text=user_input.text,
                intent="unknown",
                proposal=None,
                authorized=False,
                success=False,
                error="Request was denied.",
            )
        if (
            decision.status is not InterpretationStatus.UNDERSTOOD
            or decision.category is not RouteCategory.SPECIALIZED
            or decision.capability is not Capability.ECHO_MESSAGE
            or "message" not in decision.parameters
            or type(decision.parameters["message"]) is not str
        ):
            return ConversationResult(
                user_text=user_input.text,
                intent="unknown",
                proposal=None,
                authorized=False,
                success=False,
                error="Request was denied.",
            )

        try:
            parameters = decision.mutable_parameters()
            self.registry.validate_tool_arguments("echo_message", parameters)
            trusted_definition = self.registry.get_trusted_definition("echo_message")
        except (AttributeError, TypeError, ValueError, ToolNotRegisteredError):
            audit_entry = self._log_audit(
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
                intent="unknown",
                proposal=None,
                authorized=False,
                success=False,
                error="Request was denied.",
                audit_id=audit_entry["audit_id"],
            )

        proposal = ToolProposal(
            tool_name="echo_message",
            parameters=parameters,
            permission=trusted_definition.required_permission,
            risk_level=trusted_definition.risk_level,
            requires_confirmation=trusted_definition.requires_confirmation,
            action_ref=f"echo:{trace_id}",
        )

        try:
            result = self.tool_runner.execute(proposal, context)
        except PreExecutionAuditError:
            raise
        except (AuthorizationError, SimulatedModeError, ToolNotRegisteredError):
            details = {
                "reason": "authorization_failed",
                "authorization": "denied",
                "execution_status": "blocked",
            }
            tool_name = self._audit_tool_name(proposal)
            if tool_name is not None:
                details["tool_name"] = tool_name
            audit_entry = self._log_audit(
                session_id=context.session_id,
                action_ref=proposal.action_ref,
                outcome="denied",
                details=details,
            )
            return ConversationResult(
                user_text=user_input.text,
                intent="text_request",
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
            try:
                audit_entry = self._log_audit(
                    session_id=context.session_id,
                    action_ref=proposal.action_ref,
                    outcome="error",
                    details=details,
                )
            except AuditLoggingError:
                raise PostExecutionAuditError() from None
            return ConversationResult(
                user_text=user_input.text,
                intent="text_request",
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
            audit_entry = self._log_audit(
                session_id=context.session_id,
                action_ref=proposal.action_ref,
                outcome="error",
                details=details,
            )
            return ConversationResult(
                user_text=user_input.text,
                intent="unknown",
                proposal=proposal,
                authorized=False,
                success=False,
                error="Request could not be processed.",
                audit_id=audit_entry["audit_id"],
            )

        try:
            audit_entry = self._log_audit(
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
        except AuditLoggingError:
            raise PostExecutionAuditError() from None
        return ConversationResult(
            user_text=user_input.text,
            intent="text_request",
            proposal=proposal,
            authorized=True,
            success=True,
            tool_result=result,
            audit_id=audit_entry["audit_id"],
        )
