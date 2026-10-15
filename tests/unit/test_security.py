import pytest

from jarvis.ai.simulator import SimulatedAIProvider
from jarvis.conversation.models import ToolProposal
from jarvis.core.context import ConversationContext
from jarvis.core.errors import ToolNotRegisteredError
from jarvis.security.audit import AuditLogger
from jarvis.security.authorization import AuthorizationService
from jarvis.security.policy import SecurityPolicy
from jarvis.tools.registry import ToolRegistry, create_default_registry


def test_unknown_permission_is_denied() -> None:
    registry = create_default_registry()
    auth = AuthorizationService(policy=SecurityPolicy(), registry=registry)
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.unknown",
        parameters={"message": "hello"},
    )
    assert auth.authorize(proposal, context) is False


def test_unregistered_tool_is_rejected() -> None:
    registry = ToolRegistry()
    auth = AuthorizationService(policy=SecurityPolicy(), registry=registry)
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="missing_tool",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )
    with pytest.raises(ToolNotRegisteredError):
        auth.authorize(proposal, context)


def test_missing_authorization_is_denied() -> None:
    registry = create_default_registry()
    policy = SecurityPolicy()
    auth = AuthorizationService(policy=policy, registry=registry)
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )
    assert auth.authorize(proposal, context) is False


def test_context_permission_claim_is_not_authorization() -> None:
    registry = create_default_registry()
    policy = SecurityPolicy()
    auth = AuthorizationService(policy=policy, registry=registry)
    context = ConversationContext.create("hello", claims={"permissions": {"tool.echo_message": True}})
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )
    assert auth.authorize(proposal, context) is False


def test_context_cannot_grant_permissions() -> None:
    context = ConversationContext.create("hello", claims={"permissions": {"tool.echo_message": True}})
    assert context.claims.get("permissions") == {"tool.echo_message": True}
    assert context.claims != {"tool.echo_message": True}


def test_policy_permissions_are_read_only() -> None:
    policy = SecurityPolicy(allowed_permissions={"tool.echo_message"})
    assert policy.allowed_permissions == frozenset({"tool.echo_message"})
    with pytest.raises(AttributeError):
        policy.allowed_permissions = {"tool.other"}
    with pytest.raises(AttributeError):
        policy.allowed_permissions.add("tool.other")


def test_invalid_permission_names_are_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        SecurityPolicy(allowed_permissions={123})
    with pytest.raises(ValueError):
        SecurityPolicy(allowed_permissions={" "})


def test_simulator_cannot_grant_permissions() -> None:
    provider = SimulatedAIProvider()
    assert not hasattr(provider, "grant_permissions")
    assert callable(getattr(provider, "classify_intent", None))


def test_audit_logs_without_sensitive_data() -> None:
    logger = AuditLogger()
    entry = logger.log(
        session_id="session-1",
        action_ref="action-1",
        outcome="denied",
        details={"password": "secret-value", "message": "hello"},
    )
    assert entry["details"]["password"] == "[REDACTED]"
    assert entry["details"]["message"] == "hello"
