import pytest

from jarvis.conversation.models import ToolProposal
from jarvis.core.context import ConversationContext
from jarvis.core.errors import AuthorizationError, SimulatedModeError, ToolNotRegisteredError
from jarvis.security.audit import AuditLogger
from jarvis.security.authorization import AuthorizationService
from jarvis.security.policy import SecurityPolicy
from jarvis.tools.registry import ToolRegistry, create_default_registry
from jarvis.tools.runner import SimulatedToolRunner


def test_authorized_simulated_tool_executes() -> None:
    registry = create_default_registry()
    policy = SecurityPolicy(allowed_permissions={"tool.echo_message"})
    auth = AuthorizationService(policy=policy, registry=registry)
    runner = SimulatedToolRunner(registry=registry, authorization_service=auth, mode="simulated")
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "hello"},
        action_ref="echo:abc",
    )
    result = runner.execute(proposal, context)
    assert result["status"] == "simulated_ok"


def test_unregistered_tool_is_denied_by_runner() -> None:
    registry = create_default_registry()
    auth = AuthorizationService(policy=SecurityPolicy(), registry=registry)
    runner = SimulatedToolRunner(registry=registry, authorization_service=auth, mode="simulated")
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="missing_tool",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )
    with pytest.raises(ToolNotRegisteredError):
        runner.execute(proposal, context)


def test_real_execution_route_is_blocked() -> None:
    registry = create_default_registry()
    tool = registry.get("echo_message")
    object.__setattr__(tool, "execution_mode", "real")
    policy = SecurityPolicy(allowed_permissions={"tool.echo_message"})
    auth = AuthorizationService(policy=policy, registry=registry)
    runner = SimulatedToolRunner(registry=registry, authorization_service=auth, mode="simulated")
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )
    with pytest.raises(SimulatedModeError):
        runner.execute(proposal, context)


def test_authorization_error_stops_execution() -> None:
    registry = create_default_registry()
    auth = AuthorizationService(policy=SecurityPolicy(), registry=registry)
    runner = SimulatedToolRunner(registry=registry, authorization_service=auth, mode="simulated")
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )
    tool = registry.get("echo_message")
    original_execute = tool.execute
    calls = {"count": 0}

    def fail_if_called(*args, **kwargs):
        calls["count"] += 1
        return original_execute(*args, **kwargs)

    tool.execute = fail_if_called
    with pytest.raises(AuthorizationError):
        runner.execute(proposal, context)
    assert calls["count"] == 0


def test_authorization_exception_prevents_execution() -> None:
    registry = create_default_registry()
    policy = SecurityPolicy(allowed_permissions={"tool.echo_message"})
    auth = AuthorizationService(policy=policy, registry=registry)
    runner = SimulatedToolRunner(registry=registry, authorization_service=auth, mode="simulated")
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )

    def boom(*args, **kwargs):
        raise AuthorizationError("authorization exploded")

    auth.authorize = boom  # type: ignore[assignment]
    with pytest.raises(AuthorizationError, match="authorization exploded"):
        runner.execute(proposal, context)


def test_audit_logger_stores_redacted_metadata() -> None:
    logger = AuditLogger()
    entry = logger.log(
        session_id="session-1",
        action_ref="action-1",
        outcome="allowed",
        details={"token": "abc123", "status": "ok"},
    )
    assert entry["details"]["token"] == "[REDACTED]"
    assert entry["details"]["status"] == "ok"


def test_registry_rejects_invalid_tool_objects() -> None:
    registry = ToolRegistry()
    with pytest.raises(TypeError):
        registry.register(object())


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry()
    registry.register(create_default_registry().get("echo_message"))
    with pytest.raises(ValueError):
        registry.register(create_default_registry().get("echo_message"))
