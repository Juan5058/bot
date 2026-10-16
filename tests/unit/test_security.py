import pytest

from jarvis.ai.simulator import SimulatedAIProvider
from jarvis.app import create_application
from jarvis.config.settings import Settings
from jarvis.conversation.models import ConversationResult, ToolProposal
from jarvis.core.context import ConversationContext
from jarvis.core.errors import InvalidInputError, ToolNotRegisteredError
from jarvis.security.audit import AuditLogger
from jarvis.security.authorization import AuthorizationService
from jarvis.security.policy import SecurityPolicy
from jarvis.tools.registry import EchoTool, ToolRegistry, create_default_registry
from jarvis.tools.runner import SimulatedToolRunner


def _use_policy_for_test(app, policy: SecurityPolicy) -> None:
    authorizer = AuthorizationService(policy=policy, registry=app.registry)
    app.authorizer = authorizer
    app.tool_runner = SimulatedToolRunner(
        registry=app.registry,
        authorization_service=authorizer,
    )


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
    registry.freeze()
    auth = AuthorizationService(policy=SecurityPolicy(), registry=registry)
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="missing_tool",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )
    with pytest.raises(ToolNotRegisteredError):
        auth.authorize(proposal, context)


def test_proposal_with_malformed_or_manipulated_fields_is_denied() -> None:
    registry = create_default_registry()
    auth = AuthorizationService(
        policy=SecurityPolicy(allowed_permissions={"tool.echo_message"}),
        registry=registry,
    )
    context = ConversationContext.create("hello")

    malformed = (
        ToolProposal(tool_name="echo_message", permission=123, parameters={"message": "hello"}),
        ToolProposal(tool_name="echo_message", permission="tool.echo_message", risk_level="critical"),
        ToolProposal(
            tool_name="echo_message",
            permission="tool.echo_message",
            requires_confirmation="false",
        ),
        ToolProposal(tool_name="echo_message", permission="tool.echo_message", parameters=[]),
    )
    assert all(auth.authorize(proposal, context) is False for proposal in malformed)


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
    policy = SecurityPolicy(
        allowed_permissions={"tool.echo_message"},
        allowed_tool_permissions={("echo_message", "tool.echo_message")},
    )
    assert policy.allowed_permissions == frozenset({"tool.echo_message"})
    assert policy.allowed_tool_permissions == frozenset({("echo_message", "tool.echo_message")})
    with pytest.raises(AttributeError):
        policy.allowed_permissions = {"tool.other"}
    with pytest.raises(AttributeError):
        policy.allowed_permissions.add("tool.other")
    with pytest.raises(AttributeError):
        policy.allowed_tool_permissions = {("other_echo", "tool.echo_message")}


def test_policy_security_fields_are_immutable() -> None:
    policy = SecurityPolicy(allowed_permissions={"tool.echo_message"})
    for attribute, value in (
        ("default_deny", False),
        ("mode", "real"),
        ("identity_required", False),
    ):
        with pytest.raises(AttributeError):
            setattr(policy, attribute, value)
        with pytest.raises(AttributeError):
            delattr(policy, attribute)


def test_policy_instances_are_isolated() -> None:
    policy_a = SecurityPolicy(allowed_permissions={"tool.echo_message"})
    policy_b = SecurityPolicy(allowed_permissions={"tool.other"})
    assert policy_a.allowed_permissions == frozenset({"tool.echo_message"})
    assert policy_b.allowed_permissions == frozenset({"tool.other"})
    assert policy_a.allowed_permissions is not policy_b.allowed_permissions


def test_identity_required_denies_context_and_unverified_claims() -> None:
    registry = create_default_registry()
    auth = AuthorizationService(
        policy=SecurityPolicy(
            identity_required=True,
            allowed_permissions={"tool.echo_message"},
        ),
        registry=registry,
    )
    context = ConversationContext.create(
        "hello",
        claims={"identity_verified": True, "user_id": "claimed-user"},
    )
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )

    assert auth.authorize(proposal, context) is False


def test_authorization_rejects_invalid_session_and_mode() -> None:
    registry = create_default_registry()
    auth = AuthorizationService(
        policy=SecurityPolicy(allowed_permissions={"tool.echo_message"}),
        registry=registry,
    )
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )

    invalid_session = ConversationContext.create("hello")
    invalid_session.session_id = "model-supplied-session"
    assert auth.authorize(proposal, invalid_session) is False

    invalid_mode = ConversationContext.create("hello")
    invalid_mode.mode = "real"
    assert auth.authorize(proposal, invalid_mode) is False


def test_configuration_rejects_malformed_permission_names() -> None:
    with pytest.raises(ValueError):
        SecurityPolicy(allowed_permissions={"tool.bad permission"})
    with pytest.raises(TypeError):
        SecurityPolicy(allowed_permissions="tool.echo_message")
    with pytest.raises(ValueError):
        SecurityPolicy(allowed_permissions={"tool..echo"})


def test_policy_permission_is_bound_to_the_exact_tool_name() -> None:
    policy = SecurityPolicy.for_application()

    assert policy.can_execute_tool("echo_message", "tool.echo_message") is True
    assert policy.can_execute_tool("different_tool", "tool.echo_message") is False


def test_policy_rejects_tool_permission_pair_not_in_permission_allowlist() -> None:
    with pytest.raises(ValueError, match="must also be allowed"):
        SecurityPolicy(
            allowed_permissions={"tool.echo_message"},
            allowed_tool_permissions={("different_tool", "tool.different_tool")},
        )


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
    unique_message = "user-provided-unique-marker-xyz-42"
    entry = logger.log(
        session_id="session-12345678",
        action_ref="echo:0123456789ab",
        outcome="denied",
        details={
            "reason": "authorization_failed",
            "tool_name": "echo_message",
            "authorization": "denied",
            "execution_status": "blocked",
            "password": unique_message,
            "user_text": unique_message,
            "nested": {"message": unique_message, "items": [unique_message]},
            "result": {"text": unique_message},
            "unknown_field": unique_message,
        },
    )
    assert entry["details"] == {
        "reason": "authorization_failed",
        "authorization": "denied",
        "execution_status": "blocked",
        "tool_name": "echo_message",
    }
    assert unique_message not in str(entry)


def test_audit_log_returns_defensive_copies() -> None:
    logger = AuditLogger()
    returned = logger.log(
        session_id="session-12345678",
        action_ref="echo:0123456789ab",
        outcome="allowed",
        details={
            "authorization": "granted",
            "execution_status": "completed",
            "tool_name": "echo_message",
        },
    )
    returned["details"]["authorization"] = "denied"
    returned["session_id"] = "session-00000000"

    first_snapshot = logger.entries
    assert first_snapshot[0]["details"]["authorization"] == "granted"
    assert first_snapshot[0]["session_id"] == "session-12345678"
    first_snapshot[0]["details"]["authorization"] = "denied"
    first_snapshot.clear()

    assert logger.entries[0]["details"]["authorization"] == "granted"


def test_audit_rejects_invalid_values_and_bounds_identifiers() -> None:
    logger = AuditLogger()
    marker = "invalid-audit-value-marker"
    entry = logger.log(
        session_id=f"session-{marker}",
        action_ref="a" * 100,
        outcome=marker,
        details={
            "tool_name": {"nested": marker},
            "authorization": marker,
            "execution_status": ["completed", marker],
            "error_code": marker,
            "result_code": marker,
            "reason": "authorization_failed",
        },
    )
    assert entry["session_id"] == "invalid-session"
    assert entry["action_ref"] == "invalid-action"
    assert entry["outcome"] == "error"
    assert entry["details"] == {
        "reason": "authorization_failed",
        "identifier_status": "invalid_both",
    }
    assert marker not in str(entry)


def test_audit_accepts_generated_action_reference_and_marks_invalid_identifiers() -> None:
    logger = AuditLogger()
    context = ConversationContext.create("hello")

    entry = logger.log(
        session_id=context.session_id,
        action_ref=f"echo:{context.trace_id}",
        outcome="allowed",
        details={"authorization": "granted"},
    )
    assert entry["session_id"] == context.session_id
    assert entry["action_ref"] == f"echo:{context.trace_id}"
    assert "identifier_status" not in entry["details"]


def test_conversation_result_serializes_slotted_proposal_without_secrets() -> None:
    secret = "private-token-value"
    proposal = ToolProposal(
        tool_name="echo_message",
        parameters={
            "message": "hello",
            "credentials": {"api_key": secret, "label": "test"},
        },
        permission="tool.echo_message",
        confirmation_token=secret,
        action_ref="echo:0123456789ab",
    )
    result = ConversationResult(
        user_text="hello",
        intent="text_request",
        proposal=proposal,
        authorized=False,
        success=False,
    )

    serialized = result.to_dict()

    assert serialized["proposal"] == {
        "tool_name": "echo_message",
        "parameters": {
            "message": "hello",
            "credentials": "[REDACTED]",
        },
        "permission": "tool.echo_message",
        "risk_level": "low",
        "requires_confirmation": False,
        "action_ref": "echo:0123456789ab",
    }
    assert secret not in str(serialized)
    assert secret not in repr(proposal)


def test_conversation_result_serializes_missing_proposal() -> None:
    result = ConversationResult(
        user_text="hello",
        intent="unknown",
        proposal=None,
        authorized=False,
        success=False,
    )
    assert result.to_dict()["proposal"] is None


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("tool_name", object(), TypeError),
        ("parameters", {"unsupported": object()}, TypeError),
        ("parameters", {1: "invalid-key"}, TypeError),
        ("parameters", {"number": float("inf")}, ValueError),
    ),
)
def test_conversation_result_rejects_invalid_proposal_values(field, value, error) -> None:
    proposal = ToolProposal(tool_name="echo_message")
    setattr(proposal, field, value)
    result = ConversationResult(
        user_text="hello",
        intent="text_request",
        proposal=proposal,
        authorized=False,
        success=False,
    )
    with pytest.raises(error):
        result.to_dict()


def test_conversation_result_parameter_depth_limit() -> None:
    within_limit = "value"
    for _ in range(31):
        within_limit = [within_limit]

    accepted = ConversationResult(
        user_text="hello",
        intent="text_request",
        proposal=ToolProposal(
            tool_name="echo_message",
            parameters={"nested": within_limit},
        ),
        authorized=False,
        success=False,
    )
    assert accepted.to_dict()["proposal"]["parameters"]["nested"] == within_limit

    exceeds_limit = "value"
    for _ in range(32):
        exceeds_limit = [exceeds_limit]

    rejected = ConversationResult(
        user_text="hello",
        intent="text_request",
        proposal=ToolProposal(
            tool_name="echo_message",
            parameters={"nested": exceeds_limit},
        ),
        authorized=False,
        success=False,
    )
    with pytest.raises(ValueError, match="nesting depth"):
        rejected.to_dict()


def test_conversation_result_parameter_container_item_limit() -> None:
    accepted_list = ConversationResult(
        user_text="hello",
        intent="text_request",
        proposal=ToolProposal(
            tool_name="echo_message",
            parameters={"items": list(range(256))},
        ),
        authorized=False,
        success=False,
    )
    assert len(accepted_list.to_dict()["proposal"]["parameters"]["items"]) == 256

    rejected_list = ConversationResult(
        user_text="hello",
        intent="text_request",
        proposal=ToolProposal(
            tool_name="echo_message",
            parameters={"items": list(range(257))},
        ),
        authorized=False,
        success=False,
    )
    with pytest.raises(ValueError, match="containers"):
        rejected_list.to_dict()

    accepted_dict = ConversationResult(
        user_text="hello",
        intent="text_request",
        proposal=ToolProposal(
            tool_name="echo_message",
            parameters={f"item_{index}": index for index in range(256)},
        ),
        authorized=False,
        success=False,
    )
    assert len(accepted_dict.to_dict()["proposal"]["parameters"]) == 256

    rejected_dict = ConversationResult(
        user_text="hello",
        intent="text_request",
        proposal=ToolProposal(
            tool_name="echo_message",
            parameters={f"item_{index}": index for index in range(257)},
        ),
        authorized=False,
        success=False,
    )
    with pytest.raises(ValueError, match="containers"):
        rejected_dict.to_dict()


def test_conversation_result_parameter_global_node_limit() -> None:
    parameters = {
        f"group_{group_index}": list(range(256))
        for group_index in range(8)
    }
    result = ConversationResult(
        user_text="hello",
        intent="text_request",
        proposal=ToolProposal(
            tool_name="echo_message",
            parameters=parameters,
        ),
        authorized=False,
        success=False,
    )

    with pytest.raises(ValueError, match="supported size"):
        result.to_dict()


def test_conversation_result_rejects_cyclic_parameters_without_recursion_error() -> None:
    cyclic_list = []
    cyclic_list.append(cyclic_list)
    result = ConversationResult(
        user_text="hello",
        intent="text_request",
        proposal=ToolProposal(
            tool_name="echo_message",
            parameters={"cycle": cyclic_list},
        ),
        authorized=False,
        success=False,
    )

    with pytest.raises(ValueError, match="nesting depth"):
        result.to_dict()


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_conversation_result_rejects_non_finite_parameter_numbers(value) -> None:
    result = ConversationResult(
        user_text="hello",
        intent="text_request",
        proposal=ToolProposal(
            tool_name="echo_message",
            parameters={"number": value},
        ),
        authorized=False,
        success=False,
    )

    with pytest.raises(ValueError, match="non-finite numbers"):
        result.to_dict()


def test_conversation_result_redacts_sensitive_parameter_keys() -> None:
    secrets = {
        "password": "pw-marker-91",
        "api_key": "key-marker-82",
        "access_token": "token-marker-73",
        "credential": "credential-marker-64",
        "biometric": "bio-marker-55",
        "PaSsWoRdHint": "mixed-case-marker-46",
    }
    result = ConversationResult(
        user_text="hello",
        intent="text_request",
        proposal=ToolProposal(
            tool_name="echo_message",
            parameters=secrets,
        ),
        authorized=False,
        success=False,
    )

    serialized = result.to_dict()["proposal"]["parameters"]

    assert serialized == {key: "[REDACTED]" for key in secrets}
    assert all(secret not in str(serialized) for secret in secrets.values())


def test_pipeline_audit_keeps_only_allowlisted_metadata() -> None:
    marker = "pipeline-result-private-marker"
    app = create_application(Settings(mode="simulated", ai_enabled=False))
    result = app.process(marker)

    assert result.success is True
    assert marker not in str(app.audit_logger.entries)
    assert app.audit_logger.entries[0]["outcome"] == "allowed"
    assert app.audit_logger.entries[0]["details"] == {
        "authorization": "granted",
        "execution_status": "completed",
        "result_code": "simulated_ok",
        "tool_name": "echo_message",
    }


def test_pipeline_does_not_audit_exception_messages(monkeypatch) -> None:
    marker = "exception-private-marker"
    app = create_application(Settings(mode="simulated", ai_enabled=False))

    def fail_with_sensitive_message(self, *args, **kwargs):
        raise RuntimeError(marker)

    monkeypatch.setattr(EchoTool, "execute", fail_with_sensitive_message)
    result = app.process("hello")

    assert result.success is False
    assert result.error == "Tool execution failed."
    assert marker not in str(app.audit_logger.entries)
    assert app.audit_logger.entries[0]["outcome"] == "error"
    assert app.audit_logger.entries[0]["details"] == {
        "authorization": "granted",
        "execution_status": "error",
        "error_code": "tool_execution_failed",
        "tool_name": "echo_message",
    }


def test_pipeline_denial_audit_preserves_operational_metadata(monkeypatch) -> None:
    app = create_application(Settings(mode="simulated", ai_enabled=False))
    _use_policy_for_test(app, SecurityPolicy())
    calls = {"count": 0}

    def count_if_called(self, context, **kwargs):
        calls["count"] += 1
        return {"status": "unexpected"}

    monkeypatch.setattr(EchoTool, "execute", count_if_called)

    result = app.process("hello")

    assert result.authorized is False
    assert result.success is False
    assert calls["count"] == 0
    assert app.audit_logger.entries[0]["outcome"] == "denied"
    assert app.audit_logger.entries[0]["details"] == {
        "reason": "authorization_failed",
        "authorization": "denied",
        "execution_status": "blocked",
        "tool_name": "echo_message",
    }


def test_audit_failure_does_not_allow_a_denied_tool(monkeypatch) -> None:
    app = create_application(Settings(mode="simulated", ai_enabled=False))
    _use_policy_for_test(app, SecurityPolicy())
    calls = {"count": 0}

    def count_if_called(self, context, **kwargs):
        calls["count"] += 1
        return {"status": "unexpected"}

    def fail_audit(**kwargs):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(EchoTool, "execute", count_if_called)
    monkeypatch.setattr(app.audit_logger, "log", fail_audit)

    with pytest.raises(RuntimeError, match="audit unavailable"):
        app.process("hello")
    assert calls["count"] == 0


def test_ai_provider_context_mutation_is_denied_and_restored(monkeypatch) -> None:
    app = create_application(Settings(mode="simulated", ai_enabled=False))
    simulator = SimulatedAIProvider()
    calls = {"count": 0}
    contexts = []

    def counted_echo(self, context, **kwargs):
        calls["count"] += 1
        return {"status": "unexpected"}

    class MutatingProvider:
        def classify_intent(self, user_text):
            return simulator.classify_intent(user_text)

        def propose_action(self, user_text, context):
            contexts.append(context)
            context.mode = "real"
            context.session_id = "attacker-session"
            context.confirmed_actions.add("forged-action")
            context.claims["identity_verified"] = True
            return simulator.propose_action(user_text, context)

        def respond(self, user_text, context):
            return simulator.respond(user_text, context)

    monkeypatch.setattr(EchoTool, "execute", counted_echo)
    app.ai_provider = MutatingProvider()
    result = app.process("hello")

    assert result.success is False
    assert result.authorized is False
    assert contexts[0].mode == "simulated"
    assert contexts[0].session_id == app.audit_logger.entries[0]["session_id"]
    assert contexts[0].claims == {}
    assert contexts[0].confirmed_actions == set()
    assert calls["count"] == 0
    assert app.audit_logger.entries[0]["action_ref"].startswith("echo:")
    assert app.audit_logger.entries[0]["details"]["reason"] == "invalid_proposal"


def test_pipeline_uses_one_context_for_proposal_and_execution(monkeypatch) -> None:
    app = create_application(Settings(mode="simulated", ai_enabled=False))
    simulated_provider = SimulatedAIProvider()
    observed = {}

    class TrackingProvider:
        def classify_intent(self, user_text):
            observed["classified_text"] = user_text
            return simulated_provider.classify_intent(user_text)

        def propose_action(self, user_text, context):
            observed["proposal_context"] = context
            return simulated_provider.propose_action(user_text, context)

        def respond(self, user_text, context):
            return simulated_provider.respond(user_text, context)

    app.ai_provider = TrackingProvider()
    execute = app.tool_runner.execute

    def track_execute(proposal, context):
        observed["execution_context"] = context
        return execute(proposal, context)

    monkeypatch.setattr(app.tool_runner, "execute", track_execute)
    result = app.process("hello")

    context = observed["proposal_context"]
    assert result.success is True
    assert observed["classified_text"] == context.user_text == "hello"
    assert observed["execution_context"] is context
    assert app.audit_logger.entries[0]["session_id"] == context.session_id
    assert app.audit_logger.entries[0]["action_ref"] == f"echo:{context.trace_id}"


def test_malformed_provider_proposal_is_denied_without_leaking_data() -> None:
    marker = "malformed-proposal-private-marker"
    app = create_application(Settings(mode="simulated", ai_enabled=False))

    class MalformedProvider:
        def classify_intent(self, user_text):
            return {"intent": "text_request"}

        def propose_action(self, user_text, context):
            return {"tool_name": marker, "permission": "tool.echo_message"}

        def respond(self, user_text, context):
            return marker

    app.ai_provider = MalformedProvider()
    result = app.process("hello")

    assert result.success is False
    assert result.authorized is False
    assert result.error == "Request was denied."
    assert marker not in str(app.audit_logger.entries)
    assert app.audit_logger.entries[0]["outcome"] == "denied"


def test_pipeline_rejects_invalid_user_input_before_provider_invocation() -> None:
    app = create_application(Settings(mode="simulated", ai_enabled=False))
    calls = {"classify": 0, "propose": 0}

    class CountingProvider:
        def classify_intent(self, user_text):
            calls["classify"] += 1
            return {"intent": "text_request"}

        def propose_action(self, user_text, context):
            calls["propose"] += 1
            return SimulatedAIProvider().propose_action(user_text, context)

        def respond(self, user_text, context):
            return user_text

    app.ai_provider = CountingProvider()

    invalid_inputs = (None, 42, [], {}, "", " \t\r\n")
    for user_text in invalid_inputs:
        with pytest.raises(InvalidInputError):
            app.process(user_text)

    with pytest.raises(InvalidInputError, match="8000 characters"):
        app.process("x" * 8001)

    assert calls == {"classify": 0, "propose": 0}


def test_pipeline_accepts_user_input_at_the_maximum_length() -> None:
    app = create_application(Settings(mode="simulated", ai_enabled=False))

    result = app.process("x" * 8000)

    assert result.success is True
    assert result.tool_result is not None
    assert len(result.tool_result["result"]["message"]) == 8000


@pytest.mark.parametrize(
    "classification",
    (
        None,
        [],
        {},
        {"intent": None},
        {"intent": 1},
        {"intent": "not_a_supported_intent"},
        {"intent": "unknown"},
    ),
)
def test_pipeline_denies_invalid_classification_before_proposal(
    monkeypatch,
    classification,
) -> None:
    app = create_application(Settings(mode="simulated", ai_enabled=False))
    calls = {"propose": 0, "execute": 0}

    class ClassifyingProvider:
        def classify_intent(self, user_text):
            return classification

        def propose_action(self, user_text, context):
            calls["propose"] += 1
            return SimulatedAIProvider().propose_action(user_text, context)

        def respond(self, user_text, context):
            return user_text

    def count_execution(self, context, **kwargs):
        calls["execute"] += 1
        return {"status": "unexpected"}

    monkeypatch.setattr(EchoTool, "execute", count_execution)
    app.ai_provider = ClassifyingProvider()

    result = app.process("hello")

    assert result.success is False
    assert result.authorized is False
    assert result.intent == "unknown"
    assert calls == {"propose": 0, "execute": 0}
    assert app.audit_logger.entries[0]["details"]["reason"] == "invalid_classification"


def test_pipeline_denies_classification_with_non_string_key(monkeypatch) -> None:
    app = create_application(Settings(mode="simulated", ai_enabled=False))
    calls = {"propose": 0}

    class MalformedClassificationProvider:
        def classify_intent(self, user_text):
            return {"intent": "text_request", 1: "unexpected"}

        def propose_action(self, user_text, context):
            calls["propose"] += 1
            return SimulatedAIProvider().propose_action(user_text, context)

        def respond(self, user_text, context):
            return user_text

    app.ai_provider = MalformedClassificationProvider()
    result = app.process("hello")

    assert result.success is False
    assert result.authorized is False
    assert calls["propose"] == 0
    assert app.audit_logger.entries[0]["details"]["reason"] == "invalid_classification"


def test_pipeline_accepts_supported_text_request_classification() -> None:
    app = create_application(Settings(mode="simulated", ai_enabled=False))

    result = app.process("hello")

    assert result.intent == "text_request"
    assert result.success is True
