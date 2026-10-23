from typing import Any

import pytest
import jarvis.tools.runner as runner_module
from jarvis.app import create_application
from jarvis.conversation.models import ToolProposal
from jarvis.core.context import ConversationContext
from jarvis.core.errors import AuthorizationError, SimulatedModeError, ToolNotRegisteredError
from jarvis.config.settings import Settings
from jarvis.security.audit import AuditLogger
from jarvis.security.authorization import AuthorizationService
from jarvis.security.policy import SecurityPolicy
from jarvis.tools.base import ToolDefinition
from jarvis.tools.registry import (
    EchoTool,
    ToolArgumentDefinition,
    ToolRegistry,
    TrustedToolDefinition,
    create_default_registry,
)
from jarvis.tools.runner import SimulatedToolRunner, _SimulatedToolDispatcher


_TEST_EXECUTORS: dict[ToolRegistry, dict[str, Any]] = {}


class TestSimulatedToolRunner(_SimulatedToolDispatcher):
    __test__ = False
    __slots__ = ()

    def __init__(
        self,
        registry,
        authorization_service,
        *,
        mode,
        executors,
    ):
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(
            self, "_authorization_service", authorization_service
        )
        object.__setattr__(self, "_executors", dict(executors))
        object.__setattr__(self, "_audit_logger", AuditLogger())
        object.__setattr__(self, "mode", mode)


class CountingTool(ToolDefinition):
    __slots__ = ("calls",)

    def __init__(self, *, calls: dict[str, int]) -> None:
        super().__init__()
        self.calls = calls
        self.name = "counting_tool"
        self.description = "Test tool that counts calls."
        self.required_permission = "tool.counting_tool"
        self.risk_level = "low"
        self.requires_confirmation = False
        self.execution_mode = "simulated"

class ConfirmingTool(CountingTool):
    __slots__ = ()

    def __init__(self, *, calls: dict[str, int]) -> None:
        super().__init__(calls=calls)
        self.name = "confirming_tool"
        self.required_permission = "tool.confirming_tool"
        self.risk_level = "high"
        self.requires_confirmation = True


class AlternativeEchoTool(EchoTool):
    __slots__ = ("calls",)

    def __init__(self, *, calls: dict[str, int]) -> None:
        super().__init__()
        self.calls = calls

_COUNTING_TOOL_DEFINITION = TrustedToolDefinition(
    name="counting_tool",
    implementation_type=CountingTool,
    required_permission="tool.counting_tool",
    allow_additional_arguments=True,
)
_CONFIRMING_TOOL_DEFINITION = TrustedToolDefinition(
    name="confirming_tool",
    implementation_type=ConfirmingTool,
    required_permission="tool.confirming_tool",
    risk_level="high",
    requires_confirmation=True,
    allow_additional_arguments=True,
)


def _registry_for(
    definition: TrustedToolDefinition,
    tool: ToolDefinition,
) -> ToolRegistry:
    registry = ToolRegistry.for_trusted_composition({definition.name: definition})
    registry.register(tool)
    registry.freeze()

    def execute_counting_tool(context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        tool.calls["count"] += 1
        return {
            "kind": "counted",
            "message": kwargs.get("message", ""),
            "status": "ok",
        }

    _TEST_EXECUTORS[registry] = {definition.name: execute_counting_tool}
    return registry


def _alternative_echo_registry(calls: dict[str, int]) -> ToolRegistry:
    definition = TrustedToolDefinition(
        name="echo_message",
        implementation_type=AlternativeEchoTool,
        required_permission="tool.echo_message",
        argument_schema=(
            ToolArgumentDefinition(
                name="message",
                value_type=str,
                min_length=0,
                max_length=8000,
            ),
        ),
        risk_level="low",
        requires_confirmation=False,
        identity_required=False,
        execution_mode="simulated",
    )
    registry = ToolRegistry.for_trusted_composition({"echo_message": definition})
    tool = AlternativeEchoTool(calls=calls)
    registry.register(tool)
    registry.freeze()

    def execute_alternative_echo(
        context: ConversationContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        tool.calls["count"] += 1
        return {"kind": "alternative_echo", "status": "ok"}

    _TEST_EXECUTORS[registry] = {"echo_message": execute_alternative_echo}
    return registry


def _runner_for(
    registry: ToolRegistry,
    authorization_service: AuthorizationService,
    *,
    mode: str = "simulated",
) -> TestSimulatedToolRunner:
    executors = _TEST_EXECUTORS.pop(registry, None)
    if executors is None:
        executors = {"echo_message": runner_module._dispatch_echo_message}
    return TestSimulatedToolRunner(
        registry,
        authorization_service,
        mode=mode,
        executors=executors,
    )


def _policy_for(tool_name: str, permission: str) -> SecurityPolicy:
    return SecurityPolicy(
        allowed_permissions={permission},
        allowed_tool_permissions={(tool_name, permission)},
    )


def test_authorized_simulated_tool_executes() -> None:
    registry = create_default_registry()
    policy = _policy_for("echo_message", "tool.echo_message")
    auth = AuthorizationService(policy=policy, registry=registry)
    runner = _runner_for(registry, auth)
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "hello"},
        action_ref="echo:abc",
    )
    result = runner.execute(proposal, context)
    assert result["status"] == "simulated_ok"


def test_echo_schema_accepts_an_empty_message() -> None:
    registry = create_default_registry()
    authorization_service = AuthorizationService(
        _policy_for("echo_message", "tool.echo_message"),
        registry,
    )
    runner = _runner_for(registry, authorization_service)
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": ""},
    )

    result = runner.execute(proposal, ConversationContext.create("hello"))

    assert result["result"]["message"] == ""


def _invalid_parameter_values() -> list[tuple[str, Any]]:
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    too_deep: Any = "leaf"
    for _ in range(32):
        too_deep = [too_deep]
    too_many_nodes = {
        f"group_{index}": list(range(256))
        for index in range(8)
    }
    return [
        ("top-level-list", []),
        ("arbitrary-object", {"value": object()}),
        ("non-string-key", {1: "value"}),
        ("nan", {"value": float("nan")}),
        ("positive-infinity", {"value": float("inf")}),
        ("negative-infinity", {"value": float("-inf")}),
        ("cycle", {"value": cyclic}),
        ("excessive-depth", {"value": too_deep}),
        ("oversized-list", {"value": list(range(257))}),
        ("oversized-dict", {f"key_{index}": index for index in range(257)}),
        ("too-many-nodes", too_many_nodes),
    ]


def _nested_parameter(depth: int) -> dict[str, Any]:
    nested: Any = "leaf"
    for _ in range(depth):
        nested = [nested]
    return {"value": nested}


@pytest.mark.parametrize(
    ("case_name", "parameters"),
    _invalid_parameter_values(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_runner_rejects_invalid_parameters_before_tool_execution(
    case_name: str,
    parameters: Any,
) -> None:
    calls = {"count": 0}
    registry = _registry_for(
        _COUNTING_TOOL_DEFINITION,
        CountingTool(calls=calls),
    )
    authorization_service = AuthorizationService(
        _policy_for("counting_tool", "tool.counting_tool"),
        registry,
    )
    runner = _runner_for(registry, authorization_service)
    proposal = ToolProposal(
        tool_name="counting_tool",
        permission="tool.counting_tool",
        parameters=parameters,
    )

    with pytest.raises(AuthorizationError):
        runner.execute(proposal, ConversationContext.create("hello"))

    assert calls["count"] == 0


@pytest.mark.parametrize(
    ("case_name", "parameters"),
    (
        (
            "maximum-depth",
            _nested_parameter(31),
        ),
        ("maximum-container-size", {f"item_{index}": index for index in range(256)}),
        (
            "maximum-node-count",
            {
                **{
                    f"group_{index}": list(range(256))
                    for index in range(7)
                },
                "last_group": list(range(247)),
            },
        ),
    ),
)
def test_runner_accepts_parameter_limit_boundaries(
    case_name: str,
    parameters: dict[str, Any],
) -> None:
    calls = {"count": 0}
    registry = _registry_for(
        _COUNTING_TOOL_DEFINITION,
        CountingTool(calls=calls),
    )
    authorization_service = AuthorizationService(
        _policy_for("counting_tool", "tool.counting_tool"),
        registry,
    )
    runner = _runner_for(registry, authorization_service)

    result = runner.execute(
        ToolProposal(
            tool_name="counting_tool",
            permission="tool.counting_tool",
            parameters=parameters,
        ),
        ConversationContext.create("hello"),
    )

    assert result["status"] == "simulated_ok"
    assert calls["count"] == 1


@pytest.mark.parametrize(
    ("case_name", "parameters"),
    (
        (
            "excessive-depth",
            _nested_parameter(32),
        ),
        ("oversized-container", {f"item_{index}": index for index in range(257)}),
        (
            "excessive-node-count",
            {f"group_{index}": list(range(256)) for index in range(8)},
        ),
    ),
)
def test_runner_rejects_parameters_over_limit_boundaries(
    case_name: str,
    parameters: dict[str, Any],
) -> None:
    calls = {"count": 0}
    registry = _registry_for(
        _COUNTING_TOOL_DEFINITION,
        CountingTool(calls=calls),
    )
    authorization_service = AuthorizationService(
        _policy_for("counting_tool", "tool.counting_tool"),
        registry,
    )
    runner = _runner_for(registry, authorization_service)

    with pytest.raises(AuthorizationError):
        runner.execute(
            ToolProposal(
                tool_name="counting_tool",
                permission="tool.counting_tool",
                parameters=parameters,
            ),
            ConversationContext.create("hello"),
        )

    assert calls["count"] == 0


@pytest.mark.parametrize(
    ("case_name", "parameters"),
    _invalid_parameter_values(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_authorization_denies_invalid_tool_parameters(
    case_name: str,
    parameters: Any,
) -> None:
    registry = _registry_for(
        _COUNTING_TOOL_DEFINITION,
        CountingTool(calls={"count": 0}),
    )
    authorization_service = AuthorizationService(
        _policy_for("counting_tool", "tool.counting_tool"),
        registry,
    )
    proposal = ToolProposal(
        tool_name="counting_tool",
        permission="tool.counting_tool",
        parameters=parameters,
    )

    assert authorization_service.authorize(
        proposal,
        ConversationContext.create("hello"),
    ) is False


def test_runner_denies_if_parameter_validator_raises_ordinary_error(
    monkeypatch,
) -> None:
    calls = {"count": 0}
    registry = _registry_for(
        _COUNTING_TOOL_DEFINITION,
        CountingTool(calls=calls),
    )
    authorization_service = AuthorizationService(
        _policy_for("counting_tool", "tool.counting_tool"),
        registry,
    )
    runner = _runner_for(registry, authorization_service)
    proposal = ToolProposal(
        tool_name="counting_tool",
        permission="tool.counting_tool",
        parameters={"message": "hello"},
    )

    def fail_validation(self, tool_name, parameters):
        raise RuntimeError("validator failure")

    monkeypatch.setattr(ToolRegistry, "validate_tool_arguments", fail_validation)

    with pytest.raises(AuthorizationError, match="Authorization denied"):
        runner.execute(proposal, ConversationContext.create("hello"))

    assert calls["count"] == 0


def test_default_application_uses_shared_registry_and_echo_permission() -> None:
    app = create_application(Settings(mode="simulated", ai_enabled=False))

    result = app.process("repite: hello")

    assert app.registry.list_names() == ["echo_message"]
    assert app.registry is app.tool_runner.registry
    assert app.registry is app.authorizer.registry
    assert app.authorizer.policy.can_execute_tool("echo_message", "tool.echo_message")
    assert result.success is True
    assert result.authorized is True
    assert result.tool_result is not None
    assert result.tool_result["status"] == "simulated_ok"


def test_default_registry_constructor_does_not_accept_custom_catalog() -> None:
    with pytest.raises(TypeError):
        ToolRegistry(trusted_catalog={})


def test_authorization_rejects_registry_substitutes_and_subclasses(monkeypatch) -> None:
    calls = {"count": 0}
    registry = create_default_registry()

    def counted_echo(context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        return {"status": "unexpected"}

    monkeypatch.setattr(runner_module, "_execute_echo_message", counted_echo)

    class RegistrySubclass(ToolRegistry):
        __slots__ = ()

    substitute = object()
    with pytest.raises(TypeError, match="core tool registry"):
        AuthorizationService(SecurityPolicy.for_application(), substitute)  # type: ignore[arg-type]

    subclass_registry = RegistrySubclass()
    subclass_registry.register(EchoTool())
    subclass_registry.freeze()
    with pytest.raises(TypeError, match="core tool registry"):
        AuthorizationService(SecurityPolicy.for_application(), subclass_registry)

    authorization_service = AuthorizationService(SecurityPolicy.for_application(), registry)
    with pytest.raises(TypeError, match="core tool registry"):
        SimulatedToolRunner(subclass_registry, authorization_service)

    assert calls["count"] == 0


def test_authorization_and_runner_reject_unfrozen_registries() -> None:
    calls = {"count": 0}
    registry = ToolRegistry.for_trusted_composition(
        {"counting_tool": _COUNTING_TOOL_DEFINITION}
    )
    registry.register(CountingTool(calls=calls))

    assert registry.is_frozen is False
    assert type(registry.is_frozen) is bool
    with pytest.raises(ValueError, match="frozen tool registry"):
        AuthorizationService(
            _policy_for("counting_tool", "tool.counting_tool"),
            registry,
        )
    assert calls["count"] == 0

    registry.freeze()
    authorization_service = AuthorizationService(
        _policy_for("counting_tool", "tool.counting_tool"),
        registry,
    )
    assert registry.is_frozen is True
    assert type(registry.is_frozen) is bool

    # Simulate low-level state corruption to verify the runner checks its own boundary.
    object.__setattr__(
        registry,
        "_registered_tools",
        dict(registry._registered_tools),
    )
    assert registry.is_frozen is False
    with pytest.raises(ValueError, match="frozen tool registry"):
        SimulatedToolRunner(registry, authorization_service)
    assert calls["count"] == 0


def test_authorization_denies_registry_that_becomes_unfrozen() -> None:
    calls = {"count": 0}
    registry = _registry_for(
        _COUNTING_TOOL_DEFINITION,
        CountingTool(calls=calls),
    )
    authorization_service = AuthorizationService(
        _policy_for("counting_tool", "tool.counting_tool"),
        registry,
    )
    runner = _runner_for(registry, authorization_service)
    proposal = ToolProposal(
        tool_name="counting_tool",
        permission="tool.counting_tool",
    )
    context = ConversationContext.create("hello")
    original_tools = registry._registered_tools
    try:
        object.__setattr__(registry, "_registered_tools", dict(original_tools))

        assert registry.is_frozen is False
        assert authorization_service.authorize(proposal, context) is False
        with pytest.raises(AuthorizationError, match="could not be verified"):
            runner.execute(proposal, context)
        assert calls["count"] == 0
    finally:
        object.__setattr__(registry, "_registered_tools", original_tools)

    assert registry.is_frozen is True


def test_trusted_composition_rejects_registry_subclasses_and_custom_mappings() -> None:
    class RegistrySubclass(ToolRegistry):
        __slots__ = ()

    class Catalog(dict):
        pass

    with pytest.raises(TypeError, match="core tool registries"):
        RegistrySubclass.for_trusted_composition(
            {"counting_tool": _COUNTING_TOOL_DEFINITION}
        )
    with pytest.raises(TypeError, match="plain tool catalog"):
        ToolRegistry.for_trusted_composition(
            Catalog({"counting_tool": _COUNTING_TOOL_DEFINITION})
        )

    corrupted_definition = TrustedToolDefinition(
        name="counting_tool",
        implementation_type=CountingTool,
        required_permission="tool.counting_tool",
    )
    object.__setattr__(corrupted_definition, "required_permission", "invalid permission")
    with pytest.raises(ValueError):
        ToolRegistry.for_trusted_composition(
            {"counting_tool": corrupted_definition}
        )


def test_runner_rejects_different_registry_before_alternative_handler_runs() -> None:
    calls = {"count": 0}
    trusted_registry = create_default_registry()
    alternative_registry = _alternative_echo_registry(calls)

    authorization_service = AuthorizationService(
        policy=SecurityPolicy.for_application(),
        registry=trusted_registry,
    )

    with pytest.raises(ValueError, match="must share the same tool registry"):
        SimulatedToolRunner(
            registry=alternative_registry,
            authorization_service=authorization_service,
        )

    assert calls["count"] == 0


def test_production_runner_rejects_executor_injection_and_executes_builtin_echo(monkeypatch) -> None:
    calls = {"builtin": 0, "decoy": 0}
    registry = create_default_registry()
    authorization_service = AuthorizationService(
        policy=SecurityPolicy.for_application(),
        registry=registry,
    )
    original_builtin = runner_module._execute_echo_message

    def counted_builtin(context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        calls["builtin"] += 1
        return original_builtin(context, **kwargs)

    def decoy_executor(context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        calls["decoy"] += 1
        return {"kind": "decoy", "status": "ok"}

    monkeypatch.setattr(runner_module, "_execute_echo_message", counted_builtin)
    with pytest.raises(TypeError):
        SimulatedToolRunner(
            registry,
            authorization_service,
            executors={"echo_message": decoy_executor},  # type: ignore[call-arg]
        )
    assert calls["decoy"] == 0

    runner = SimulatedToolRunner(registry, authorization_service)
    result = runner.execute(
        ToolProposal(
            tool_name="echo_message",
            permission="tool.echo_message",
            parameters={"message": "trusted"},
        ),
        ConversationContext.create("trusted"),
    )

    assert result["result"]["message"] == "trusted"
    assert calls == {"builtin": 1, "decoy": 0}


def test_dispatcher_constructor_rejects_custom_executors() -> None:
    registry = create_default_registry()
    authorization_service = AuthorizationService(
        SecurityPolicy.for_application(),
        registry,
    )

    def alternative_executor(context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        return {"status": "alternative"}

    with pytest.raises(TypeError):
        _SimulatedToolDispatcher(
            registry,
            authorization_service,
            mode="simulated",
            executors={"echo_message": alternative_executor},  # type: ignore[call-arg]
        )

    dispatcher = _SimulatedToolDispatcher(
        registry,
        authorization_service,
        mode="simulated",
    )
    result = dispatcher.execute(
        ToolProposal(
            tool_name="echo_message",
            permission="tool.echo_message",
            parameters={"message": "built-in"},
        ),
        ConversationContext.create("built-in"),
    )

    assert result["result"]["message"] == "built-in"


def test_production_runner_does_not_dispatch_when_authorization_denies(monkeypatch) -> None:
    calls = {"builtin": 0, "decoy": 0}
    registry = create_default_registry()
    authorization_service = AuthorizationService(SecurityPolicy(), registry)
    runner = SimulatedToolRunner(registry, authorization_service)

    def counted_builtin(context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        calls["builtin"] += 1
        return {"status": "unexpected"}

    def decoy_executor(context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        calls["decoy"] += 1
        return {"status": "unexpected"}

    monkeypatch.setattr(runner_module, "_execute_echo_message", counted_builtin)
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "denied"},
    )

    with pytest.raises(AuthorizationError):
        runner.execute(proposal, ConversationContext.create("denied"))

    with pytest.raises(TypeError):
        SimulatedToolRunner(
            registry,
            authorization_service,
            executors={"echo_message": decoy_executor},  # type: ignore[call-arg]
        )

    assert calls == {"builtin": 0, "decoy": 0}


def test_production_runner_rejects_non_core_catalog_before_execution() -> None:
    calls = {"count": 0}
    registry = _alternative_echo_registry(calls)
    authorization_service = AuthorizationService(
        policy=SecurityPolicy.for_application(),
        registry=registry,
    )

    with pytest.raises(ValueError, match="trusted core definitions"):
        SimulatedToolRunner(registry, authorization_service)

    assert calls["count"] == 0


def test_runner_rejects_fake_authorizer_with_shared_registry(monkeypatch) -> None:
    calls = {"handler": 0, "authorize": 0}
    registry = create_default_registry()
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.unapproved",
        parameters={"message": "hello"},
    )

    def counted_echo(context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        calls["handler"] += 1
        return {"status": "ok"}

    class AlwaysAllowAuthorizer:
        def __init__(self, shared_registry: ToolRegistry) -> None:
            self.registry = shared_registry

        def authorize(self, proposal: ToolProposal, context: ConversationContext) -> bool:
            calls["authorize"] += 1
            return True

    monkeypatch.setattr(runner_module, "_execute_echo_message", counted_echo)
    replacement = AlwaysAllowAuthorizer(registry)

    with pytest.raises(TypeError, match="core authorization service"):
        SimulatedToolRunner(registry, replacement)  # type: ignore[arg-type]

    assert calls == {"handler": 0, "authorize": 0}


def test_runner_registry_reassignment_is_rejected_and_integrity_checked() -> None:
    calls = {"count": 0}
    registry = create_default_registry()
    authorization_service = AuthorizationService(
        policy=SecurityPolicy.for_application(),
        registry=registry,
    )
    runner = _runner_for(registry, authorization_service)
    alternative_registry = _alternative_echo_registry(calls)
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )

    with pytest.raises(AttributeError):
        setattr(runner, "registry", alternative_registry)

    # Simulate deliberate low-level state corruption beyond the ordinary API.
    object.__setattr__(runner, "_registry", alternative_registry)
    with pytest.raises(AuthorizationError, match="could not be verified"):
        runner.execute(proposal, ConversationContext.create("hello"))

    assert calls["count"] == 0


def test_authorizer_registry_reassignment_is_rejected_and_integrity_checked() -> None:
    calls = {"count": 0}
    registry = create_default_registry()
    authorization_service = AuthorizationService(
        policy=SecurityPolicy.for_application(),
        registry=registry,
    )
    runner = _runner_for(registry, authorization_service)
    alternative_registry = _alternative_echo_registry(calls)
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )

    with pytest.raises(AttributeError):
        setattr(authorization_service, "registry", alternative_registry)

    # Simulate deliberate low-level state corruption beyond the ordinary API.
    object.__setattr__(authorization_service, "_registry", alternative_registry)
    with pytest.raises(AuthorizationError, match="could not be verified"):
        runner.execute(proposal, ConversationContext.create("hello"))

    assert calls["count"] == 0


def test_runner_authorizer_replacement_is_rejected_before_unapproved_tool_execution(monkeypatch) -> None:
    calls = {"count": 0}
    app = create_application(Settings(mode="simulated", ai_enabled=False))

    def counted_echo(context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        return {"kind": "counted_echo", "status": "ok"}

    monkeypatch.setattr(runner_module, "_execute_echo_message", counted_echo)
    replacement_authorizer = type(
        "AlwaysAllowAuthorizer",
        (),
        {
            "registry": app.registry,
            "authorize": lambda self, proposal, context: True,
        },
    )()

    with pytest.raises(AttributeError):
        app.tool_runner._authorization_service = replacement_authorizer

    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.unapproved",
        parameters={"message": "hello"},
    )
    with pytest.raises(AuthorizationError):
        app.tool_runner.execute(proposal, ConversationContext.create("hello"))

    assert calls["count"] == 0


def test_runner_and_authorizer_do_not_expose_instance_dicts_for_security_state() -> None:
    calls = {"count": 0}
    registry = create_default_registry()
    deny_all = AuthorizationService(SecurityPolicy(), registry)
    runner = SimulatedToolRunner(registry, deny_all)

    assert not hasattr(runner, "__dict__")
    assert not hasattr(deny_all, "__dict__")
    with pytest.raises(AttributeError):
        _ = runner.__dict__
    with pytest.raises(AttributeError):
        _ = deny_all.__dict__
    with pytest.raises(AttributeError):
        runner._executors = {"echo_message": lambda *args, **kwargs: calls.update(count=1)}
    with pytest.raises(AttributeError):
        runner._authorization_service = AuthorizationService(
            SecurityPolicy.for_application(),
            registry,
        )
    with pytest.raises(AttributeError):
        runner._registry = ToolRegistry()
    with pytest.raises(AttributeError):
        deny_all.policy = SecurityPolicy.for_application()
    with pytest.raises(AttributeError):
        deny_all._registry = ToolRegistry()

    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "must remain denied"},
    )
    with pytest.raises(AuthorizationError):
        runner.execute(proposal, ConversationContext.create("hello"))

    assert calls["count"] == 0


def test_coordinated_registry_replacement_is_rejected_before_alternative_execution() -> None:
    calls = {"count": 0}
    registry = create_default_registry()
    authorization_service = AuthorizationService(
        policy=SecurityPolicy.for_application(),
        registry=registry,
    )
    runner = _runner_for(registry, authorization_service)
    alternative_registry = _alternative_echo_registry(calls)

    with pytest.raises(AttributeError):
        runner._registry = alternative_registry
    with pytest.raises(AttributeError):
        authorization_service._registry = alternative_registry

    assert runner.registry is registry
    assert authorization_service.registry is registry
    result = runner.execute(
        ToolProposal(
            tool_name="echo_message",
            permission="tool.echo_message",
            parameters={"message": "hello"},
        ),
        ConversationContext.create("hello"),
    )

    assert result["status"] == "simulated_ok"
    assert calls["count"] == 0


def test_authorization_service_policy_is_immutable(monkeypatch) -> None:
    calls = {"count": 0}
    registry = create_default_registry()
    policy = SecurityPolicy(
        identity_required=True,
        allowed_permissions={"tool.echo_message"},
        allowed_tool_permissions={("echo_message", "tool.echo_message")},
    )
    authorization_service = AuthorizationService(policy, registry)
    runner = _runner_for(registry, authorization_service)
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )
    context = ConversationContext.create("hello")

    def counted_echo(tool_context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        return {"status": "ok"}

    monkeypatch.setattr(runner_module, "_execute_echo_message", counted_echo)

    assert authorization_service.authorize(proposal, context) is False
    with pytest.raises(AttributeError):
        authorization_service.policy = SecurityPolicy.for_application()
    with pytest.raises(AttributeError):
        del authorization_service.policy

    with pytest.raises(AuthorizationError):
        runner.execute(proposal, context)

    assert calls["count"] == 0


def test_authorization_method_cannot_be_replaced_to_approve_unknown_permission(monkeypatch) -> None:
    calls = {"count": 0}
    registry = create_default_registry()
    authorization_service = AuthorizationService(SecurityPolicy.for_application(), registry)
    runner = _runner_for(registry, authorization_service)
    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.unapproved",
        parameters={"message": "hello"},
    )

    def counted_echo(context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        return {"status": "ok"}

    monkeypatch.setattr(runner_module, "_execute_echo_message", counted_echo)

    with pytest.raises(AttributeError):
        authorization_service.authorize = lambda proposal, context: True  # type: ignore[method-assign]
    with pytest.raises(AttributeError):
        del authorization_service.authorize
    with pytest.raises(AuthorizationError):
        runner.execute(proposal, ConversationContext.create("hello"))

    assert calls["count"] == 0


def test_unregistered_tool_is_denied_by_runner() -> None:
    calls = {"count": 0}
    registry = _registry_for(_COUNTING_TOOL_DEFINITION, CountingTool(calls=calls))
    auth = AuthorizationService(
        policy=_policy_for("counting_tool", "tool.counting_tool"),
        registry=registry,
    )
    runner = _runner_for(registry, auth)
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="missing_tool",
        permission="tool.echo_message",
        parameters={"message": "hello"},
    )
    with pytest.raises(ToolNotRegisteredError):
        runner.execute(proposal, context)
    assert calls["count"] == 0


def test_registry_does_not_expose_an_executable_tool_instance() -> None:
    registry = create_default_registry()
    definition = registry.get_trusted_definition("echo_message")

    assert not hasattr(registry, "get")
    assert not hasattr(definition, "execute")
    assert not hasattr(EchoTool(), "execute")

    with pytest.raises(AttributeError):
        EchoTool().execute(ConversationContext.create("hello"), message="bypass")


def test_untrusted_direct_dispatch_attempt_never_invokes_test_executor() -> None:
    calls = {"count": 0}
    registry = _registry_for(
        _COUNTING_TOOL_DEFINITION,
        CountingTool(calls=calls),
    )
    authorization_service = AuthorizationService(
        SecurityPolicy(),
        registry,
    )
    runner = _runner_for(registry, authorization_service)

    with pytest.raises(AuthorizationError):
        runner.execute(
            ToolProposal(
                tool_name="counting_tool",
                permission="tool.counting_tool",
                parameters={"message": "denied"},
            ),
            ConversationContext.create("hello"),
        )

    assert calls["count"] == 0


def test_runner_rejects_non_simulated_mode_before_tool_execution() -> None:
    calls = {"count": 0}
    registry = _registry_for(_COUNTING_TOOL_DEFINITION, CountingTool(calls=calls))
    auth = AuthorizationService(
        policy=_policy_for("counting_tool", "tool.counting_tool"),
        registry=registry,
    )
    runner = _runner_for(registry, auth, mode="real")
    proposal = ToolProposal(
        tool_name="counting_tool",
        permission="tool.counting_tool",
    )

    with pytest.raises(SimulatedModeError):
        runner.execute(proposal, ConversationContext.create("hello"))
    assert calls["count"] == 0


def test_production_runner_mode_cannot_be_modified_or_deleted() -> None:
    registry = create_default_registry()
    authorization_service = AuthorizationService(
        SecurityPolicy.for_application(),
        registry,
    )
    runner = SimulatedToolRunner(registry, authorization_service)

    assert runner.mode == "simulated"
    with pytest.raises(AttributeError):
        runner.mode = "real"
    with pytest.raises(AttributeError):
        del runner.mode
    assert runner.mode == "simulated"


def test_production_runner_rejects_invalid_mode_during_construction(monkeypatch) -> None:
    calls = {"count": 0}
    registry = create_default_registry()
    authorization_service = AuthorizationService(
        SecurityPolicy.for_application(),
        registry,
    )

    def count_echo(context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        return {"status": "unexpected"}

    monkeypatch.setattr(runner_module, "_execute_echo_message", count_echo)

    class ModeString(str):
        pass

    for invalid_mode in ("real", True, None, ModeString("simulated")):
        with pytest.raises(SimulatedModeError):
            SimulatedToolRunner(
                registry,
                authorization_service,
                mode=invalid_mode,
            )

    assert calls["count"] == 0


def test_authorization_error_stops_execution() -> None:
    calls = {"count": 0}
    registry = _registry_for(_COUNTING_TOOL_DEFINITION, CountingTool(calls=calls))
    policy = SecurityPolicy()
    auth = AuthorizationService(policy=policy, registry=registry)
    runner = _runner_for(registry, auth)
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="counting_tool",
        permission="tool.counting_tool",
        parameters={"message": "hello"},
    )
    with pytest.raises(AuthorizationError):
        runner.execute(proposal, context)
    assert calls["count"] == 0


def test_authorization_exception_prevents_execution(monkeypatch) -> None:
    calls = {"count": 0}
    registry = _registry_for(_COUNTING_TOOL_DEFINITION, CountingTool(calls=calls))
    policy = _policy_for("counting_tool", "tool.counting_tool")
    auth = AuthorizationService(policy=policy, registry=registry)
    runner = _runner_for(registry, auth)
    context = ConversationContext.create("hello")
    proposal = ToolProposal(
        tool_name="counting_tool",
        permission="tool.counting_tool",
        parameters={"message": "hello"},
    )

    def boom(*args, **kwargs):
        raise AuthorizationError("authorization exploded")

    monkeypatch.setattr(AuthorizationService, "authorize", boom)
    with pytest.raises(AuthorizationError, match="could not be verified"):
        runner.execute(proposal, context)
    assert calls["count"] == 0


def test_confirmation_cannot_be_forged_or_replayed() -> None:
    calls = {"count": 0}
    registry = _registry_for(_CONFIRMING_TOOL_DEFINITION, ConfirmingTool(calls=calls))
    registry.freeze()
    auth = AuthorizationService(
        policy=_policy_for("confirming_tool", "tool.confirming_tool"),
        registry=registry,
    )
    runner = _runner_for(registry, auth)
    context = ConversationContext.create("confirm")
    context.confirmed_actions.add("confirming_tool:approved")

    proposals = (
        ToolProposal(
            tool_name="confirming_tool",
            permission="tool.confirming_tool",
            risk_level="high",
            requires_confirmation=False,
            action_ref="confirming_tool:approved",
        ),
        ToolProposal(
            tool_name="confirming_tool",
            permission="tool.confirming_tool",
            risk_level="high",
            requires_confirmation=True,
            confirmation_token=None,
            action_ref="confirming_tool:approved",
        ),
        ToolProposal(
            tool_name="confirming_tool",
            permission="tool.confirming_tool",
            risk_level="high",
            requires_confirmation=True,
            confirmation_token="",
            action_ref="confirming_tool:approved",
        ),
        ToolProposal(
            tool_name="confirming_tool",
            permission="tool.confirming_tool",
            risk_level="high",
            requires_confirmation=True,
            confirmation_token="reused-fake-token",
            action_ref="another-action",
        ),
    )
    for proposal in proposals:
        with pytest.raises(AuthorizationError):
            runner.execute(proposal, context)

    assert calls["count"] == 0


def test_audit_logger_stores_only_allowlisted_metadata() -> None:
    logger = AuditLogger()
    entry = logger.log(
        session_id="session-12345678",
        action_ref="echo:0123456789ab",
        outcome="allowed",
        details={
            "authorization": "granted",
            "execution_status": "completed",
            "result_code": "simulated_ok",
            "tool_name": "echo_message",
            "token": "abc123",
            "status": "ok",
        },
    )
    assert entry["details"] == {
        "authorization": "granted",
        "execution_status": "completed",
        "result_code": "simulated_ok",
        "tool_name": "echo_message",
    }
    assert "abc123" not in str(entry)


def test_registry_rejects_invalid_tool_objects() -> None:
    registry = ToolRegistry()
    with pytest.raises(TypeError):
        registry.register(object())


def test_registry_rejects_malformed_tool_security_metadata() -> None:
    invalid_changes = (
        {"name": "Invalid Name"},
        {"required_permission": ""},
        {"required_permission": "tool.bad permission"},
        {"execution_mode": "real"},
        {"risk_level": "critical"},
        {"requires_confirmation": 1},
    )
    for changes in invalid_changes:
        tool = CountingTool(calls={"count": 0})
        for attribute, value in changes.items():
            setattr(tool, attribute, value)
        with pytest.raises((TypeError, ValueError)):
            _registry_for(_COUNTING_TOOL_DEFINITION, tool)


def test_untrusted_tool_cannot_reuse_echo_permission() -> None:
    calls = {"count": 0}
    tool = CountingTool(calls=calls)
    tool.required_permission = "tool.echo_message"

    with pytest.raises(ValueError, match="not in the trusted catalog"):
        ToolRegistry().register(tool)

    assert calls["count"] == 0


def test_tool_metadata_cannot_reduce_trusted_risk_or_confirmation() -> None:
    calls = {"count": 0}
    tool = ConfirmingTool(calls=calls)
    tool.risk_level = "low"
    tool.requires_confirmation = False
    registry = ToolRegistry.for_trusted_composition(
        {"confirming_tool": _CONFIRMING_TOOL_DEFINITION}
    )

    with pytest.raises(ValueError, match="does not match the trusted catalog"):
        registry.register(tool)

    assert calls["count"] == 0


def test_tool_subclasses_cannot_create_instance_dictionaries() -> None:
    with pytest.raises(TypeError, match="must use slots"):
        class DictionaryTool(ToolDefinition):
            pass

    with pytest.raises(TypeError, match="cannot override frozen security attributes"):
        class ShadowedSecurityTool(ToolDefinition):
            __slots__ = ()
            execution_mode = "real"

    with pytest.raises(TypeError, match="cannot implement executable behavior"):
        class ExecutingTool(ToolDefinition):
            __slots__ = ()

            def execute(self, context, **kwargs):
                return {}

    tool = EchoTool()
    assert not hasattr(tool, "__dict__")
    assert not hasattr(tool, "execute")


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    with pytest.raises(ValueError):
        registry.register(EchoTool())


def test_registered_tool_rejects_subclass_attribute_reassignment() -> None:
    calls = {"count": 0}
    tool = CountingTool(calls=calls)
    registry = _registry_for(_COUNTING_TOOL_DEFINITION, tool)

    for attribute, value in (
        ("calls", {"count": 10}),
        ("runtime_state", "changed"),
        ("required_permission", "tool.echo_message"),
        ("risk_level", "high"),
        ("requires_confirmation", True),
        ("execution_mode", "real"),
    ):
        with pytest.raises(AttributeError):
            setattr(tool, attribute, value)

    assert registry.has("counting_tool")
    assert registry.get_trusted_definition("counting_tool") == _COUNTING_TOOL_DEFINITION


def test_registry_freeze_blocks_modification() -> None:
    calls = {"count": 0}
    registry = ToolRegistry.for_trusted_composition(
        {"counting_tool": _COUNTING_TOOL_DEFINITION}
    )
    registry.register(CountingTool(calls=calls))
    assert registry.is_frozen is False
    registry.freeze()
    assert registry.is_frozen is True
    assert type(registry.is_frozen) is bool
    assert not hasattr(registry, "_sealed")
    with pytest.raises(AttributeError):
        del registry._registered_tools
    with pytest.raises((AttributeError, TypeError)):
        registry._registered_tools["extra"] = object()
    with pytest.raises(RuntimeError, match="sealed"):
        registry.register(CountingTool(calls=calls))
    assert calls["count"] == 0


def test_runner_rejects_low_level_executor_map_replacement() -> None:
    calls = {"decoy": 0}
    registry = create_default_registry()
    authorization_service = AuthorizationService(
        SecurityPolicy.for_application(),
        registry,
    )
    runner = SimulatedToolRunner(registry, authorization_service)

    def decoy_executor(
        context: ConversationContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls["decoy"] += 1
        return {"status": "unexpected"}

    object.__setattr__(
        runner,
        "_executors",
        {"echo_message": decoy_executor},
    )

    proposal = ToolProposal(
        tool_name="echo_message",
        permission="tool.echo_message",
        parameters={"message": "integrity-check"},
    )

    try:
        with pytest.raises(AuthorizationError):
            runner.execute(proposal, ConversationContext.create("integrity-check"))
        assert calls["decoy"] == 0
    finally:
        object.__setattr__(
            runner,
            "_executors",
            runner_module._TRUSTED_EXECUTORS,
        )
