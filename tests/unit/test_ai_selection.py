import pytest
from pydantic import ValidationError

import jarvis.tools.runner as runner_module
from jarvis.ai.selection import (
    AIProviderSelector,
    SelectionReason,
    SelectionTarget,
)
from jarvis.ai.contracts import InformationalRequest
from jarvis.ai.simulator import SimulatedAIProvider
from jarvis.app import create_application
from jarvis.config.settings import Settings
from jarvis.core.context import ConversationContext
from jarvis.conversation.models import RequestEnvelope
from jarvis.conversation.router import DeterministicRequestRouter
from jarvis.internet import (
    InternetAccessPolicy,
    InternetServiceCapability,
    InternetServiceRequest,
    InternetServiceSelector,
    InternetServiceUnavailableReason,
)
from jarvis.security.authorization import AuthorizationService
from jarvis.tools.registry import ToolRegistry, create_default_registry


def _decision(text: str):
    return DeterministicRequestRouter().route(RequestEnvelope(text))


def test_registered_echo_tool_is_selected_before_any_ai_provider() -> None:
    provider = SimulatedAIProvider()
    selector = AIProviderSelector({"simulated": provider})

    selection = selector.select(
        _decision("echo: hello"),
        create_default_registry(),
        Settings(ai_enabled=True),
    )

    assert selection.target is SelectionTarget.DETERMINISTIC_TOOL
    assert selection.reason is SelectionReason.REGISTERED_TOOL
    assert selection.provider is None


def test_echo_is_unavailable_when_the_registered_deterministic_tool_is_missing() -> None:
    empty_registry = ToolRegistry.for_trusted_composition({})
    empty_registry.freeze()
    selector = AIProviderSelector({"simulated": SimulatedAIProvider()})

    selection = selector.select(
        _decision("echo: hello"),
        empty_registry,
        Settings(ai_enabled=True),
    )

    assert selection.target is SelectionTarget.UNAVAILABLE
    assert selection.provider is None


def test_local_provider_is_selected_only_when_registered_and_generation_is_enabled() -> None:
    local_provider = SimulatedAIProvider()
    selector = AIProviderSelector({"local": local_provider})

    selection = selector.select(
        _decision("pregunta: ¿qué significa?"),
        create_default_registry(),
        Settings(ai_enabled=True, ai_provider="local"),
    )

    assert selection.target is SelectionTarget.AI_PROVIDER
    assert selection.provider_name == "local"
    assert selection.provider is local_provider


def test_missing_local_provider_does_not_fall_back_to_simulated_provider() -> None:
    simulated_provider = SimulatedAIProvider()
    selector = AIProviderSelector({"simulated": simulated_provider})

    selection = selector.select(
        _decision("pregunta: ¿qué significa?"),
        create_default_registry(),
        Settings(ai_enabled=True, ai_provider="local"),
    )

    assert selection.target is SelectionTarget.UNAVAILABLE
    assert selection.reason is SelectionReason.PROVIDER_NOT_AVAILABLE
    assert selection.provider is None


def test_generation_is_unavailable_when_ai_is_disabled() -> None:
    selector = AIProviderSelector({"simulated": SimulatedAIProvider()})

    selection = selector.select(
        _decision("pregunta: ¿qué significa?"),
        create_default_registry(),
        Settings(ai_enabled=False),
    )

    assert selection.target is SelectionTarget.UNAVAILABLE
    assert selection.reason is SelectionReason.AI_DISABLED
    assert selection.provider is None


def test_remote_provider_is_blocked_even_when_configured_and_registered() -> None:
    remote_provider = SimulatedAIProvider()
    simulated_provider = SimulatedAIProvider()
    selector = AIProviderSelector(
        {"remote": remote_provider, "simulated": simulated_provider}
    )

    selection = selector.select(
        _decision("pregunta: no enviar este texto"),
        create_default_registry(),
        Settings(
            ai_enabled=True,
            ai_provider="remote",
            remote_ai_enabled=True,
        ),
    )

    assert selection.target is SelectionTarget.UNAVAILABLE
    assert selection.reason is SelectionReason.REMOTE_PRIVACY_POLICY
    assert selection.provider is None


def test_remote_configuration_never_invokes_remote_or_simulated_fallback() -> None:
    app = create_application(
        Settings(
            ai_enabled=True,
            ai_provider="remote",
            remote_ai_enabled=True,
        )
    )
    calls = {"remote": 0, "simulated": 0}

    class CountingProvider(SimulatedAIProvider):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def respond_informational(self, request):
            calls[self.name] += 1
            return super().respond_informational(request)

    app.provider_selector = AIProviderSelector(
        {
            "remote": CountingProvider("remote"),
            "simulated": CountingProvider("simulated"),
        }
    )

    result = app.process("pregunta: no enviar este texto")

    assert result.success is False
    assert result.authorized is False
    assert result.proposal is None
    assert result.error == "Informative processing is unavailable."
    assert calls == {"remote": 0, "simulated": 0}


def test_remote_provider_cannot_be_enabled_without_explicit_configuration() -> None:
    with pytest.raises(ValidationError):
        Settings(
            ai_enabled=True,
            ai_provider="remote",
            remote_ai_enabled=False,
        )


def test_provider_selection_for_generation_does_not_execute_tools(monkeypatch) -> None:
    app = create_application(Settings(ai_enabled=True, ai_provider="local"))
    calls = {"provider": 0, "runner": 0}

    class CountingLocalProvider(SimulatedAIProvider):
        def respond_informational(self, request):
            calls["provider"] += 1
            return super().respond_informational(request)

    local_provider = CountingLocalProvider()
    app.provider_selector = AIProviderSelector({"local": local_provider})

    def unexpected_runner(self, proposal, context):
        calls["runner"] += 1
        assert self is app.tool_runner
        assert proposal is not None
        assert context is not None
        raise AssertionError("Provider selection reached the tool runner.")

    monkeypatch.setattr(type(app.tool_runner), "execute", unexpected_runner)

    result = app.process("pregunta: contenido informativo")

    assert result.success is True
    assert result.authorized is False
    assert result.proposal is None
    assert result.tool_result is None
    assert calls == {"provider": 1, "runner": 0}


def test_application_reports_unavailable_provider_without_exposing_details() -> None:
    app = create_application(Settings(ai_enabled=True, ai_provider="local"))
    result = app.process("pregunta: texto privado")

    assert result.success is False
    assert result.error == "Informative processing is unavailable."
    assert "texto privado" not in str(result.to_dict()["error"])


@pytest.mark.parametrize("text", ("pregunta:", "abre notepad", "texto libre"))
def test_terminal_routes_do_not_select_any_provider(text: str) -> None:
    selector = AIProviderSelector({"simulated": SimulatedAIProvider()})

    selection = selector.select(
        _decision(text),
        create_default_registry(),
        Settings(ai_enabled=True),
    )

    assert selection.target is SelectionTarget.TERMINAL
    assert selection.provider is None


def test_web_search_is_not_selected_or_blocked_by_ai_provider_selection() -> None:
    ai_selector = AIProviderSelector({"simulated": SimulatedAIProvider()})
    internet_selector = InternetServiceSelector()
    request = InternetServiceRequest(
        capability=InternetServiceCapability.WEB_SEARCH,
        query="local search query",
    )

    ai_selection = ai_selector.select(
        _decision("web: local search query"),
        create_default_registry(),
        Settings(ai_enabled=False),
    )
    internet_selection = internet_selector.select(request)

    assert ai_selection.target is SelectionTarget.TERMINAL
    assert ai_selection.provider is None
    assert internet_selection.capability is InternetServiceCapability.WEB_SEARCH
    assert (
        internet_selection.reason
        is InternetServiceUnavailableReason.POLICY_DENIED
    )


def test_allowed_web_search_without_adapter_is_controlled_unavailability() -> None:
    selector = InternetServiceSelector(
        InternetAccessPolicy(
            allowed_capabilities=frozenset(
                {InternetServiceCapability.WEB_SEARCH}
            )
        )
    )

    selection = selector.select(
        InternetServiceRequest(
            capability=InternetServiceCapability.WEB_SEARCH,
            query="query that must not get invented results",
        )
    )

    assert selection.capability is InternetServiceCapability.WEB_SEARCH
    assert (
        selection.reason
        is InternetServiceUnavailableReason.ADAPTER_UNAVAILABLE
    )


@pytest.mark.parametrize(
    "query",
    (
        "",
        " \t ",
        "bad\x00query",
        "x" * 8001,
        42,
    ),
)
def test_internet_service_request_rejects_malformed_query_values(query) -> None:
    with pytest.raises((TypeError, ValueError)):
        InternetServiceRequest(
            capability=InternetServiceCapability.WEB_SEARCH,
            query=query,
        )


def test_pipeline_returns_controlled_error_for_nul_in_web_query() -> None:
    app = create_application(Settings(ai_enabled=False))
    selector_calls = []

    class UnexpectedInternetSelector:
        def select(self, request):
            selector_calls.append(request)
            raise AssertionError("Malformed query reached the Internet selector.")

    app.internet_service_selector = UnexpectedInternetSelector()

    result = app.process("web: malformed\x00query")

    assert result.success is False
    assert result.authorized is False
    assert result.proposal is None
    assert result.tool_result is None
    assert result.error == "Request was denied."
    assert "malformed" not in str(result.to_dict()["error"])
    assert selector_calls == []


def test_pipeline_routes_web_search_without_ai_or_tool_authorization(monkeypatch) -> None:
    app = create_application(Settings(ai_enabled=True))
    calls = {
        "provider": 0,
        "runner": 0,
        "tool": 0,
        "adapter": 0,
        "authorization": 0,
        "service": 0,
    }
    observed_requests = []
    contexts = []

    class ProviderSpy(SimulatedAIProvider):
        def classify_intent(self, user_text):
            assert type(user_text) is str
            calls["provider"] += 1
            raise AssertionError("Search route classified through an AI provider.")

        def propose_action(self, user_text):
            assert type(user_text) is str
            calls["provider"] += 1
            raise AssertionError("Search route requested an AI action proposal.")

        def respond(self, user_text):
            assert type(user_text) is str
            calls["provider"] += 1
            raise AssertionError("Search route invoked a general AI response.")

        def respond_informational(self, request):
            assert type(request) is InformationalRequest
            calls["provider"] += 1
            raise AssertionError("Search route invoked an informational provider.")

    class SearchSpy:
        def select(self, request):
            assert type(request) is InternetServiceRequest
            calls["service"] += 1
            observed_requests.append(request)
            return InternetServiceSelector().select(request)

        def search(self, request):
            calls["adapter"] += 1
            raise AssertionError("A policy-denied request reached an adapter.")

    def unexpected_runner(self, proposal, context):
        calls["runner"] += 1
        assert self is app.tool_runner
        assert proposal is not None
        assert context in contexts
        raise AssertionError("Search route reached the local tool runner.")

    def unexpected_authorization(self, proposal, context):
        calls["authorization"] += 1
        assert self is app.authorizer
        assert proposal is not None
        assert context in contexts
        raise AssertionError("Search route invoked tool authorization.")

    original_tool = runner_module._execute_echo_message

    def unexpected_tool(context, **kwargs):
        calls["tool"] += 1
        return original_tool(context, **kwargs)

    original_create = ConversationContext.create

    def track_context(cls, user_text, claims=None):
        assert cls is ConversationContext
        context = original_create(user_text, claims)
        contexts.append(context)
        return context

    monkeypatch.setattr(type(app.tool_runner), "execute", unexpected_runner)
    monkeypatch.setattr(AuthorizationService, "authorize", unexpected_authorization)
    monkeypatch.setattr(runner_module, "_execute_echo_message", unexpected_tool)
    monkeypatch.setattr(
        ConversationContext,
        "create",
        classmethod(track_context),
    )
    app.provider_selector = AIProviderSelector({"simulated": ProviderSpy()})
    app.internet_service_selector = SearchSpy()

    query_marker = "PRIVATE_QUERY_MARKER"
    result = app.process(f"busca en Internet: {query_marker}")

    assert result.success is False
    assert result.authorized is False
    assert result.proposal is None
    assert result.tool_result is None
    assert result.error == "Internet search is unavailable."
    assert query_marker not in result.error
    assert calls == {
        "provider": 0,
        "runner": 0,
        "tool": 0,
        "adapter": 0,
        "authorization": 0,
        "service": 1,
    }
    assert observed_requests
    assert (
        InternetServiceSelector().select(observed_requests[0]).reason
        is InternetServiceUnavailableReason.POLICY_DENIED
    )
    assert len(observed_requests) == 1
    assert observed_requests[0].capability is InternetServiceCapability.WEB_SEARCH
    assert observed_requests[0].query == query_marker
    assert "PRIVATE_QUERY_MARKER" not in repr(observed_requests[0])
    assert len(contexts) == 1
    assert contexts[0].claims == {}
    assert contexts[0].metadata == {}
    assert contexts[0].confirmed_actions == set()


def test_unexpected_internet_selector_error_is_controlled_and_isolated(
    monkeypatch,
) -> None:
    app = create_application(Settings(ai_enabled=True))
    private_marker = "PRIVATE_SELECTOR_FAILURE"
    calls = {"provider": 0, "runner": 0, "authorization": 0, "tool": 0}

    class ProviderSpy(SimulatedAIProvider):
        def respond_informational(self, request):
            calls["provider"] += 1
            raise AssertionError("Web search invoked an AI provider.")

    class FailingSelector:
        def select(self, request):
            assert type(request) is InternetServiceRequest
            raise RuntimeError(private_marker)

    def unexpected_runner(self, proposal, context):
        calls["runner"] += 1
        raise AssertionError("Web search reached the local tool runner.")

    def unexpected_authorization(self, proposal, context):
        calls["authorization"] += 1
        raise AssertionError("Web search invoked tool authorization.")

    def unexpected_tool(context, **kwargs):
        calls["tool"] += 1
        raise AssertionError("Web search invoked the Echo tool.")

    provider = ProviderSpy()
    app.ai_provider = provider
    app.provider_selector = AIProviderSelector({"simulated": provider})
    app.internet_service_selector = FailingSelector()
    monkeypatch.setattr(type(app.tool_runner), "execute", unexpected_runner)
    monkeypatch.setattr(AuthorizationService, "authorize", unexpected_authorization)
    monkeypatch.setattr(runner_module, "_execute_echo_message", unexpected_tool)

    result = app.process("web: a valid query")

    assert result.success is False
    assert result.authorized is False
    assert result.proposal is None
    assert result.tool_result is None
    assert result.error == "Internet search is unavailable."
    assert private_marker not in str(result.to_dict())
    assert calls == {"provider": 0, "runner": 0, "authorization": 0, "tool": 0}


def test_web_policy_changes_preserve_echo_and_informational_isolation(monkeypatch) -> None:
    app = create_application(Settings(ai_enabled=True))
    calls = {"provider": 0, "runner": 0, "authorization": 0, "tool": 0}

    class ProviderSpy(SimulatedAIProvider):
        def respond_informational(self, request):
            calls["provider"] += 1
            return super().respond_informational(request)

    provider = ProviderSpy()
    app.provider_selector = AIProviderSelector({"simulated": provider})
    app.ai_provider = provider

    original_execute = type(app.tool_runner).execute

    def count_runner(self, proposal, context):
        calls["runner"] += 1
        return original_execute(self, proposal, context)

    original_authorize = AuthorizationService.authorize

    def count_authorization(self, proposal, context):
        calls["authorization"] += 1
        return original_authorize(self, proposal, context)

    original_tool = runner_module._execute_echo_message

    def count_tool(context, **kwargs):
        calls["tool"] += 1
        return original_tool(context, **kwargs)

    monkeypatch.setattr(type(app.tool_runner), "execute", count_runner)
    monkeypatch.setattr(AuthorizationService, "authorize", count_authorization)
    monkeypatch.setattr(runner_module, "_execute_echo_message", count_tool)

    echo_result = app.process("echo: hola")
    assert echo_result.success is True
    assert echo_result.authorized is True
    assert echo_result.tool_result is not None
    assert calls == {
        "provider": 0,
        "runner": 1,
        "authorization": 1,
        "tool": 1,
    }

    informational_result = app.process("pregunta: define this")
    assert informational_result.success is True
    assert informational_result.authorized is False
    assert informational_result.proposal is None
    assert informational_result.tool_result is None
    assert calls == {
        "provider": 1,
        "runner": 1,
        "authorization": 1,
        "tool": 1,
    }


def test_internet_selector_validation_errors_are_sanitized() -> None:
    app = create_application(Settings(ai_enabled=False))
    private_marker = "PRIVATE_POLICY_FAILURE"

    class FailingSelector:
        def select(self, request):
            assert type(request) is InternetServiceRequest
            raise ValueError(private_marker)

    app.internet_service_selector = FailingSelector()

    result = app.process("web: a valid query")

    assert result.success is False
    assert result.error == "Internet search is unavailable."
    assert private_marker not in str(result.to_dict())


def test_web_search_selector_has_separate_deny_by_default_network_policy() -> None:
    selector = InternetServiceSelector()
    selection = selector.select(
        InternetServiceRequest(
            capability=InternetServiceCapability.WEB_SEARCH,
            query="private query",
        )
    )

    assert selection.reason is InternetServiceUnavailableReason.POLICY_DENIED
    assert not hasattr(selection, "permission")
    assert not hasattr(selection, "claims")


def test_selector_rejects_untrusted_provider_catalog_entries() -> None:
    with pytest.raises(ValueError, match="unsupported provider"):
        AIProviderSelector({"arbitrary_callable": SimulatedAIProvider()})

    with pytest.raises(TypeError, match="invalid provider"):
        AIProviderSelector({"local": object()})


def test_provider_catalog_cannot_be_replaced_after_construction() -> None:
    selector = AIProviderSelector({"simulated": SimulatedAIProvider()})

    with pytest.raises(AttributeError, match="immutable"):
        selector._providers = {}
    with pytest.raises(AttributeError, match="cannot be deleted"):
        del selector._providers


def test_selector_rejects_invalid_route_registry_and_settings_types() -> None:
    selector = AIProviderSelector({"simulated": SimulatedAIProvider()})
    registry = create_default_registry()
    settings = Settings(ai_enabled=True)

    with pytest.raises(TypeError, match="validated route"):
        selector.select(object(), registry, settings)
    with pytest.raises(TypeError, match="trusted tool registry"):
        selector.select(_decision("pregunta: algo"), object(), settings)
    with pytest.raises(TypeError, match="validated settings"):
        selector.select(_decision("pregunta: algo"), registry, object())
