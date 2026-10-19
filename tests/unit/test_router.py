import pytest

from jarvis.conversation.models import MAX_USER_INPUT_LENGTH, RequestEnvelope
from jarvis.conversation.router import (
    AmbiguityCode,
    Capability,
    DeterministicRequestRouter,
    InterpretationSource,
    InterpretationStatus,
    RequiredField,
    RouteCategory,
    RouteDecision,
    RouteReason,
)


def _route_decision(**fields) -> RouteDecision:
    return RouteDecision(request=RequestEnvelope("echo: hello"), **fields)


def test_request_envelope_validates_and_redacts_request_text() -> None:
    marker = "private request"
    request = RequestEnvelope(marker)

    assert request.text == marker
    assert len(request.request_id) == 32
    assert marker not in repr(request)


@pytest.mark.parametrize("text", (None, 1, [], {}))
def test_request_envelope_rejects_non_string_text(text) -> None:
    with pytest.raises(TypeError, match="Request text must be a string"):
        RequestEnvelope(text)


@pytest.mark.parametrize("text", ("", " \t\n"))
def test_request_envelope_rejects_empty_text(text: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        RequestEnvelope(text)


def test_request_envelope_rejects_text_over_the_existing_input_limit() -> None:
    with pytest.raises(ValueError, match="8000 characters"):
        RequestEnvelope("x" * (MAX_USER_INPUT_LENGTH + 1))


def test_request_contract_has_no_permission_field_or_arbitrary_extras() -> None:
    with pytest.raises(TypeError):
        RequestEnvelope("repite: hi", permission="tool.echo_message")


def test_router_recognizes_only_explicit_echo_commands() -> None:
    router = DeterministicRequestRouter()

    for text in ("repite: Hola", "ECHO: Hola"):
        request = RequestEnvelope(text)
        decision = router.route(request)
        assert decision.category is RouteCategory.SPECIALIZED
        assert decision.reason is RouteReason.ECHO_REQUEST
        assert decision.capability is Capability.ECHO_MESSAGE
        assert decision.intent is Capability.ECHO_MESSAGE
        assert decision.request is request
        assert decision.status is InterpretationStatus.UNDERSTOOD
        assert decision.source is InterpretationSource.DETERMINISTIC_ROUTER
        assert decision.parameters == {"message": "Hola"}
        assert decision.message == "Hola"
        assert not decision.needs_clarification
        assert not hasattr(decision, "permission")


@pytest.mark.parametrize(
    ("text", "expected_message"),
    (
        ("  eChO  :  Hola  ", "Hola"),
        ("  RePiTe: hola: de nuevo  ", "hola: de nuevo"),
        ("echo:: hola", ": hola"),
    ),
)
def test_router_normalizes_aliases_and_preserves_delimited_message(
    text: str,
    expected_message: str,
) -> None:
    decision = DeterministicRequestRouter().route(RequestEnvelope(text))

    assert decision.category is RouteCategory.SPECIALIZED
    assert decision.capability is Capability.ECHO_MESSAGE
    assert decision.message == expected_message
    assert not hasattr(decision, "permission")
    assert not hasattr(decision, "claims")


@pytest.mark.parametrize("text", ("repite", "echo:", "repite:   "))
def test_router_clarifies_missing_echo_arguments(text: str) -> None:
    decision = DeterministicRequestRouter().route(RequestEnvelope(text))

    assert decision.category is RouteCategory.CLARIFY
    assert decision.reason is RouteReason.MISSING_ARGUMENT
    assert decision.needs_clarification
    assert decision.missing_fields == (RequiredField.MESSAGE,)


def test_router_clarifies_unrecognized_free_text_instead_of_echoing() -> None:
    decision = DeterministicRequestRouter().route(RequestEnvelope("Hola JARVIS"))

    assert decision.category is RouteCategory.CLARIFY
    assert decision.reason is RouteReason.UNKNOWN_COMMAND
    assert decision.missing_fields == (RequiredField.COMMAND,)


@pytest.mark.parametrize(
    "text",
    (
        "foo: bar",
        "resume: texto",
        "abre notepad",
        "abrir aplicación calculadora",
        "busca en Internet gatos",
    ),
)
def test_router_rejects_unknown_commands_and_unsupported_effectful_capabilities(
    text: str,
) -> None:
    decision = DeterministicRequestRouter().route(RequestEnvelope(text))

    assert decision.category is RouteCategory.REJECT
    assert decision.reason is RouteReason.UNSUPPORTED_CAPABILITY
    assert decision.capability is None
    assert decision.message is None


@pytest.mark.parametrize(
    ("text", "capability"),
    (
        ("pregunta: ¿qué hora es?", Capability.ANSWER_QUESTION),
        ("resumen: este texto", Capability.SUMMARIZE_TEXT),
    ),
)
def test_router_marks_future_generation_without_implicitly_executing_it(
    text: str,
    capability: Capability,
) -> None:
    decision = DeterministicRequestRouter().route(RequestEnvelope(text))

    assert decision.category is RouteCategory.GENERATIVE
    assert decision.reason is RouteReason.GENERATIVE_REQUEST
    assert decision.capability is capability
    assert decision.status is InterpretationStatus.UNDERSTOOD
    assert decision.message is not None


@pytest.mark.parametrize(
    ("text", "query"),
    (
        ("web: local query", "local query"),
        ("busca en Internet: noticias", "noticias"),
    ),
)
def test_router_recognizes_explicit_web_search_as_external_service(
    text: str,
    query: str,
) -> None:
    decision = DeterministicRequestRouter().route(RequestEnvelope(text))

    assert decision.category is RouteCategory.EXTERNAL_SERVICE
    assert decision.reason is RouteReason.EXTERNAL_SERVICE_REQUEST
    assert decision.capability is Capability.WEB_SEARCH
    assert decision.status is InterpretationStatus.UNDERSTOOD
    assert decision.parameters == {"query": query}
    assert not hasattr(decision, "permission")


@pytest.mark.parametrize("text", ("web:", "busca en Internet:"))
def test_router_requests_web_search_query_when_missing(text: str) -> None:
    decision = DeterministicRequestRouter().route(RequestEnvelope(text))

    assert decision.category is RouteCategory.CLARIFY
    assert decision.capability is Capability.WEB_SEARCH
    assert decision.missing_fields == (RequiredField.QUERY,)


@pytest.mark.parametrize("text", ("pregunta:", "resumen:"))
def test_router_clarifies_missing_generative_input(text: str) -> None:
    decision = DeterministicRequestRouter().route(RequestEnvelope(text))

    assert decision.category is RouteCategory.CLARIFY
    assert decision.missing_fields == (RequiredField.TEXT,)


def test_route_decision_rejects_invalid_category_and_inconsistent_shape() -> None:
    with pytest.raises(TypeError, match="Route category is invalid"):
        _route_decision(category="specialized", reason=RouteReason.ECHO_REQUEST)

    with pytest.raises(ValueError, match="Specialized routes require"):
        _route_decision(
            category=RouteCategory.SPECIALIZED,
            reason=RouteReason.ECHO_REQUEST,
            capability=Capability.ECHO_MESSAGE,
        )

    with pytest.raises(ValueError, match="does not match its category"):
        _route_decision(
            category=RouteCategory.REJECT,
            reason=RouteReason.ECHO_REQUEST,
        )


def test_route_decision_rejects_extra_security_fields() -> None:
    with pytest.raises(TypeError):
        _route_decision(
            category=RouteCategory.SPECIALIZED,
            reason=RouteReason.ECHO_REQUEST,
            capability=Capability.ECHO_MESSAGE,
            parameters={"message": "hello"},
            permission="tool.echo_message",
        )


@pytest.mark.parametrize(
    "fields",
    (
        {
            "category": RouteCategory.CLARIFY,
            "reason": RouteReason.MISSING_ARGUMENT,
            "capability": Capability.ECHO_MESSAGE,
            "missing_fields": (RequiredField.TEXT,),
        },
        {
            "category": RouteCategory.CLARIFY,
            "reason": RouteReason.UNKNOWN_COMMAND,
            "parameters": {"unexpected": "hidden payload"},
            "missing_fields": (RequiredField.COMMAND,),
        },
    ),
)
def test_route_decision_rejects_mismatched_clarification_fields(fields) -> None:
    with pytest.raises(ValueError):
        _route_decision(**fields)


def test_route_decision_deeply_freezes_parameters_and_returns_independent_copy() -> None:
    original = {
        "message": "hello",
        "metadata": {"items": [1, {"enabled": True}]},
    }
    decision = _route_decision(
        category=RouteCategory.SPECIALIZED,
        reason=RouteReason.ECHO_REQUEST,
        capability=Capability.ECHO_MESSAGE,
        parameters=original,
    )

    original["message"] = "changed"
    original["metadata"]["items"][1]["enabled"] = False
    original["metadata"]["items"].append(2)

    assert decision.parameters["message"] == "hello"
    assert decision.parameters["metadata"]["items"] == (1, {"enabled": True})
    with pytest.raises(TypeError):
        decision.parameters["message"] = "changed"
    with pytest.raises(TypeError):
        decision.parameters["metadata"]["items"][1]["enabled"] = False
    with pytest.raises(AttributeError):
        decision.parameters["metadata"]["items"].append(2)

    mutable_copy = decision.mutable_parameters()
    mutable_copy["metadata"]["items"][1]["enabled"] = False
    assert decision.parameters["metadata"]["items"][1]["enabled"] is True


@pytest.mark.parametrize(
    "parameters",
    (
        [],
        {"message": True},
        {"message": "hello", "bad": object()},
        {"message": "hello", "nested": {"too_deep": []}},
    ),
)
def test_route_decision_rejects_malformed_or_unsupported_parameter_values(
    parameters,
) -> None:
    if parameters == {"message": "hello", "nested": {"too_deep": []}}:
        nested = {}
        current = nested
        for _ in range(33):
            child = {}
            current["child"] = child
            current = child
        parameters = {"message": "hello", "nested": nested}

    with pytest.raises((TypeError, ValueError)):
        _route_decision(
            category=RouteCategory.SPECIALIZED,
            reason=RouteReason.ECHO_REQUEST,
            capability=Capability.ECHO_MESSAGE,
            parameters=parameters,
        )


def test_route_decision_rejects_parameter_cycles() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)

    with pytest.raises(ValueError, match="cycles"):
        _route_decision(
            category=RouteCategory.SPECIALIZED,
            reason=RouteReason.ECHO_REQUEST,
            capability=Capability.ECHO_MESSAGE,
            parameters={"message": "hello", "extra": cyclic},
        )


def test_route_decision_rejects_unknown_source_and_ambiguous_understood_state() -> None:
    with pytest.raises(TypeError, match="source"):
        _route_decision(
            category=RouteCategory.SPECIALIZED,
            reason=RouteReason.ECHO_REQUEST,
            capability=Capability.ECHO_MESSAGE,
            parameters={"message": "hello"},
            source="untrusted_provider",
        )

    with pytest.raises(ValueError, match="Only clarification"):
        _route_decision(
            category=RouteCategory.SPECIALIZED,
            reason=RouteReason.ECHO_REQUEST,
            capability=Capability.ECHO_MESSAGE,
            parameters={"message": "hello"},
            ambiguities=(AmbiguityCode.INTENT_UNCLEAR,),
        )


def test_route_decision_rejects_unknown_intent_and_derived_security_fields() -> None:
    with pytest.raises(TypeError, match="capability"):
        _route_decision(
            category=RouteCategory.SPECIALIZED,
            reason=RouteReason.ECHO_REQUEST,
            capability="echo_message",
            parameters={"message": "hello"},
        )

    with pytest.raises(TypeError):
        _route_decision(
            category=RouteCategory.SPECIALIZED,
            reason=RouteReason.ECHO_REQUEST,
            capability=Capability.ECHO_MESSAGE,
            parameters={"message": "hello"},
            status=InterpretationStatus.UNDERSTOOD,
        )


def test_ambiguous_interpretation_requires_clarification_without_guessing() -> None:
    decision = _route_decision(
        category=RouteCategory.CLARIFY,
        reason=RouteReason.UNKNOWN_COMMAND,
        ambiguities=(AmbiguityCode.INTENT_UNCLEAR,),
    )

    assert decision.status is InterpretationStatus.NEEDS_CLARIFICATION
    assert decision.ambiguities == (AmbiguityCode.INTENT_UNCLEAR,)
    assert decision.intent is None
    assert decision.parameters == {}
