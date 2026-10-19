import pytest

from jarvis.ai.contracts import (
    ActionSuggestion,
    InformationalRequest,
    InformationalResponse,
    MAX_INFORMATIONAL_RESPONSE_LENGTH,
)
from jarvis.ai.simulator import SimulatedAIProvider


def test_simulated_provider_returns_untrusted_action_suggestion_only() -> None:
    provider = SimulatedAIProvider()

    suggestion = provider.propose_action("hello")

    assert type(suggestion) is ActionSuggestion
    assert suggestion.tool_name == "echo_message"
    assert suggestion.parameters == {"message": "hello"}
    assert not hasattr(suggestion, "permission")
    assert not hasattr(suggestion, "requires_confirmation")
    assert not hasattr(suggestion, "confirmation_token")
    assert not hasattr(suggestion, "identity_verified")


def test_informational_contract_is_separate_from_action_suggestions() -> None:
    request = InformationalRequest(
        capability="summarize_text",
        content="A short source text",
    )
    response = InformationalResponse(text="A short summary")

    assert request.capability == "summarize_text"
    assert request.content == "A short source text"
    assert response.text == "A short summary"
    assert "A short" not in repr(request)
    assert "A short" not in repr(response)
    assert not hasattr(response, "tool_name")
    assert not hasattr(response, "parameters")
    assert not hasattr(response, "permission")


@pytest.mark.parametrize(
    "kwargs",
    (
        {"capability": "echo_message", "content": "hello"},
        {"capability": "answer_question", "content": ""},
        {"capability": "answer_question", "content": "x" * 8001},
        {
            "capability": "answer_question",
            "content": "hello",
            "permission": "tool.echo_message",
        },
    ),
)
def test_informational_request_rejects_unsupported_or_malformed_fields(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        InformationalRequest(**kwargs)


@pytest.mark.parametrize(
    "text",
    (
        "",
        " \t",
        "x" * (MAX_INFORMATIONAL_RESPONSE_LENGTH + 1),
        "valid\x00invalid",
        42,
    ),
)
def test_informational_response_rejects_empty_oversized_or_non_text(text) -> None:
    with pytest.raises((TypeError, ValueError)):
        InformationalResponse(text=text)


def test_simulated_provider_returns_only_informational_text() -> None:
    response = SimulatedAIProvider().respond_informational(
        InformationalRequest(
            capability="answer_question",
            content="What is this?",
        )
    )

    assert type(response) is InformationalResponse
    assert response.text == "Respuesta simulada para: What is this?"
    assert not hasattr(response, "tool_name")
    assert not hasattr(response, "permission")
