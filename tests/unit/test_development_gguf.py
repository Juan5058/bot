from __future__ import annotations

import pytest

from jarvis.ai.contracts import InformationalRequest, InformationalResponse
from jarvis.ai.development_gguf import DevelopmentGGUFChatProvider


class FakeModel:
    def __init__(self, content: str = "Respuesta de prueba.") -> None:
        self.content = content
        self.calls: list[dict] = []
        self.closed = False

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "choices": [
                {"message": {"content": self.content}}
            ]
        }

    def close(self) -> None:
        self.closed = True


def make_request() -> InformationalRequest:
    return InformationalRequest(
        capability="answer_question",
        content="¿Qué es JARVIS CORE?",
    )


def make_provider(tmp_path, model, **kwargs):
    model_path = tmp_path / "test-model.gguf"
    model_path.write_bytes(b"fake model marker")
    return DevelopmentGGUFChatProvider(
        model_path,
        enabled=True,
        model_factory=lambda **factory_kwargs: model,
        **kwargs,
    )


def test_provider_does_not_load_model_during_construction(tmp_path):
    calls = []
    model_path = tmp_path / "test-model.gguf"
    model_path.write_bytes(b"fake model marker")

    provider = DevelopmentGGUFChatProvider(
        model_path,
        enabled=True,
        model_factory=lambda **kwargs: calls.append(kwargs) or FakeModel(),
    )

    assert calls == []


def test_disabled_provider_rejects_inference_without_loading(tmp_path):
    provider = DevelopmentGGUFChatProvider(tmp_path / "missing.gguf")

    with pytest.raises(RuntimeError, match="disabled"):
        provider.respond_informational(make_request())


def test_informational_request_returns_validated_response(tmp_path):
    model = FakeModel("JARVIS CORE es un proyecto de asistente.")
    provider = make_provider(tmp_path, model)

    response = provider.respond_informational(make_request())

    assert type(response) is InformationalResponse
    assert response.text == "JARVIS CORE es un proyecto de asistente."
    assert model.calls[0]["max_tokens"] == 128
    assert model.calls[0]["messages"][1]["content"] == "¿Qué es JARVIS CORE?"


@pytest.mark.parametrize(
    "method,args",
    [
        ("classify_intent", ("abre el navegador",)),
        ("propose_action", ("abre el navegador",)),
        ("respond", ("abre el navegador",)),
    ],
)
def test_provider_rejects_action_capable_methods(tmp_path, method, args):
    provider = DevelopmentGGUFChatProvider(tmp_path / "missing.gguf")
    with pytest.raises(RuntimeError):
        getattr(provider, method)(*args)


def test_provider_rejects_wrong_request_type(tmp_path):
    provider = DevelopmentGGUFChatProvider(tmp_path / "missing.gguf", enabled=True)

    with pytest.raises(TypeError):
        provider.respond_informational("pregunta")


def test_model_is_loaded_only_once(tmp_path):
    model = FakeModel()
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return model

    model_path = tmp_path / "test-model.gguf"
    model_path.write_bytes(b"fake model marker")
    provider = DevelopmentGGUFChatProvider(
        model_path,
        enabled=True,
        model_factory=factory,
    )

    provider.respond_informational(make_request())
    provider.respond_informational(make_request())

    assert len(factory_calls) == 1
    assert len(model.calls) == 2


def test_invalid_model_output_is_rejected(tmp_path):
    provider = make_provider(tmp_path, FakeModel(""))

    with pytest.raises(RuntimeError, match="empty response"):
        provider.respond_informational(make_request())


def test_model_output_with_control_characters_is_rejected(tmp_path):
    provider = make_provider(tmp_path, FakeModel("respuesta\x00invalida"))

    with pytest.raises(RuntimeError, match="invalid response"):
        provider.respond_informational(make_request())


def test_close_releases_model_and_allows_reload(tmp_path):
    models = []

    def factory(**kwargs):
        model = FakeModel()
        models.append(model)
        return model

    model_path = tmp_path / "test-model.gguf"
    model_path.write_bytes(b"fake model marker")
    provider = DevelopmentGGUFChatProvider(
        model_path,
        enabled=True,
        model_factory=factory,
    )

    provider.respond_informational(make_request())
    provider.close()

    assert models[0].closed is True

    provider.respond_informational(make_request())
    assert len(models) == 2