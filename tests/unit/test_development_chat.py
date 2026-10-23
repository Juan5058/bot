import json
from urllib.error import URLError

import pytest

from jarvis.ai.contracts import InformationalRequest, InformationalResponse
from jarvis.ai.development_chat import DevelopmentOllamaChatProvider
from jarvis.config.settings import Settings


class FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")
        self.read_limit = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        self.read_limit = limit
        return self.body[:limit]


class FakeOpener:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.request = None
        self.timeout = None
        self.response = None

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout
        if self.error is not None:
            raise self.error
        self.response = FakeResponse(self.payload)
        return self.response


def make_settings(**overrides):
    values = {
        "ai_enabled": True,
        "ai_provider": "local",
        "local_development_chat_enabled": True,
        "local_ollama_model": "qwen3:1.7b",
    }
    values.update(overrides)
    return Settings(**values)


def make_request():
    return InformationalRequest(
        capability="answer_question",
        content="¿Quién desarrolla JARVIS CORE?",
    )


def test_informational_request_posts_to_local_ollama(monkeypatch):
    opener = FakeOpener({
        "done": True,
        "message": {"content": "JARVIS CORE es tu proyecto."},
    })
    monkeypatch.setattr(
        "jarvis.ai.development_chat.build_opener",
        lambda *args: opener,
    )

    provider = DevelopmentOllamaChatProvider(make_settings())
    result = provider.respond_informational(make_request())

    assert type(result) is InformationalResponse
    assert result.text == "JARVIS CORE es tu proyecto."
    assert opener.request.full_url == "http://127.0.0.1:11434/api/chat"
    assert opener.request.get_method() == "POST"
    assert opener.timeout == 60

    payload = json.loads(opener.request.data.decode("utf-8"))
    assert payload["model"] == "qwen3:1.7b"
    assert payload["stream"] is False
    assert payload["keep_alive"] == "5m"
    assert payload["options"] == {"num_ctx": 4096, "num_predict": 384}
    assert "No afirmes que JARVIS CORE fue desarrollado por Google" in (
        payload["messages"][0]["content"]
    )
    assert opener.response.read_limit == 1_000_001


def test_provider_requires_explicit_opt_in():
    settings = Settings(ai_enabled=True, ai_provider="local")

    with pytest.raises(ValueError, match="not enabled"):
        DevelopmentOllamaChatProvider(settings)


def test_settings_rejects_development_chat_when_ai_is_disabled():
    with pytest.raises(
        ValueError,
        match="Development chat requires enabled local AI",
    ):
        make_settings(ai_enabled=False)


def test_provider_rejects_incomplete_ollama_response(monkeypatch):
    opener = FakeOpener({
        "done": False,
        "message": {"content": "respuesta incompleta"},
    })
    monkeypatch.setattr(
        "jarvis.ai.development_chat.build_opener",
        lambda *args: opener,
    )

    provider = DevelopmentOllamaChatProvider(make_settings())

    with pytest.raises(RuntimeError, match="invalid response"):
        provider.respond_informational(make_request())


def test_provider_rejects_empty_content_without_exposing_thinking(monkeypatch):
    private_thinking = "razonamiento interno que no debe mostrarse"
    opener = FakeOpener({
        "done": True,
        "message": {"content": "", "thinking": private_thinking},
    })
    monkeypatch.setattr(
        "jarvis.ai.development_chat.build_opener",
        lambda *args: opener,
    )

    provider = DevelopmentOllamaChatProvider(make_settings())

    with pytest.raises(RuntimeError, match="invalid response") as exc:
        provider.respond_informational(make_request())

    assert private_thinking not in str(exc.value)


def test_provider_handles_connection_failure_without_leaking_details(monkeypatch):
    opener = FakeOpener(error=URLError("private transport detail"))
    monkeypatch.setattr(
        "jarvis.ai.development_chat.build_opener",
        lambda *args: opener,
    )

    provider = DevelopmentOllamaChatProvider(make_settings())

    with pytest.raises(RuntimeError, match="could not be reached reliably") as exc:
        provider.respond_informational(make_request())

    assert "private transport detail" not in str(exc.value)


def test_provider_fails_closed_for_action_methods():
    provider = DevelopmentOllamaChatProvider(make_settings())

    with pytest.raises(RuntimeError, match="does not classify actionable intent"):
        provider.classify_intent("ejecuta un comando")

    with pytest.raises(RuntimeError, match="cannot propose tool actions"):
        provider.propose_action("abre una aplicación")

    with pytest.raises(RuntimeError, match="only supports informational requests"):
        provider.respond("ejecuta un comando")


def test_application_registers_development_chat_only_with_opt_in():
    from jarvis.app import create_application

    disabled_settings = Settings(
        ai_enabled=True,
        ai_provider="local",
    )
    disabled_app = create_application(disabled_settings)

    assert "local" not in disabled_app.provider_selector._providers

    enabled_app = create_application(make_settings())

    assert type(
        enabled_app.provider_selector._providers["local"]
    ) is DevelopmentOllamaChatProvider


def test_application_rejects_development_chat_with_model_manager():
    from jarvis.app import create_application

    with pytest.raises(
        ValueError,
        match="cannot be combined with a production local model manager",
    ):
        create_application(
            make_settings(),
            local_model_manager=object(),
        )
