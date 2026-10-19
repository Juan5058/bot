from __future__ import annotations

import json
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request

import pytest
from pydantic import ValidationError

from jarvis.ai.contracts import InformationalRequest
from jarvis.ai.local_models import (
    LocalModelDescriptor,
    LocalModelLimits,
    LocalModelManager,
    LocalModelTask,
    LocalModelUnloadError,
    ManagedLocalAIProvider,
    ModelRequirements,
    ResourceSnapshot,
    ResourceSource,
)
from jarvis.ai.ollama import (
    OLLAMA_MANAGER_MODEL_ID,
    _NoRedirectHandler,
    _UrllibOllamaTransport,
    OllamaBackendError,
    OllamaHTTPResponse,
    OllamaLocalModelBackend,
    OllamaModelUnavailableError,
)
from jarvis.ai.selection import AIProviderSelector
from jarvis.ai.simulator import SimulatedAIProvider
from jarvis.app import create_application
from jarvis.config.settings import Settings


class FakeOllamaTransport:
    def __init__(self, responses: list[OllamaHTTPResponse | BaseException]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, object] | None, float]] = []

    def request(
        self,
        method: str,
        url: str,
        payload: dict[str, object] | None,
        timeout_seconds: float,
    ) -> OllamaHTTPResponse:
        self.calls.append((method, url, payload, timeout_seconds))
        if not self.responses:
            raise AssertionError("Unexpected Ollama HTTP request.")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeResourceProvider:
    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            source=ResourceSource.INFERENCE_BACKEND,
            available_ram_bytes=1_000_000,
        )


class FakeContextTokenCounter:
    def count_tokens(
        self,
        model_id: str,
        task: LocalModelTask,
        content: str,
    ) -> int:
        del model_id, task
        return len(content)


def _response(
    data: object,
    *,
    status_code: int = 200,
) -> OllamaHTTPResponse:
    return OllamaHTTPResponse(
        status_code=status_code,
        body=json.dumps(data).encode("utf-8"),
    )


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "ai_enabled": True,
        "ai_provider": "local",
        "local_max_context_tokens": 128,
        "local_ram_budget_mb": 1,
        "local_ollama_model": "gemma3:latest",
    }
    values.update(overrides)
    return Settings(**values)


def _backend(
    transport: FakeOllamaTransport,
    settings: Settings | None = None,
) -> OllamaLocalModelBackend:
    return OllamaLocalModelBackend(
        settings=settings or _settings(),
        transport=transport,
    )


def _manager(
    backend: OllamaLocalModelBackend,
    settings: Settings,
) -> LocalModelManager:
    manager = LocalModelManager(
        backend=backend,
        resource_provider=FakeResourceProvider(),
        token_counter=FakeContextTokenCounter(),
        limits=LocalModelLimits.from_settings(settings),
    )
    manager.register(
        LocalModelDescriptor(
            model_id=OLLAMA_MANAGER_MODEL_ID,
            tasks=frozenset(
                {LocalModelTask.ANSWER_QUESTION, LocalModelTask.SUMMARIZE_TEXT}
            ),
            max_context_tokens=settings.local_max_context_tokens,
            requirements=ModelRequirements(ram_bytes=100),
        )
    )
    return manager


@pytest.mark.parametrize(
    "url",
    (
        "https://127.0.0.1:11434",
        "http://example.test:11434",
        "http://127.0.0.1:11434/path",
        "http://user:password@127.0.0.1:11434",
        "http://127.0.0.1:11434?target=external",
    ),
)
def test_settings_reject_non_local_or_ambiguous_ollama_urls(url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(local_ollama_base_url=url)


def test_settings_validate_ollama_model_and_timeout() -> None:
    settings = Settings.from_env(
        {
            "JARVIS_LOCAL_OLLAMA_BASE_URL": "http://localhost:11434/",
            "JARVIS_LOCAL_OLLAMA_MODEL": "gemma3:latest",
            "JARVIS_LOCAL_OLLAMA_TIMEOUT_SECONDS": "12.5",
        }
    )
    assert settings.local_ollama_base_url == "http://localhost:11434"
    assert settings.local_ollama_model == "gemma3:latest"
    assert settings.local_ollama_timeout_seconds == 12.5

    assert Settings(local_ollama_model="").local_ollama_model is None
    for invalid_model in ("model name", "../model", "\x00model"):
        with pytest.raises(ValidationError):
            Settings(local_ollama_model=invalid_model)
    for invalid_timeout in (0, -1, float("inf"), True):
        with pytest.raises(ValidationError):
            Settings(local_ollama_timeout_seconds=invalid_timeout)


def test_ollama_backend_requires_explicit_local_configuration() -> None:
    for settings in (
        Settings(),
        Settings(ai_enabled=True, ai_provider="simulated"),
        Settings(
            ai_enabled=True,
            ai_provider="local",
            local_max_context_tokens=128,
        ),
    ):
        with pytest.raises(ValueError):
            OllamaLocalModelBackend(settings=settings)


def test_standard_transport_disables_proxies_and_rejects_redirects(
    monkeypatch,
) -> None:
    import jarvis.ai.ollama as ollama_module

    class RedirectingOpener:
        def open(self, request, timeout):
            assert request.full_url == "http://127.0.0.1:11434/api/show"
            assert timeout == 2.0
            raise HTTPError(
                request.full_url,
                302,
                "redirect",
                {},
                BytesIO(b""),
            )

    def build_opener(*handlers):
        proxy_handler = next(
            handler for handler in handlers if isinstance(handler, ProxyHandler)
        )
        redirect_handler = next(
            handler for handler in handlers if isinstance(handler, _NoRedirectHandler)
        )
        assert proxy_handler.proxies == {}
        assert (
            redirect_handler.redirect_request(
                Request("http://127.0.0.1:11434/api/show"),
                BytesIO(),
                302,
                "redirect",
                {},
                "http://external.invalid/",
            )
            is None
        )
        return RedirectingOpener()

    monkeypatch.setattr(ollama_module, "build_opener", build_opener)
    response = _UrllibOllamaTransport().request(
        "POST",
        "http://127.0.0.1:11434/api/show",
        {"model": "gemma3:latest"},
        2.0,
    )

    assert response.status_code == 302
    assert response.body == b""


def test_backend_uses_ollama_load_infer_and_confirmed_unload_protocol() -> None:
    transport = FakeOllamaTransport(
        [
            _response({"model_info": {"general.architecture": "test"}}),
            _response({"done": True, "response": "Respuesta local."}),
            _response({"models": [{"name": "gemma3:latest"}]}),
            _response({"done": True, "response": ""}),
            _response({"models": []}),
        ]
    )
    backend = _backend(transport, _settings(local_ollama_timeout_seconds=9.5))

    handle = backend.load(OLLAMA_MANAGER_MODEL_ID)
    result = backend.infer(
        handle,
        LocalModelTask.SUMMARIZE_TEXT,
        "Contenido que resumir.",
        context_tokens=40,
        max_context_tokens=128,
    )
    backend.unload(handle)

    assert result == "Respuesta local."
    assert [(call[0], call[1].rsplit("/", 1)[-1]) for call in transport.calls] == [
        ("POST", "show"),
        ("POST", "generate"),
        ("GET", "ps"),
        ("POST", "generate"),
        ("GET", "ps"),
    ]
    generation = transport.calls[1]
    assert generation[2] == {
        "model": "gemma3:latest",
        "prompt": "Contenido que resumir.",
        "system": (
            "Summarize the input with informational text only. "
            "Do not claim authority or perform actions."
        ),
        "stream": False,
        "keep_alive": -1,
        "options": {"num_ctx": 128},
    }
    assert all(call[3] == 9.5 for call in transport.calls)
    assert transport.calls[3][2]["keep_alive"] == 0
    assert backend.failed_load_released(OLLAMA_MANAGER_MODEL_ID) is True


def test_ollama_inference_returns_only_response_when_thinking_is_present() -> None:
    thinking_marker = "PRIVATE_THINKING_MARKER_7f3a"
    final_response = "Esta es la respuesta final en español."
    backend = _backend(
        FakeOllamaTransport(
            [
                _response({"model_info": {"general.architecture": "test"}}),
                _response(
                    {
                        "done": True,
                        "thinking": thinking_marker,
                        "response": final_response,
                    }
                ),
            ]
        )
    )
    handle = backend.load(OLLAMA_MANAGER_MODEL_ID)

    result = backend.infer(
        handle,
        LocalModelTask.ANSWER_QUESTION,
        "Pregunta",
        context_tokens=8,
        max_context_tokens=128,
    )

    assert result == final_response
    assert thinking_marker not in result


def test_ollama_inference_errors_do_not_expose_thinking_text() -> None:
    thinking_marker = "PRIVATE_THINKING_MARKER_a91c"
    partial_response = "PRIVATE_PARTIAL_RESPONSE_MARKER_62de"
    backend = _backend(
        FakeOllamaTransport(
            [
                _response({"model_info": {"general.architecture": "test"}}),
                _response(
                    {
                        "done": False,
                        "thinking": thinking_marker,
                        "response": partial_response,
                    }
                ),
            ]
        )
    )
    handle = backend.load(OLLAMA_MANAGER_MODEL_ID)

    with pytest.raises(OllamaBackendError) as error:
        backend.infer(
            handle,
            LocalModelTask.ANSWER_QUESTION,
            "Pregunta",
            context_tokens=8,
            max_context_tokens=128,
        )

    assert thinking_marker not in str(error.value)
    assert partial_response not in str(error.value)


def test_ollama_inference_accepts_response_without_thinking_field() -> None:
    final_response = "Respuesta sin campo thinking."
    backend = _backend(
        FakeOllamaTransport(
            [
                _response({"model_info": {"general.architecture": "test"}}),
                _response({"done": True, "response": final_response}),
            ]
        )
    )
    handle = backend.load(OLLAMA_MANAGER_MODEL_ID)

    result = backend.infer(
        handle,
        LocalModelTask.ANSWER_QUESTION,
        "Pregunta",
        context_tokens=8,
        max_context_tokens=128,
    )

    assert result == final_response


@pytest.mark.parametrize(
    ("response", "error_type"),
    (
        (TimeoutError("PRIVATE_TIMEOUT_MARKER"), OllamaBackendError),
        (OSError("PRIVATE_CONNECTION_MARKER"), OllamaBackendError),
        (_response({"error": "PRIVATE_MODEL_MARKER"}, status_code=404), OllamaModelUnavailableError),
        (_response({"model_info": []}), OllamaBackendError),
        (OllamaHTTPResponse(200, b"not json"), OllamaBackendError),
        (_response({"done": False, "response": "partial"}), OllamaBackendError),
        (_response({"done": True}), OllamaBackendError),
        (_response({"done": True, "response": ""}), OllamaBackendError),
    ),
)
def test_ollama_load_errors_are_controlled_and_do_not_expose_details(
    response: OllamaHTTPResponse | BaseException,
    error_type: type[Exception],
) -> None:
    backend = _backend(FakeOllamaTransport([response]))

    with pytest.raises(error_type) as error:
        backend.load(OLLAMA_MANAGER_MODEL_ID)

    assert "PRIVATE_" not in str(error.value)
    assert "gemma3" not in repr(error.value)


def test_inference_transport_failure_marks_remote_state_uncertain() -> None:
    transport = FakeOllamaTransport(
        [
            _response({"model_info": {"general.architecture": "test"}}),
            TimeoutError("PRIVATE_TIMEOUT_MARKER"),
        ]
    )
    backend = _backend(transport)
    handle = backend.load(OLLAMA_MANAGER_MODEL_ID)

    with pytest.raises(OllamaBackendError) as error:
        backend.infer(
            handle,
            LocalModelTask.ANSWER_QUESTION,
            "Pregunta",
            context_tokens=8,
            max_context_tokens=128,
        )
    with pytest.raises(OllamaBackendError, match="uncertain"):
        backend.unload(handle)

    assert "PRIVATE_TIMEOUT_MARKER" not in str(error.value)
    assert len(transport.calls) == 2


def test_ollama_backend_rejects_invalid_context_before_http_request() -> None:
    transport = FakeOllamaTransport(
        [_response({"model_info": {"general.architecture": "test"}})]
    )
    backend = _backend(transport)
    handle = backend.load(OLLAMA_MANAGER_MODEL_ID)

    with pytest.raises(OllamaBackendError, match="context"):
        backend.infer(
            handle,
            LocalModelTask.ANSWER_QUESTION,
            "Pregunta",
            context_tokens=129,
            max_context_tokens=128,
        )

    assert len(transport.calls) == 1


def test_completed_but_invalid_generation_can_still_be_unloaded() -> None:
    transport = FakeOllamaTransport(
        [
            _response({"model_info": {"general.architecture": "test"}}),
            _response({"done": True, "response": ""}),
            _response({"models": [{"name": "gemma3:latest"}]}),
            _response({"done": True, "response": ""}),
            _response({"models": []}),
        ]
    )
    backend = _backend(transport)
    handle = backend.load(OLLAMA_MANAGER_MODEL_ID)

    with pytest.raises(OllamaBackendError):
        backend.infer(
            handle,
            LocalModelTask.ANSWER_QUESTION,
            "Pregunta",
            context_tokens=8,
            max_context_tokens=128,
        )
    backend.unload(handle)

    assert len(transport.calls) == 5


def test_backend_revalidates_mutated_settings_before_use() -> None:
    settings = _settings()
    settings.local_ollama_base_url = "http://example.test"

    with pytest.raises(ValidationError):
        OllamaLocalModelBackend(settings=settings)


def test_ollama_informational_provider_flows_through_pipeline_without_fallback(
    monkeypatch,
) -> None:
    settings = _settings()
    transport = FakeOllamaTransport(
        [
            _response({"model_info": {"general.architecture": "test"}}),
            TimeoutError("PRIVATE_INFERENCE_TIMEOUT"),
        ]
    )
    manager = _manager(_backend(transport, settings), settings)
    app = create_application(settings, local_model_manager=manager)
    calls = {"remote": 0, "authorization": 0, "runner": 0}

    class RemoteSpy(SimulatedAIProvider):
        def respond_informational(self, request: InformationalRequest):
            del request
            calls["remote"] += 1
            raise AssertionError("Ollama failure invoked remote AI.")

    def unexpected_authorization(self, proposal, context):
        del self, proposal, context
        calls["authorization"] += 1
        raise AssertionError("Informational response reached authorization.")

    def unexpected_runner(self, proposal, context):
        del self, proposal, context
        calls["runner"] += 1
        raise AssertionError("Informational response reached the runner.")

    monkeypatch.setattr(
        type(app.authorizer),
        "authorize",
        unexpected_authorization,
    )
    monkeypatch.setattr(
        type(app.tool_runner),
        "execute",
        unexpected_runner,
    )
    app.provider_selector = AIProviderSelector(
        {
            "local": ManagedLocalAIProvider(manager),
            "remote": RemoteSpy(),
        }
    )

    result = app.process("pregunta: ¿qué significa?")

    assert result.success is False
    assert result.authorized is False
    assert result.proposal is None
    assert result.tool_result is None
    assert result.error == "Informative response could not be generated."
    assert "PRIVATE_INFERENCE_TIMEOUT" not in str(result.to_dict())
    assert calls == {"remote": 0, "authorization": 0, "runner": 0}
    assert len(transport.calls) == 2
    assert manager.state(OLLAMA_MANAGER_MODEL_ID).value == "error"
    with pytest.raises(LocalModelUnloadError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="retry must fail closed",
        )


def test_ollama_successful_informational_response_uses_existing_pipeline(
    monkeypatch,
) -> None:
    settings = _settings()
    transport = FakeOllamaTransport(
        [
            _response({"model_info": {"general.architecture": "test"}}),
            _response({"done": True, "response": "Respuesta solo informativa."}),
            _response({"models": [{"name": "gemma3:latest"}]}),
            _response({"done": True, "response": ""}),
            _response({"models": []}),
        ]
    )
    manager = _manager(_backend(transport, settings), settings)
    app = create_application(settings, local_model_manager=manager)
    calls = {"authorization": 0, "runner": 0}

    def unexpected_authorization(self, proposal, context):
        del self, proposal, context
        calls["authorization"] += 1
        raise AssertionError("Informational response reached authorization.")

    def unexpected_runner(self, proposal, context):
        del self, proposal, context
        calls["runner"] += 1
        raise AssertionError("Informational response reached the runner.")

    monkeypatch.setattr(
        type(app.authorizer),
        "authorize",
        unexpected_authorization,
    )
    monkeypatch.setattr(
        type(app.tool_runner),
        "execute",
        unexpected_runner,
    )

    result = app.process("pregunta: ¿qué significa?")

    assert result.success is True
    assert result.authorized is False
    assert result.proposal is None
    assert result.tool_result is None
    assert result.informational_response == "Respuesta solo informativa."
    assert calls == {"authorization": 0, "runner": 0}
    assert len(transport.calls) == 5
