from __future__ import annotations

from dataclasses import dataclass
import json
from threading import Lock
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from jarvis.ai.contracts import InformationalResponse
from jarvis.ai.local_models import LocalModelTask
from jarvis.config.settings import Settings


_MAX_RESPONSE_BYTES = 1_048_576
OLLAMA_MANAGER_MODEL_ID = "ollama-local"


class OllamaBackendError(RuntimeError):
    pass


class OllamaModelUnavailableError(OllamaBackendError):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class OllamaHTTPResponse:
    status_code: int
    body: bytes

    def __post_init__(self) -> None:
        if type(self.status_code) is not int or not 100 <= self.status_code <= 599:
            raise ValueError("Ollama HTTP status is invalid.")
        if type(self.body) is not bytes:
            raise TypeError("Ollama HTTP response body must be bytes.")

    def __repr__(self) -> str:
        return "OllamaHTTPResponse(<redacted>)"


class OllamaHTTPTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        payload: dict[str, object] | None,
        timeout_seconds: float,
    ) -> OllamaHTTPResponse:
        ...


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        del request, file, code, message, headers, new_url
        return None


class _UrllibOllamaTransport:
    def request(
        self,
        method: str,
        url: str,
        payload: dict[str, object] | None,
        timeout_seconds: float,
    ) -> OllamaHTTPResponse:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
        try:
            response = opener.open(request, timeout=timeout_seconds)
        except HTTPError as error:
            error.close()
            return OllamaHTTPResponse(error.code, b"")
        except (OSError, TimeoutError, URLError, ValueError):
            raise OllamaBackendError(
                "Ollama service is unavailable."
            ) from None
        try:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
            return OllamaHTTPResponse(response.status, body)
        except (OSError, TimeoutError, ValueError):
            raise OllamaBackendError(
                "Ollama response could not be read."
            ) from None
        finally:
            response.close()


@dataclass(frozen=True, slots=True)
class _OllamaModelHandle:
    manager_model_id: str


class OllamaLocalModelBackend:
    """Informational-only Ollama backend for the existing local model manager."""

    __slots__ = (
        "_base_url",
        "_model_name",
        "_timeout_seconds",
        "_max_context_tokens",
        "_transport",
        "_uncertain_model_ids",
        "_uncertainty_lock",
    )

    def __init__(
        self,
        *,
        settings: Settings,
        transport: OllamaHTTPTransport | None = None,
    ) -> None:
        if type(settings) is not Settings:
            raise TypeError("Ollama backend requires validated settings.")
        settings = Settings.model_validate(settings.model_dump())
        if not settings.ai_enabled or settings.ai_provider != "local":
            raise ValueError("Ollama requires enabled local AI configuration.")
        if settings.local_ollama_model is None:
            raise ValueError("An Ollama model name must be configured.")
        if settings.local_max_context_tokens <= 0:
            raise ValueError("A positive local context limit must be configured.")
        selected_transport = (
            _UrllibOllamaTransport() if transport is None else transport
        )
        if not callable(getattr(selected_transport, "request", None)):
            raise TypeError("Ollama transport does not implement its contract.")
        self._base_url = settings.local_ollama_base_url
        self._model_name = settings.local_ollama_model
        self._timeout_seconds = settings.local_ollama_timeout_seconds
        self._max_context_tokens = settings.local_max_context_tokens
        self._transport = selected_transport
        self._uncertain_model_ids: set[str] = set()
        self._uncertainty_lock = Lock()

    def load(self, model_id: str) -> object:
        self._require_manager_model_id(model_id)
        response = self._request_json(
            "POST",
            "/api/show",
            {"model": self._model_name},
        )
        if not (
            type(response.get("model_info")) is dict
            or type(response.get("details")) is dict
            or any(
                type(response.get(key)) is str
                for key in ("modelfile", "parameters", "template")
            )
            or type(response.get("capabilities")) is list
        ):
            raise OllamaBackendError("Ollama model information is incomplete.")
        return _OllamaModelHandle(model_id)

    def failed_load_released(self, model_id: str) -> bool:
        return type(model_id) is str and model_id == OLLAMA_MANAGER_MODEL_ID

    def infer(
        self,
        handle: object,
        task: LocalModelTask,
        content: str,
        context_tokens: int,
        max_context_tokens: int,
    ) -> str:
        model_id = self._require_handle(handle)
        if type(task) is not LocalModelTask:
            raise TypeError("Ollama task is invalid.")
        if (
            type(content) is not str
            or not content.strip()
            or len(content) > 8000
            or "\x00" in content
        ):
            raise ValueError("Ollama input is invalid.")
        if (
            type(context_tokens) is not int
            or type(max_context_tokens) is not int
            or context_tokens <= 0
            or max_context_tokens <= 0
            or context_tokens > max_context_tokens
            or max_context_tokens > self._max_context_tokens
        ):
            raise OllamaBackendError("Ollama context limit was exceeded.")

        instructions = {
            LocalModelTask.ANSWER_QUESTION: (
                "Answer the input with informational text only. "
                "Do not claim authority or perform actions."
            ),
            LocalModelTask.SUMMARIZE_TEXT: (
                "Summarize the input with informational text only. "
                "Do not claim authority or perform actions."
            ),
        }
        payload: dict[str, object] = {
            "model": self._model_name,
            "prompt": content,
            "system": instructions[task],
            "stream": False,
            "keep_alive": -1,
            "options": {"num_ctx": max_context_tokens},
        }
        try:
            response = self._request_json("POST", "/api/generate", payload)
        except Exception:
            self._mark_uncertain(model_id)
            raise OllamaBackendError(
                "Ollama inference did not complete reliably."
            ) from None
        except BaseException:
            self._mark_uncertain(model_id)
            raise
        if response.get("done") is not True:
            self._mark_uncertain(model_id)
            raise OllamaBackendError(
                "Ollama inference did not complete reliably."
            )
        text = response.get("response")
        if type(text) is not str:
            raise OllamaBackendError("Ollama generation response is invalid.")
        try:
            InformationalResponse(text=text)
        except (TypeError, ValueError):
            raise OllamaBackendError(
                "Ollama generation response is invalid."
            ) from None
        return text

    def unload(self, handle: object) -> None:
        model_id = self._require_handle(handle)
        with self._uncertainty_lock:
            if model_id in self._uncertain_model_ids:
                raise OllamaBackendError(
                    "Ollama model resource state is uncertain."
                )
        try:
            running_models = self._running_model_names()
            if self._model_name not in running_models and (
                f"{self._model_name}:latest" not in running_models
            ):
                return
            response = self._request_json(
                "POST",
                "/api/generate",
                {
                    "model": self._model_name,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                },
            )
            self._generation_text(response, allow_empty=True)
            remaining_models = self._running_model_names()
            if self._model_name in remaining_models or (
                f"{self._model_name}:latest" in remaining_models
            ):
                raise OllamaBackendError(
                    "Ollama model remains loaded after unload."
                )
        except Exception:
            self._mark_uncertain(model_id)
            raise OllamaBackendError(
                "Ollama model unload could not be confirmed."
            ) from None
        except BaseException:
            self._mark_uncertain(model_id)
            raise

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        try:
            response = self._transport.request(
                method,
                self._base_url + path,
                payload,
                self._timeout_seconds,
            )
        except Exception:
            raise OllamaBackendError(
                "Ollama service is unavailable."
            ) from None
        if type(response) is not OllamaHTTPResponse:
            raise OllamaBackendError("Ollama response is invalid.")
        if len(response.body) > _MAX_RESPONSE_BYTES:
            raise OllamaBackendError("Ollama response exceeds the supported size.")
        if not 200 <= response.status_code < 300:
            if response.status_code == 404 and path != "/api/ps":
                raise OllamaModelUnavailableError(
                    "Configured Ollama model is unavailable."
                )
            raise OllamaBackendError("Ollama request failed.")
        try:
            decoded = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise OllamaBackendError("Ollama returned invalid JSON.") from None
        if type(decoded) is not dict or "error" in decoded:
            raise OllamaBackendError("Ollama returned an invalid response.")
        return decoded

    def _running_model_names(self) -> set[str]:
        response = self._request_json("GET", "/api/ps", None)
        models = response.get("models")
        if type(models) is not list:
            raise OllamaBackendError("Ollama model status is invalid.")
        names: set[str] = set()
        for item in models:
            if type(item) is not dict:
                raise OllamaBackendError("Ollama model status is invalid.")
            name = item.get("name", item.get("model"))
            if type(name) is not str or not name:
                raise OllamaBackendError("Ollama model status is invalid.")
            names.add(name)
        return names

    @staticmethod
    def _generation_text(
        response: dict[str, object],
        *,
        allow_empty: bool,
    ) -> str:
        if response.get("done") is not True or type(response.get("response")) is not str:
            raise OllamaBackendError("Ollama generation response is incomplete.")
        text = response["response"]
        if not allow_empty:
            try:
                InformationalResponse(text=text)
            except (TypeError, ValueError):
                raise OllamaBackendError(
                    "Ollama generation response is invalid."
                ) from None
        return text

    @staticmethod
    def _require_manager_model_id(model_id: str) -> None:
        if type(model_id) is not str or model_id != OLLAMA_MANAGER_MODEL_ID:
            raise OllamaModelUnavailableError(
                "Configured Ollama model is unavailable."
            )

    @staticmethod
    def _require_handle(handle: object) -> str:
        if (
            type(handle) is not _OllamaModelHandle
            or handle.manager_model_id != OLLAMA_MANAGER_MODEL_ID
        ):
            raise OllamaBackendError("Ollama model handle is invalid.")
        return handle.manager_model_id

    def _mark_uncertain(self, model_id: str) -> None:
        with self._uncertainty_lock:
            self._uncertain_model_ids.add(model_id)
