from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import re
from typing import Protocol
from urllib.error import HTTPError
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from jarvis.ai.contracts import InformationalResponse
from jarvis.ai.local_models import (
    ContextTokenCounter,
    LocalModelDescriptor,
    LocalModelBusyError,
    LocalModelError,
    LocalModelResourceError,
    LocalModelTask,
    RemoteOperationLedger,
    RemoteOperationOutcome,
    RemoteOperationType,
)
from jarvis.ai.model_catalog import (
    ModelCatalog,
    ModelCatalogError,
    ModelRuntime,
    RequirementBasis,
    ResolvedModel,
    ResourceKind,
)
from jarvis.config.settings import Settings


_MAX_RESPONSE_BYTES = 1_048_576
OLLAMA_MANAGER_MODEL_ID = "ollama-local"
_OLLAMA_MODEL_NAME_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z"
)


class OllamaOperationClassification(str, Enum):
    NOT_SENT = "not_sent"
    OUTCOME_UNKNOWN = "outcome_unknown"
    HTTP_RESPONSE_RECEIVED = "http_response_received"
    TERMINAL_RESPONSE_VALIDATED = "terminal_response_validated"
    RESPONSE_INVALID = "response_invalid"
    TERMINAL_RESPONSE_INVALID = "terminal_response_invalid"


class OllamaBackendError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        classification: OllamaOperationClassification | None = None,
    ) -> None:
        super().__init__(message)
        if classification is not None and (
            type(classification) is not OllamaOperationClassification
        ):
            raise TypeError("Ollama operation classification is invalid.")
        self.classification = classification


class OllamaModelUnavailableError(OllamaBackendError):
    pass


class _OllamaTransportFailure(OllamaBackendError):
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


class OllamaModelUseAuthorizer(Protocol):
    def authorize(
        self,
        model: ResolvedModel,
        task: LocalModelTask | None,
    ) -> bool:
        """Authorize model inspection or a specific informational inference."""
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
        try:
            data = None
            headers = {"Accept": "application/json"}
            if payload is not None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                headers["Content-Type"] = "application/json"
            request = Request(url, data=data, headers=headers, method=method)
            opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
        except Exception:
            raise _OllamaTransportFailure(
                "Ollama request could not be prepared.",
                classification=OllamaOperationClassification.NOT_SENT,
            ) from None
        try:
            response = opener.open(request, timeout=timeout_seconds)
        except HTTPError as error:
            error.close()
            return OllamaHTTPResponse(error.code, b"")
        except Exception:
            raise _OllamaTransportFailure(
                "Ollama request outcome is unknown.",
                classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
            ) from None
        try:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
            return OllamaHTTPResponse(response.status, body)
        except Exception:
            raise _OllamaTransportFailure(
                "Ollama response could not be validated.",
                classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
            ) from None
        finally:
            response.close()


@dataclass(frozen=True, slots=True)
class _OllamaModelHandle:
    manager_model_id: str
    resolved_model: ResolvedModel


class OllamaLocalModelBackend:
    """Informational-only Ollama backend for the existing local model manager."""

    __slots__ = (
        "_base_url",
        "_model_name",
        "_timeout_seconds",
        "_max_context_tokens",
        "_transport",
        "_ledger",
        "_model_catalog",
        "_catalog_entry_id",
        "_model_use_authorizer",
        "_token_counter",
    )

    def __init__(
        self,
        *,
        settings: Settings,
        transport: OllamaHTTPTransport | None = None,
        ledger: RemoteOperationLedger | None = None,
        model_catalog: ModelCatalog | None = None,
        catalog_entry_id: str | None = None,
        model_use_authorizer: OllamaModelUseAuthorizer | None = None,
        token_counter: ContextTokenCounter | None = None,
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
        if ledger is None:
            local_app_data = os.environ.get("LOCALAPPDATA")
            if (
                type(local_app_data) is not str
                or not local_app_data
                or not Path(local_app_data).is_absolute()
            ):
                raise OllamaBackendError(
                    "Persistent Ollama operation state is unavailable.",
                    classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
                )
            try:
                selected_ledger = RemoteOperationLedger(
                    max_concurrent_operations=settings.local_max_concurrent_inferences,
                    sqlite_path=(
                        Path(local_app_data)
                        / "JARVIS"
                        / "remote-operations.sqlite3"
                    ),
                )
            except LocalModelError:
                raise OllamaBackendError(
                    "Persistent Ollama operation state is unavailable.",
                    classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
                ) from None
        else:
            if type(ledger) is not RemoteOperationLedger:
                raise TypeError("Ollama operation ledger is invalid.")
            if not ledger.is_persistent:
                raise ValueError("Ollama operation ledger must be persistent.")
            if (
                ledger.max_concurrent_operations
                != settings.local_max_concurrent_inferences
            ):
                raise ValueError(
                    "Ollama operation ledger limit does not match settings."
                )
            selected_ledger = ledger
        self._base_url = settings.local_ollama_base_url
        self._model_name = settings.local_ollama_model
        self._timeout_seconds = settings.local_ollama_timeout_seconds
        self._max_context_tokens = settings.local_max_context_tokens
        self._transport = selected_transport
        self._ledger = selected_ledger
        if model_catalog is not None and type(model_catalog) is not ModelCatalog:
            raise TypeError("Ollama model catalog is invalid.")
        if catalog_entry_id is not None and (
            type(catalog_entry_id) is not str or not catalog_entry_id
        ):
            raise TypeError("Ollama catalog entry identifier is invalid.")
        if model_use_authorizer is not None and not callable(
            getattr(model_use_authorizer, "authorize", None)
        ):
            raise TypeError("Ollama model-use authorizer is invalid.")
        if token_counter is not None and not callable(
            getattr(token_counter, "count_tokens", None)
        ):
            raise TypeError("Ollama context token counter is invalid.")
        self._model_catalog = model_catalog
        self._catalog_entry_id = catalog_entry_id
        self._model_use_authorizer = model_use_authorizer
        self._token_counter = token_counter

    def load(self, model_id: str) -> object:
        self._require_manager_model_id(model_id)
        self._require_strict_dependencies()
        self._require_authenticated_artifact_identity()
        operation_id = self._reserve_operation(RemoteOperationType.INSPECT)
        self._mark_dispatch_attempted(operation_id)
        try:
            runtime_version = self._runtime_version()
            resolved_model = self._resolve_model(
                runtime_version,
                requested_context_tokens=self._max_context_tokens,
                task=None,
            )
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
                raise OllamaBackendError(
                    "Ollama model information is incomplete.",
                    classification=OllamaOperationClassification.RESPONSE_INVALID,
                )
            self._complete_operation(operation_id, content_valid=True)
            return _OllamaModelHandle(model_id, resolved_model)
        except OllamaBackendError as error:
            self._finalize_failed_operation(operation_id, error)
            raise
        except Exception:
            error = OllamaBackendError(
                "Ollama model inspection did not complete reliably.",
                classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
            )
            self._finalize_failed_operation(operation_id, error)
            raise error from None

    def list_installed_models(self) -> tuple[str, ...]:
        """Return validated installed model names without changing model state."""
        try:
            response = self._request_json("GET", "/api/tags", None)
        except OllamaBackendError as error:
            raise OllamaBackendError(
                "Installed Ollama model catalog could not be retrieved.",
                classification=error.classification,
            ) from None

        models = response.get("models")
        if type(models) is not list:
            raise OllamaBackendError(
                "Ollama installed model catalog is invalid.",
                classification=OllamaOperationClassification.RESPONSE_INVALID,
            )

        names: list[str] = []
        seen: set[str] = set()
        for model in models:
            if type(model) is not dict:
                raise OllamaBackendError(
                    "Ollama installed model catalog is invalid.",
                    classification=OllamaOperationClassification.RESPONSE_INVALID,
                )
            name = model.get("name")
            if (
                type(name) is not str
                or _OLLAMA_MODEL_NAME_PATTERN.fullmatch(name) is None
                or name in seen
            ):
                raise OllamaBackendError(
                    "Ollama installed model catalog is invalid.",
                    classification=OllamaOperationClassification.RESPONSE_INVALID,
                )
            names.append(name)
            seen.add(name)
        return tuple(names)

    @property
    def catalog_entry_id(self) -> str | None:
        return self._catalog_entry_id

    def validate_descriptor(self, descriptor: LocalModelDescriptor) -> None:
        if type(descriptor) is not LocalModelDescriptor:
            raise TypeError("Ollama requires a validated local model descriptor.")
        catalog = self._model_catalog
        entry_id = self._catalog_entry_id
        token_counter = self._token_counter
        if (
            catalog is None
            or entry_id is None
            or not catalog.has_identity_verifier
            or self._model_use_authorizer is None
            or token_counter is None
        ):
            raise OllamaBackendError(
                "Strict local model dependencies are unavailable.",
                classification=OllamaOperationClassification.NOT_SENT,
            )
        try:
            entry = catalog.inspect(entry_id)
        except ModelCatalogError:
            raise OllamaBackendError(
                "Configured local model is not available in the catalog.",
                classification=OllamaOperationClassification.NOT_SENT,
            ) from None
        requirements = {requirement.resource: requirement for requirement in entry.requirements}
        ram = requirements.get(ResourceKind.RAM)
        vram = requirements.get(ResourceKind.VRAM)
        if (
            descriptor.model_id != OLLAMA_MANAGER_MODEL_ID
            or descriptor.catalog_entry_id != entry_id
            or descriptor.tokenizer_model_id != self._model_name
            or descriptor.tokenizer_id != entry.tokenizer_id
            or getattr(token_counter, "tokenizer_id", None) != entry.tokenizer_id
            or descriptor.max_context_tokens > (entry.max_context_tokens or 0)
            or entry.identity.runtime is not ModelRuntime.OLLAMA
            or entry.identity.artifact_id != self._model_name
            or not entry.is_strictly_eligible()
            or ResourceKind.VRAM not in entry.required_resources
            or ram is None
            or vram is None
            or ram.basis is not RequirementBasis.CONSERVATIVE_BOUND
            or vram.basis is not RequirementBasis.CONSERVATIVE_BOUND
            or descriptor.requirements.ram_bytes != ram.bytes_required
            or descriptor.requirements.vram_bytes != vram.bytes_required
        ):
            raise OllamaBackendError(
                "Local model descriptor does not match strict catalog policy.",
                classification=OllamaOperationClassification.NOT_SENT,
            )

    def failed_load_released(self, model_id: str) -> bool:
        """Confirm Ollama no longer reports the selected model as running.

        This is runtime residency evidence only; Ollama's process list does not
        prove that physical RAM or VRAM has been returned to the operating system.
        """
        if type(model_id) is not str or model_id != OLLAMA_MANAGER_MODEL_ID:
            return False
        operation_id: str | None = None
        try:
            operation_id = self._reserve_operation(RemoteOperationType.INSPECT)
            self._mark_dispatch_attempted(operation_id)
            running_models = self._running_model_names()
            self._complete_operation(operation_id, content_valid=True)
            return not self._configured_model_is_running(running_models)
        except OllamaBackendError as error:
            if operation_id is not None:
                try:
                    self._finalize_failed_operation(operation_id, error)
                except OllamaBackendError:
                    pass
            return False
        except Exception:
            if operation_id is not None:
                sanitized = OllamaBackendError(
                    "Ollama model cleanup could not be confirmed.",
                    classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
                )
                try:
                    self._finalize_failed_operation(operation_id, sanitized)
                except OllamaBackendError:
                    pass
            return False

    def infer(
        self,
        handle: object,
        task: LocalModelTask,
        content: str,
        context_tokens: int,
        max_context_tokens: int,
    ) -> str:
        model_handle = self._require_handle(handle)
        self._require_strict_dependencies()
        self._require_authenticated_artifact_identity()
        model_id = model_handle.manager_model_id
        if type(task) is not LocalModelTask:
            raise OllamaBackendError(
                "Ollama task is invalid.",
                classification=OllamaOperationClassification.NOT_SENT,
            )
        if (
            type(content) is not str
            or not content.strip()
            or len(content) > 8000
            or "\x00" in content
        ):
            raise OllamaBackendError(
                "Ollama input is invalid.",
                classification=OllamaOperationClassification.NOT_SENT,
            )
        if (
            type(context_tokens) is not int
            or type(max_context_tokens) is not int
            or context_tokens <= 0
            or max_context_tokens <= 0
            or context_tokens > max_context_tokens
            or max_context_tokens > self._max_context_tokens
        ):
            raise OllamaBackendError(
                "Ollama context limit was exceeded.",
                classification=OllamaOperationClassification.NOT_SENT,
            )
        token_counter = self._token_counter
        if token_counter is None:
            raise OllamaBackendError(
                "A compatible model token counter is unavailable.",
                classification=OllamaOperationClassification.NOT_SENT,
            )
        try:
            counted_tokens = token_counter.count_tokens(
                self._model_name,
                task,
                content,
            )
        except Exception:
            raise OllamaBackendError(
                "Input context could not be determined.",
                classification=OllamaOperationClassification.NOT_SENT,
            ) from None
        if (
            type(counted_tokens) is not int
            or counted_tokens <= 0
            or counted_tokens > context_tokens
        ):
            raise OllamaBackendError(
                "Input context was undercounted.",
                classification=OllamaOperationClassification.NOT_SENT,
            )

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
        operation_id = self._reserve_operation(RemoteOperationType.INFER)
        self._mark_dispatch_attempted(operation_id)
        try:
            current_model = self._resolve_model(
                self._runtime_version(),
                requested_context_tokens=max_context_tokens,
                task=task,
            )
            if (
                current_model.entry.entry_id != model_handle.resolved_model.entry.entry_id
                or current_model.verified_identity
                != model_handle.resolved_model.verified_identity
            ):
                raise OllamaBackendError(
                    "Configured Ollama model identity changed.",
                    classification=OllamaOperationClassification.RESPONSE_INVALID,
                )
            response = self._request_json("POST", "/api/generate", payload)
            if response.get("done") is not True:
                raise OllamaBackendError(
                    "Ollama inference did not complete reliably.",
                    classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
                )
            _, text = self._generation_text(response, allow_empty=False)
            self._complete_operation(operation_id, content_valid=True)
            return text
        except OllamaBackendError as error:
            self._finalize_failed_operation(operation_id, error)
            raise OllamaBackendError(
                "Ollama inference did not complete reliably.",
                classification=error.classification
                or OllamaOperationClassification.OUTCOME_UNKNOWN,
            ) from None
        except Exception:
            error = OllamaBackendError(
                "Ollama inference did not complete reliably.",
                classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
            )
            self._finalize_failed_operation(operation_id, error)
            raise error from None

    def unload(self, handle: object) -> None:
        self._require_handle(handle)
        operation_id = self._reserve_operation(RemoteOperationType.UNLOAD)
        self._mark_dispatch_attempted(operation_id)
        completed_requests = 0
        try:
            running_models = self._running_model_names()
            completed_requests += 1
            if not self._configured_model_is_running(running_models):
                self._complete_operation(operation_id, content_valid=True)
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
            completed_requests += 1
            remaining_models = self._running_model_names()
            completed_requests += 1
            if self._configured_model_is_running(remaining_models):
                raise OllamaBackendError(
                    "Ollama model remains loaded after unload.",
                    classification=OllamaOperationClassification.TERMINAL_RESPONSE_INVALID,
                )
            self._complete_operation(operation_id, content_valid=True)
        except OllamaBackendError as error:
            self._finalize_failed_operation(
                operation_id,
                error,
                completed_requests=completed_requests,
            )
            raise OllamaBackendError(
                "Ollama model unload could not be confirmed.",
                classification=error.classification
                or OllamaOperationClassification.OUTCOME_UNKNOWN,
            ) from None
        except Exception as error:
            sanitized = OllamaBackendError(
                "Ollama model unload could not be confirmed.",
                classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
            )
            self._finalize_failed_operation(
                operation_id,
                sanitized,
                completed_requests=completed_requests,
            )
            raise sanitized from None

    def _reserve_operation(
        self,
        operation_type: RemoteOperationType,
    ) -> str:
        try:
            record = self._ledger.reserve(
                model_id=OLLAMA_MANAGER_MODEL_ID,
                operation_type=operation_type,
            )
            return record.operation_id
        except LocalModelBusyError:
            raise OllamaBackendError(
                "Ollama operation capacity is unavailable.",
                classification=OllamaOperationClassification.NOT_SENT,
            ) from None
        except LocalModelResourceError:
            raise OllamaBackendError(
                "Ollama operation state is uncertain.",
                classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
            ) from None
        except LocalModelError:
            raise OllamaBackendError(
                "Ollama operation could not be admitted.",
                classification=OllamaOperationClassification.NOT_SENT,
            ) from None

    def _mark_dispatch_attempted(self, operation_id: str) -> None:
        try:
            self._ledger.mark_dispatch_attempted(operation_id)
        except LocalModelError:
            raise OllamaBackendError(
                "Ollama operation state could not be saved.",
                classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
            ) from None

    def _complete_operation(self, operation_id: str, *, content_valid: bool) -> None:
        try:
            self._ledger.complete(
                operation_id,
                content_valid=content_valid,
            )
        except LocalModelError:
            raise OllamaBackendError(
                "Ollama operation result could not be saved.",
                classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
            ) from None

    def _finalize_failed_operation(
        self,
        operation_id: str,
        error: OllamaBackendError,
        *,
        completed_requests: int = 0,
    ) -> None:
        try:
            record = self._ledger.get(operation_id)
            if record.outcome is not RemoteOperationOutcome.PENDING:
                return
            classification = (
                error.classification
                or OllamaOperationClassification.OUTCOME_UNKNOWN
            )
            if (
                classification is OllamaOperationClassification.NOT_SENT
                and completed_requests == 0
            ):
                self._ledger.mark_not_sent_after_dispatch(operation_id)
            elif (
                classification
                is OllamaOperationClassification.TERMINAL_RESPONSE_INVALID
                and record.operation_type is not RemoteOperationType.UNLOAD
            ):
                self._ledger.complete(operation_id, content_valid=False)
            else:
                self._ledger.mark_uncertain(operation_id)
        except LocalModelError:
            raise OllamaBackendError(
                "Ollama operation state could not be saved.",
                classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
            ) from None

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        try:
            url = self._base_url + path
        except Exception:
            raise OllamaBackendError(
                "Ollama request could not be prepared.",
                classification=OllamaOperationClassification.NOT_SENT,
            ) from None
        try:
            response = self._transport.request(
                method,
                url,
                payload,
                self._timeout_seconds,
            )
        except _OllamaTransportFailure:
            raise
        except Exception:
            raise _OllamaTransportFailure(
                "Ollama request outcome is unknown.",
                classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
            ) from None
        if type(response) is not OllamaHTTPResponse:
            raise OllamaBackendError(
                "Ollama response is invalid.",
                classification=OllamaOperationClassification.RESPONSE_INVALID,
            )
        if len(response.body) > _MAX_RESPONSE_BYTES:
            raise OllamaBackendError(
                "Ollama response exceeds the supported size.",
                classification=OllamaOperationClassification.RESPONSE_INVALID,
            )
        if not 200 <= response.status_code < 300:
            if response.status_code == 404 and path not in {
                "/api/ps",
                "/api/tags",
            }:
                raise OllamaModelUnavailableError(
                    "Configured Ollama model is unavailable.",
                    classification=OllamaOperationClassification.HTTP_RESPONSE_RECEIVED,
                )
            raise OllamaBackendError(
                "Ollama request failed.",
                classification=OllamaOperationClassification.HTTP_RESPONSE_RECEIVED,
            )
        try:
            decoded = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise OllamaBackendError(
                "Ollama returned invalid JSON.",
                classification=OllamaOperationClassification.RESPONSE_INVALID,
            ) from None
        if type(decoded) is not dict or "error" in decoded:
            raise OllamaBackendError(
                "Ollama returned an invalid response.",
                classification=OllamaOperationClassification.RESPONSE_INVALID,
            )
        return decoded

    def _running_model_names(self) -> set[str]:
        response = self._request_json("GET", "/api/ps", None)
        models = response.get("models")
        if type(models) is not list:
            raise OllamaBackendError(
                "Ollama model status is invalid.",
                classification=OllamaOperationClassification.RESPONSE_INVALID,
            )
        names: set[str] = set()
        for item in models:
            if type(item) is not dict:
                raise OllamaBackendError(
                    "Ollama model status is invalid.",
                    classification=OllamaOperationClassification.RESPONSE_INVALID,
                )
            name = item.get("name", item.get("model"))
            if (
                type(name) is not str
                or _OLLAMA_MODEL_NAME_PATTERN.fullmatch(name) is None
            ):
                raise OllamaBackendError(
                    "Ollama model status is invalid.",
                    classification=OllamaOperationClassification.RESPONSE_INVALID,
                )
            names.add(name)
        return names

    def _configured_model_is_running(self, running_names: set[str]) -> bool:
        configured_names = self._model_name_aliases(self._model_name)
        return any(
            configured_names.intersection(self._model_name_aliases(name))
            for name in running_names
        )

    @staticmethod
    def _model_name_aliases(name: str) -> frozenset[str]:
        aliases = {name}
        final_component = name.rsplit("/", 1)[-1]
        if ":" not in final_component:
            aliases.add(f"{name}:latest")
        return frozenset(aliases)

    def _runtime_version(self) -> str:
        response = self._request_json("GET", "/api/version", None)
        version = response.get("version")
        if type(version) is not str or not version:
            raise OllamaBackendError(
                "Ollama runtime version is unavailable.",
                classification=OllamaOperationClassification.RESPONSE_INVALID,
            )
        return version

    def _resolve_model(
        self,
        runtime_version: str,
        *,
        requested_context_tokens: int,
        task: LocalModelTask | None,
    ) -> ResolvedModel:
        catalog = self._model_catalog
        entry_id = self._catalog_entry_id
        authorizer = self._model_use_authorizer
        if (
            catalog is None
            or entry_id is None
            or not catalog.has_identity_verifier
            or authorizer is None
            or self._token_counter is None
        ):
            raise OllamaBackendError(
                "Strict local model verification and authorization are unavailable.",
                classification=OllamaOperationClassification.NOT_SENT,
            )
        try:
            declared_entry = catalog.inspect(entry_id)
            if (
                declared_entry.identity.runtime is not ModelRuntime.OLLAMA
                or declared_entry.identity.artifact_id != self._model_name
                or not {
                    ResourceKind.RAM,
                    ResourceKind.VRAM,
                }.issubset(declared_entry.required_resources)
                or not declared_entry.is_strictly_eligible()
                or any(
                    requirement.basis is not RequirementBasis.CONSERVATIVE_BOUND
                    for requirement in declared_entry.requirements
                    if requirement.resource in {
                        ResourceKind.RAM,
                        ResourceKind.VRAM,
                    }
                )
                or getattr(self._token_counter, "tokenizer_id", None)
                != declared_entry.tokenizer_id
            ):
                raise ModelCatalogError("Catalog policy is incomplete.")
            resolved = catalog.resolve(
                entry_id,
                runtime_version=runtime_version,
                requested_context_tokens=requested_context_tokens,
            )
        except (ModelCatalogError, TypeError, ValueError):
            raise OllamaBackendError(
                "Configured local model is not strictly verified.",
                classification=OllamaOperationClassification.RESPONSE_INVALID,
            ) from None
        try:
            authorized = authorizer.authorize(resolved, task) is True
        except Exception:
            authorized = False
        if not authorized:
            raise OllamaBackendError(
                "Local model use is not authorized.",
                classification=OllamaOperationClassification.RESPONSE_INVALID,
            )
        return resolved

    def _require_strict_dependencies(self) -> None:
        catalog = self._model_catalog
        if (
            catalog is None
            or not catalog.has_identity_verifier
            or self._catalog_entry_id is None
            or self._model_use_authorizer is None
            or self._token_counter is None
        ):
            raise OllamaBackendError(
                "Strict local model verification and authorization are unavailable.",
                classification=OllamaOperationClassification.NOT_SENT,
            )

    def _require_authenticated_artifact_identity(self) -> None:
        catalog = self._model_catalog
        if (
            catalog is None
            or not catalog.has_authenticated_identity_verifier
        ):
            raise OllamaBackendError(
                "Production artifact identity verification is unavailable.",
                classification=OllamaOperationClassification.NOT_SENT,
            )

    @staticmethod
    def _generation_text(
        response: dict[str, object],
        *,
        allow_empty: bool,
    ) -> tuple[OllamaOperationClassification, str]:
        if response.get("done") is not True:
            raise OllamaBackendError(
                "Ollama generation response is incomplete.",
                classification=OllamaOperationClassification.OUTCOME_UNKNOWN,
            )
        if type(response.get("response")) is not str:
            raise OllamaBackendError(
                "Ollama generation response is incomplete.",
                classification=OllamaOperationClassification.TERMINAL_RESPONSE_INVALID,
            )
        text = response["response"]
        if not allow_empty:
            try:
                InformationalResponse(text=text)
            except (TypeError, ValueError):
                raise OllamaBackendError(
                    "Ollama generation response is invalid.",
                    classification=OllamaOperationClassification.TERMINAL_RESPONSE_INVALID,
                ) from None
        return OllamaOperationClassification.TERMINAL_RESPONSE_VALIDATED, text

    @staticmethod
    def _require_manager_model_id(model_id: str) -> None:
        if type(model_id) is not str or model_id != OLLAMA_MANAGER_MODEL_ID:
            raise OllamaModelUnavailableError(
                "Configured Ollama model is unavailable.",
                classification=OllamaOperationClassification.NOT_SENT,
            )

    @staticmethod
    def _require_handle(handle: object) -> _OllamaModelHandle:
        if (
            type(handle) is not _OllamaModelHandle
            or handle.manager_model_id != OLLAMA_MANAGER_MODEL_ID
            or type(handle.resolved_model) is not ResolvedModel
        ):
            raise OllamaBackendError(
                "Ollama model handle is invalid.",
                classification=OllamaOperationClassification.NOT_SENT,
            )
        return handle
