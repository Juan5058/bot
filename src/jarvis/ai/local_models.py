from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from threading import Lock
import time
from typing import Protocol

from jarvis.ai.contracts import (
    AIProviderContract,
    ActionSuggestion,
    InformationalRequest,
    InformationalResponse,
)
from jarvis.config.settings import Settings


_BYTES_PER_MIB = 1024 * 1024
_MAX_RESOURCE_SNAPSHOT_AGE_SECONDS = 5.0


def _resource_snapshot_is_fresh(snapshot: ResourceSnapshot) -> bool:
    age = time.monotonic() - snapshot.captured_at_monotonic
    return 0 <= age <= _MAX_RESOURCE_SNAPSHOT_AGE_SECONDS


class LocalModelTask(str, Enum):
    ANSWER_QUESTION = "answer_question"
    SUMMARIZE_TEXT = "summarize_text"


class ResourceSource(str, Enum):
    UNKNOWN = "unknown"
    SIMULATED = "simulated"
    OPERATING_SYSTEM = "operating_system"
    INFERENCE_BACKEND = "inference_backend"


class ModelAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ModelLifecycleState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    UNLOADING = "unloading"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class LocalModelError(RuntimeError):
    pass


class LocalModelUnavailableError(LocalModelError):
    pass


class LocalModelResourceError(LocalModelError):
    pass


class LocalModelBusyError(LocalModelError):
    pass


class LocalModelLoadError(LocalModelError):
    pass


class LocalModelInferenceError(LocalModelError):
    pass


class LocalModelUnloadError(LocalModelError):
    pass


@dataclass(frozen=True, slots=True)
class ModelRequirements:
    ram_bytes: int
    vram_bytes: int | None = None

    def __post_init__(self) -> None:
        if type(self.ram_bytes) is not int or self.ram_bytes <= 0:
            raise ValueError("Model RAM estimate must be a positive integer.")
        if self.vram_bytes is not None and (
            type(self.vram_bytes) is not int or self.vram_bytes <= 0
        ):
            raise ValueError("Model VRAM estimate must be a positive integer.")


@dataclass(frozen=True, slots=True)
class LocalModelDescriptor:
    model_id: str
    tasks: frozenset[LocalModelTask]
    max_context_tokens: int
    requirements: ModelRequirements
    availability: ModelAvailability = ModelAvailability.AVAILABLE

    def __post_init__(self) -> None:
        if (
            type(self.model_id) is not str
            or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", self.model_id)
            is None
        ):
            raise ValueError("Model identifier is invalid.")
        if (
            type(self.tasks) is not frozenset
            or not self.tasks
            or any(type(task) is not LocalModelTask for task in self.tasks)
        ):
            raise ValueError("Model task capabilities are invalid.")
        if (
            type(self.max_context_tokens) is not int
            or self.max_context_tokens <= 0
        ):
            raise ValueError("Model context limit must be positive.")
        if type(self.requirements) is not ModelRequirements:
            raise TypeError("Model resource requirements are invalid.")
        if type(self.availability) is not ModelAvailability:
            raise TypeError("Model availability is invalid.")


@dataclass(frozen=True, slots=True, repr=False)
class ResourceSnapshot:
    source: ResourceSource
    available_ram_bytes: int | None
    available_vram_bytes: int | None = None
    system_load: float | None = None
    captured_at_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if type(self.source) is not ResourceSource:
            raise TypeError("Resource measurement source is invalid.")
        for value in (self.available_ram_bytes, self.available_vram_bytes):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Available resource amounts must be non-negative.")
        if self.system_load is not None and (
            type(self.system_load) not in {int, float}
            or not math.isfinite(self.system_load)
            or not 0.0 <= self.system_load <= 1.0
        ):
            raise ValueError("System load must be a finite value from 0 to 1.")
        if (
            type(self.captured_at_monotonic) not in {int, float}
            or not math.isfinite(self.captured_at_monotonic)
            or self.captured_at_monotonic < 0
        ):
            raise ValueError("Resource measurement timestamp is invalid.")
        if self.source is ResourceSource.UNKNOWN and any(
            value is not None
            for value in (
                self.available_ram_bytes,
                self.available_vram_bytes,
                self.system_load,
            )
        ):
            raise ValueError("Unknown resource measurements cannot contain values.")

    def __repr__(self) -> str:
        return f"ResourceSnapshot(source={self.source.value!r}, <redacted>)"


@dataclass(frozen=True, slots=True)
class LocalModelLimits:
    max_context_tokens: int
    ram_budget_bytes: int | None
    vram_budget_bytes: int | None = None
    max_system_load: float | None = None
    max_concurrent_loads: int = 1
    max_concurrent_inferences: int = 1
    unload_after_inference: bool = True

    def __post_init__(self) -> None:
        if type(self.max_context_tokens) is not int or self.max_context_tokens < 0:
            raise ValueError("Configured context limit must be non-negative.")
        for value in (self.ram_budget_bytes, self.vram_budget_bytes):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("Configured memory budgets must be positive.")
        if self.max_system_load is not None and (
            type(self.max_system_load) not in {int, float}
            or not math.isfinite(self.max_system_load)
            or not 0.0 <= self.max_system_load <= 1.0
        ):
            raise ValueError("Configured system load limit is invalid.")
        for value in (
            self.max_concurrent_loads,
            self.max_concurrent_inferences,
        ):
            if type(value) is not int or not 1 <= value <= 16:
                raise ValueError("Configured concurrency must be from 1 to 16.")
        if type(self.unload_after_inference) is not bool:
            raise TypeError("Unload policy must be a boolean.")

    @classmethod
    def from_settings(cls, settings: Settings) -> LocalModelLimits:
        if type(settings) is not Settings:
            raise TypeError("Local model limits require validated settings.")
        return cls(
            max_context_tokens=settings.local_max_context_tokens,
            ram_budget_bytes=(
                None
                if settings.local_ram_budget_mb is None
                else settings.local_ram_budget_mb * _BYTES_PER_MIB
            ),
            vram_budget_bytes=(
                None
                if settings.local_vram_budget_mb is None
                else settings.local_vram_budget_mb * _BYTES_PER_MIB
            ),
            max_system_load=settings.local_max_system_load,
            max_concurrent_loads=settings.local_max_concurrent_loads,
            max_concurrent_inferences=settings.local_max_concurrent_inferences,
            unload_after_inference=settings.local_unload_after_inference,
        )


class ResourceProvider(Protocol):
    def snapshot(self) -> ResourceSnapshot:
        ...


class ContextTokenCounter(Protocol):
    def count_tokens(
        self,
        model_id: str,
        task: LocalModelTask,
        content: str,
    ) -> int:
        ...


class LocalModelBackend(Protocol):
    def load(self, model_id: str) -> object:
        ...

    def failed_load_released(self, model_id: str) -> bool:
        """Confirm a failed load left no model resources allocated."""
        ...

    def infer(
        self,
        handle: object,
        task: LocalModelTask,
        content: str,
        context_tokens: int,
        max_context_tokens: int,
    ) -> str:
        ...

    def unload(self, handle: object) -> None:
        ...


@dataclass(slots=True)
class _ModelEntry:
    descriptor: LocalModelDescriptor
    state: ModelLifecycleState
    handle: object | None = None
    active: bool = False
    resource_reservation_held: bool = False


class LocalModelManager:
    """Synchronous lifecycle/admission coordinator; it does not implement inference."""

    def __init__(
        self,
        *,
        backend: LocalModelBackend,
        resource_provider: ResourceProvider,
        token_counter: ContextTokenCounter,
        limits: LocalModelLimits,
        allow_simulated_resources_for_tests: bool = False,
    ) -> None:
        if not callable(getattr(backend, "load", None)) or not callable(
            getattr(backend, "infer", None)
        ) or not callable(getattr(backend, "unload", None)) or not callable(
            getattr(backend, "failed_load_released", None)
        ):
            raise TypeError("Local model backend does not implement its contract.")
        if not callable(getattr(resource_provider, "snapshot", None)):
            raise TypeError("Resource provider does not implement its contract.")
        if not callable(getattr(token_counter, "count_tokens", None)):
            raise TypeError("Context token counter does not implement its contract.")
        if type(limits) is not LocalModelLimits:
            raise TypeError("Local model manager requires validated limits.")
        if type(allow_simulated_resources_for_tests) is not bool:
            raise TypeError("Simulated resource test mode must be a boolean.")
        self._backend = backend
        self._resource_provider = resource_provider
        self._token_counter = token_counter
        self._limits = limits
        self._allow_simulated_resources_for_tests = (
            allow_simulated_resources_for_tests
        )
        self._entries: dict[str, _ModelEntry] = {}
        self._lock = Lock()
        self._active_loads = 0
        self._active_inferences = 0

    @property
    def limits(self) -> LocalModelLimits:
        return self._limits

    @property
    def allows_simulated_resources_for_tests(self) -> bool:
        return self._allow_simulated_resources_for_tests

    def register(self, descriptor: LocalModelDescriptor) -> None:
        if type(descriptor) is not LocalModelDescriptor:
            raise TypeError("Only validated local model descriptors can be registered.")
        with self._lock:
            if descriptor.model_id in self._entries:
                raise ValueError("Model identifier is already registered.")
            state = (
                ModelLifecycleState.UNLOADED
                if descriptor.availability is ModelAvailability.AVAILABLE
                else ModelLifecycleState.UNAVAILABLE
            )
            self._entries[descriptor.model_id] = _ModelEntry(descriptor, state)

    def state(self, model_id: str) -> ModelLifecycleState:
        if type(model_id) is not str:
            raise TypeError("Model identifier must be text.")
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                raise LocalModelUnavailableError("Local model is unavailable.")
            return entry.state

    def run(
        self,
        *,
        task: LocalModelTask,
        content: str,
    ) -> str:
        if type(task) is not LocalModelTask:
            raise TypeError("Local model task is invalid.")
        if type(content) is not str or not content.strip():
            raise ValueError("Local model input must be non-empty text.")
        if self._limits.max_context_tokens == 0:
            raise LocalModelResourceError("Configured context limit was exceeded.")

        entry, context_tokens = self._select_entry(task, content)
        if entry.state is ModelLifecycleState.ERROR and entry.handle is not None:
            self.unload(entry.descriptor.model_id)
        elif entry.state is ModelLifecycleState.ERROR:
            with self._lock:
                reservation_held = entry.resource_reservation_held
            if reservation_held:
                self._confirm_failed_load_cleanup(entry)
            else:
                with self._lock:
                    if (
                        entry.state is ModelLifecycleState.ERROR
                        and entry.handle is None
                    ):
                        entry.state = ModelLifecycleState.UNLOADED

        needs_load = entry.state is ModelLifecycleState.UNLOADED
        snapshot = self._get_snapshot(entry.descriptor) if needs_load else None
        self._reserve(entry, snapshot, needs_load)

        if needs_load:
            if snapshot is None or not _resource_snapshot_is_fresh(snapshot):
                self._release_unstarted_load(entry)
                raise LocalModelResourceError(
                    "Local model resource measurements are stale."
                )
            try:
                handle = self._backend.load(entry.descriptor.model_id)
                if handle is None:
                    raise ValueError("Backend returned an empty model handle.")
            except BaseException as error:
                cleanup_confirmed = False
                try:
                    cleanup_confirmed = (
                        self._backend.failed_load_released(
                            entry.descriptor.model_id
                        )
                        is True
                    )
                except BaseException:
                    cleanup_confirmed = False
                self._finish_load_failure(entry, cleanup_confirmed)
                if isinstance(error, Exception):
                    raise LocalModelLoadError(
                        "Local model could not be loaded."
                    ) from None
                raise
            with self._lock:
                entry.handle = handle
                entry.state = ModelLifecycleState.BUSY
                self._active_loads -= 1

        try:
            result = self._backend.infer(
                entry.handle,
                task,
                content,
                context_tokens,
                min(
                    self._limits.max_context_tokens,
                    entry.descriptor.max_context_tokens,
                ),
            )
            result = InformationalResponse(text=result).text
        except BaseException as error:
            self._finish_inference_failure(entry)
            if isinstance(error, Exception):
                raise LocalModelInferenceError(
                    "Local model inference failed."
                ) from None
            raise

        if self._limits.unload_after_inference:
            self._begin_unload_after_run(entry)
            self._unload_reserved(entry)
        else:
            with self._lock:
                entry.active = False
                entry.state = ModelLifecycleState.READY
                self._active_inferences -= 1
        return result

    def unload(self, model_id: str) -> None:
        if type(model_id) is not str:
            raise TypeError("Model identifier must be text.")
        confirm_failed_load_cleanup = False
        unload_handle = False
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                raise LocalModelUnavailableError("Local model is unavailable.")
            if entry.active or entry.state in {
                ModelLifecycleState.LOADING,
                ModelLifecycleState.BUSY,
                ModelLifecycleState.UNLOADING,
            }:
                raise LocalModelBusyError("Local model is currently in use.")
            if entry.handle is None:
                if entry.resource_reservation_held:
                    confirm_failed_load_cleanup = True
                elif entry.state is not ModelLifecycleState.UNAVAILABLE:
                    if entry.handle is None:
                        entry.state = ModelLifecycleState.UNLOADED
            else:
                entry.state = ModelLifecycleState.UNLOADING
                unload_handle = True
        if confirm_failed_load_cleanup:
            self._confirm_failed_load_cleanup(entry)
            return
        if unload_handle:
            self._unload_reserved(entry)

    def _select_entry(
        self,
        task: LocalModelTask,
        content: str,
    ) -> tuple[_ModelEntry, int]:
        with self._lock:
            candidates = sorted(
                (
                    entry
                    for entry in self._entries.values()
                    if entry.descriptor.availability is ModelAvailability.AVAILABLE
                    and task in entry.descriptor.tasks
                ),
                key=lambda entry: (
                    entry.descriptor.requirements.ram_bytes,
                    entry.descriptor.model_id,
                ),
            )
            if not candidates:
                raise LocalModelUnavailableError(
                    "No compatible local model is available."
                )
            candidates = [
                entry
                for entry in candidates
                if entry.state not in {
                    ModelLifecycleState.LOADING,
                    ModelLifecycleState.BUSY,
                    ModelLifecycleState.UNLOADING,
                }
            ]
            if not candidates:
                raise LocalModelBusyError("Compatible local models are currently busy.")

        context_limits: list[tuple[_ModelEntry, int]] = []
        for entry in candidates:
            try:
                context_tokens = self._token_counter.count_tokens(
                    entry.descriptor.model_id,
                    task,
                    content,
                )
            except Exception:
                raise LocalModelResourceError(
                    "Input context could not be determined."
                ) from None
            if type(context_tokens) is not int or context_tokens <= 0:
                raise LocalModelResourceError(
                    "Input context could not be determined."
                )
            if context_tokens <= min(
                self._limits.max_context_tokens,
                entry.descriptor.max_context_tokens,
            ):
                context_limits.append((entry, context_tokens))
        if not context_limits:
            raise LocalModelResourceError("Configured context limit was exceeded.")
        return context_limits[0]

    def _get_snapshot(self, descriptor: LocalModelDescriptor) -> ResourceSnapshot:
        try:
            snapshot = self._resource_provider.snapshot()
        except Exception:
            raise LocalModelResourceError(
                "Local model resources are unavailable."
            ) from None
        if (
            type(snapshot) is not ResourceSnapshot
            or snapshot.source is ResourceSource.UNKNOWN
        ):
            raise LocalModelResourceError(
                "Local model resources are unavailable."
            )
        if (
            snapshot.source is ResourceSource.SIMULATED
            and not self._allow_simulated_resources_for_tests
        ):
            raise LocalModelResourceError(
                "Simulated resources cannot authorize a model load."
            )
        if not _resource_snapshot_is_fresh(snapshot):
            raise LocalModelResourceError(
                "Local model resource measurements are stale."
            )
        if snapshot.available_ram_bytes is None:
            raise LocalModelResourceError(
                "Available RAM could not be determined."
            )
        requirements = descriptor.requirements
        if requirements.vram_bytes is not None and (
            snapshot.available_vram_bytes is None
            or self._limits.vram_budget_bytes is None
        ):
            raise LocalModelResourceError(
                "Required VRAM could not be determined or budgeted."
            )
        if self._limits.ram_budget_bytes is None:
            raise LocalModelResourceError("A RAM budget must be configured.")
        if (
            self._limits.max_system_load is not None
            and (
                snapshot.system_load is None
                or snapshot.system_load > self._limits.max_system_load
            )
        ):
            raise LocalModelResourceError(
                "System load exceeds the configured limit or is unknown."
            )
        if (
            requirements.ram_bytes > self._limits.ram_budget_bytes
            or requirements.ram_bytes > snapshot.available_ram_bytes
        ):
            raise LocalModelResourceError("Insufficient available RAM.")
        if requirements.vram_bytes is not None and (
            requirements.vram_bytes > self._limits.vram_budget_bytes
            or requirements.vram_bytes > snapshot.available_vram_bytes
        ):
            raise LocalModelResourceError("Insufficient available VRAM.")
        return snapshot

    def _reserve(
        self,
        entry: _ModelEntry,
        snapshot: ResourceSnapshot | None,
        needs_load: bool,
    ) -> None:
        with self._lock:
            if entry.active:
                raise LocalModelBusyError("Local model is currently in use.")
            if self._active_inferences >= self._limits.max_concurrent_inferences:
                raise LocalModelBusyError(
                    "Local inference concurrency limit was reached."
                )
            if needs_load:
                if (
                    snapshot is None
                    or entry.state is not ModelLifecycleState.UNLOADED
                ):
                    raise LocalModelBusyError(
                        "Local model state changed during admission."
                    )
                if self._active_loads >= self._limits.max_concurrent_loads:
                    raise LocalModelBusyError(
                        "Local model load concurrency limit was reached."
                    )
                self._check_aggregate_budget(entry.descriptor, snapshot)
                entry.state = ModelLifecycleState.LOADING
                entry.resource_reservation_held = True
                self._active_loads += 1
            elif entry.state is not ModelLifecycleState.READY:
                raise LocalModelBusyError("Local model is not ready.")
            entry.active = True
            self._active_inferences += 1
            if not needs_load:
                entry.state = ModelLifecycleState.BUSY

    def _check_aggregate_budget(
        self,
        new_model: LocalModelDescriptor,
        snapshot: ResourceSnapshot,
    ) -> None:
        if self._limits.ram_budget_bytes is None:
            raise LocalModelResourceError("A RAM budget must be configured.")
        ram_in_use = new_model.requirements.ram_bytes
        vram_in_use = new_model.requirements.vram_bytes or 0
        for entry in self._entries.values():
            if entry.resource_reservation_held:
                ram_in_use += entry.descriptor.requirements.ram_bytes
                vram_in_use += entry.descriptor.requirements.vram_bytes or 0
        if (
            ram_in_use > self._limits.ram_budget_bytes
            or snapshot.available_ram_bytes is None
            or ram_in_use > snapshot.available_ram_bytes
        ):
            raise LocalModelResourceError(
                "Managed models exceed the available RAM budget."
            )
        if vram_in_use or self._limits.vram_budget_bytes is not None:
            if (
                self._limits.vram_budget_bytes is None
                or snapshot.available_vram_bytes is None
                or vram_in_use > self._limits.vram_budget_bytes
                or vram_in_use > snapshot.available_vram_bytes
            ):
                raise LocalModelResourceError(
                    "Managed models exceed the available VRAM budget."
                )

    def _finish_load_failure(
        self,
        entry: _ModelEntry,
        cleanup_confirmed: bool,
    ) -> None:
        with self._lock:
            entry.active = False
            entry.state = ModelLifecycleState.ERROR
            if cleanup_confirmed:
                entry.resource_reservation_held = False
            self._active_loads -= 1
            self._active_inferences -= 1

    def _release_unstarted_load(self, entry: _ModelEntry) -> None:
        with self._lock:
            if (
                entry.state is not ModelLifecycleState.LOADING
                or entry.handle is not None
                or not entry.resource_reservation_held
            ):
                raise LocalModelBusyError(
                    "Local model state changed before loading."
                )
            entry.active = False
            entry.resource_reservation_held = False
            entry.state = ModelLifecycleState.UNLOADED
            self._active_loads -= 1
            self._active_inferences -= 1

    def _confirm_failed_load_cleanup(self, entry: _ModelEntry) -> None:
        try:
            released = (
                self._backend.failed_load_released(entry.descriptor.model_id)
                is True
            )
        except Exception:
            released = False
        if not released:
            raise LocalModelResourceError(
                "Local model resource state is uncertain."
            )
        with self._lock:
            if (
                entry.handle is not None
                or entry.state not in {
                    ModelLifecycleState.ERROR,
                    ModelLifecycleState.UNLOADED,
                }
            ):
                raise LocalModelBusyError(
                    "Local model state changed during cleanup confirmation."
                )
            entry.resource_reservation_held = False
            entry.state = ModelLifecycleState.UNLOADED

    def _finish_inference_failure(self, entry: _ModelEntry) -> None:
        with self._lock:
            entry.active = False
            entry.state = ModelLifecycleState.ERROR
            self._active_inferences -= 1

    def _begin_unload_after_run(self, entry: _ModelEntry) -> None:
        with self._lock:
            entry.active = False
            entry.state = ModelLifecycleState.UNLOADING
            self._active_inferences -= 1

    def _unload_reserved(self, entry: _ModelEntry) -> None:
        handle = entry.handle
        if handle is None:
            with self._lock:
                entry.state = ModelLifecycleState.UNLOADED
            return
        try:
            self._backend.unload(handle)
        except BaseException as error:
            with self._lock:
                entry.state = ModelLifecycleState.ERROR
            if isinstance(error, Exception):
                raise LocalModelUnloadError(
                    "Local model could not be unloaded."
                ) from None
            raise
        with self._lock:
            entry.handle = None
            entry.resource_reservation_held = False
            entry.state = ModelLifecycleState.UNLOADED


class ManagedLocalAIProvider(AIProviderContract):
    """Informational-only AI adapter over a trusted local model manager."""

    def __init__(self, manager: LocalModelManager) -> None:
        if type(manager) is not LocalModelManager:
            raise TypeError("Managed local provider requires a local model manager.")
        self._manager = manager

    def classify_intent(self, _user_text: str) -> dict[str, object]:
        del _user_text
        raise LocalModelUnavailableError(
            "Managed local provider supports informational requests only."
        )

    def propose_action(self, _user_text: str) -> ActionSuggestion:
        del _user_text
        raise LocalModelUnavailableError(
            "Managed local provider does not propose executable actions."
        )

    def respond(self, _user_text: str) -> str:
        del _user_text
        raise LocalModelUnavailableError(
            "Managed local provider supports typed informational requests only."
        )

    def respond_informational(
        self,
        request: InformationalRequest,
    ) -> InformationalResponse:
        if type(request) is not InformationalRequest:
            raise TypeError("Managed local provider requires an informational request.")
        tasks = {
            "answer_question": LocalModelTask.ANSWER_QUESTION,
            "summarize_text": LocalModelTask.SUMMARIZE_TEXT,
        }
        task = tasks.get(request.capability)
        if task is None:
            raise LocalModelUnavailableError(
                "Informational task is unavailable."
            )
        return InformationalResponse(
            text=self._manager.run(task=task, content=request.content)
        )
