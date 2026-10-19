from __future__ import annotations

from threading import Event, Thread

import pytest

from jarvis.ai.local_models import (
    ContextTokenCounter,
    LocalModelBusyError,
    LocalModelDescriptor,
    LocalModelInferenceError,
    LocalModelLimits,
    LocalModelLoadError,
    LocalModelManager,
    LocalModelResourceError,
    LocalModelTask,
    LocalModelUnloadError,
    LocalModelUnavailableError,
    ManagedLocalAIProvider,
    ModelAvailability,
    ModelLifecycleState,
    ModelRequirements,
    ResourceSnapshot,
    ResourceSource,
)
from jarvis.ai.selection import AIProviderSelector
from jarvis.ai.simulator import SimulatedAIProvider
from jarvis.app import create_application
from jarvis.config.settings import Settings


def _descriptor(
    model_id: str = "test-model",
    *,
    tasks: frozenset[LocalModelTask] | None = None,
    max_context_tokens: int = 128,
    ram_bytes: int = 100,
    vram_bytes: int | None = None,
    availability: ModelAvailability = ModelAvailability.AVAILABLE,
) -> LocalModelDescriptor:
    return LocalModelDescriptor(
        model_id=model_id,
        tasks=tasks or frozenset({LocalModelTask.ANSWER_QUESTION}),
        max_context_tokens=max_context_tokens,
        requirements=ModelRequirements(ram_bytes, vram_bytes),
        availability=availability,
    )


class SimulatedResourceProvider:
    def __init__(self, snapshot: ResourceSnapshot | None = None) -> None:
        self.value = snapshot or ResourceSnapshot(
            source=ResourceSource.SIMULATED,
            available_ram_bytes=10_000,
            available_vram_bytes=10_000,
            system_load=0.1,
        )
        self.calls = 0

    def snapshot(self) -> ResourceSnapshot:
        self.calls += 1
        return self.value


class SimulatedContextTokenCounter:
    def count_tokens(
        self,
        model_id: str,
        task: LocalModelTask,
        content: str,
    ) -> int:
        del model_id, task
        return len(content)


class SimulatedLocalModelBackend:
    def __init__(self) -> None:
        self.load_calls = 0
        self.infer_calls = 0
        self.active_infer_calls = 0
        self.unload_calls = 0
        self.unloaded_model_ids: list[str] = []
        self.fail_load = False
        self.load_cleanup_confirmed = True
        self.fail_infer = False
        self.fail_unload = False
        self.invalid_response = False
        self.last_context_tokens: int | None = None
        self.last_context_limit: int | None = None
        self.last_model_id: str | None = None
        self.load_entered: Event | None = None
        self.infer_entered: Event | None = None
        self.two_inferences_entered: Event | None = None
        self.release_load: Event | None = None
        self.release_infer: Event | None = None

    def load(self, model_id: str) -> object:
        self.load_calls += 1
        if self.load_entered is not None:
            self.load_entered.set()
        if self.release_load is not None:
            if not self.release_load.wait(timeout=2):
                raise TimeoutError("simulated test load wait expired")
        if self.fail_load:
            raise RuntimeError("PRIVATE_LOAD_FAILURE")
        return {"model_id": model_id}

    def failed_load_released(self, model_id: str) -> bool:
        del model_id
        return self.load_cleanup_confirmed

    def infer(
        self,
        handle: object,
        task: LocalModelTask,
        content: str,
        context_tokens: int,
        max_context_tokens: int,
    ) -> str:
        self.infer_calls += 1
        self.active_infer_calls += 1
        if type(handle) is dict and type(handle.get("model_id")) is str:
            self.last_model_id = handle["model_id"]
        self.last_context_tokens = context_tokens
        self.last_context_limit = max_context_tokens
        if self.infer_entered is not None:
            self.infer_entered.set()
        if self.active_infer_calls >= 2 and self.two_inferences_entered is not None:
            self.two_inferences_entered.set()
        if self.release_infer is not None:
            if not self.release_infer.wait(timeout=2):
                self.active_infer_calls -= 1
                raise TimeoutError("simulated test inference wait expired")
        if self.fail_infer:
            self.active_infer_calls -= 1
            raise RuntimeError("PRIVATE_INFERENCE_FAILURE")
        if self.invalid_response:
            self.active_infer_calls -= 1
            return ""
        self.active_infer_calls -= 1
        return f"simulated {task.value}: {content}"

    def unload(self, handle: object) -> None:
        self.unload_calls += 1
        if type(handle) is dict and type(handle.get("model_id")) is str:
            self.unloaded_model_ids.append(handle["model_id"])
        if self.fail_unload:
            raise RuntimeError("PRIVATE_UNLOAD_FAILURE")


def _manager(
    *,
    backend: SimulatedLocalModelBackend | None = None,
    resource_provider: SimulatedResourceProvider | None = None,
    token_counter: ContextTokenCounter | None = None,
    limits: LocalModelLimits | None = None,
    descriptors: tuple[LocalModelDescriptor, ...] = (_descriptor(),),
) -> tuple[LocalModelManager, SimulatedLocalModelBackend, SimulatedResourceProvider]:
    selected_backend = backend or SimulatedLocalModelBackend()
    selected_resources = resource_provider or SimulatedResourceProvider()
    manager = LocalModelManager(
        backend=selected_backend,
        resource_provider=selected_resources,
        token_counter=token_counter or SimulatedContextTokenCounter(),
        limits=limits
        or LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=10_000,
            vram_budget_bytes=10_000,
            unload_after_inference=False,
        ),
        allow_simulated_resources_for_tests=True,
    )
    for descriptor in descriptors:
        manager.register(descriptor)
    return manager, selected_backend, selected_resources


@pytest.mark.parametrize(
    "kwargs",
    (
        {"model_id": "../model"},
        {"tasks": frozenset()},
        {"tasks": frozenset({"answer_question"})},
        {"max_context_tokens": 0},
        {"requirements": lambda: ModelRequirements(0)},
    ),
)
def test_model_descriptor_rejects_invalid_capabilities_and_limits(kwargs) -> None:
    values = {
        "model_id": "model",
        "tasks": frozenset({LocalModelTask.ANSWER_QUESTION}),
        "max_context_tokens": 128,
        "requirements": ModelRequirements(100),
    }
    with pytest.raises((TypeError, ValueError)):
        values.update(
            {
                key: value() if callable(value) else value
                for key, value in kwargs.items()
            }
        )
        LocalModelDescriptor(**values)


def test_resource_and_lifecycle_contracts_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        ResourceSnapshot(ResourceSource.UNKNOWN, available_ram_bytes=10)
    with pytest.raises(ValueError):
        ResourceSnapshot(ResourceSource.SIMULATED, available_ram_bytes=-1)
    with pytest.raises(ValueError):
        ResourceSnapshot(ResourceSource.SIMULATED, available_ram_bytes=1, system_load=2)
    with pytest.raises(ValueError):
        LocalModelLimits(
            max_context_tokens=-1,
            ram_budget_bytes=100,
        )
    with pytest.raises(ValueError):
        ModelRequirements(100, vram_bytes=0)


def test_local_limits_come_from_validated_settings_and_disable_by_default() -> None:
    defaults = LocalModelLimits.from_settings(Settings())
    assert defaults.max_context_tokens == 0
    assert defaults.ram_budget_bytes is None

    settings = Settings(
        local_max_context_tokens=256,
        local_ram_budget_mb=64,
        local_vram_budget_mb=32,
        local_max_system_load=0.8,
        local_max_concurrent_loads=2,
        local_max_concurrent_inferences=3,
        local_unload_after_inference=False,
    )
    limits = LocalModelLimits.from_settings(settings)
    assert limits.max_context_tokens == 256
    assert limits.ram_budget_bytes == 64 * 1024 * 1024
    assert limits.vram_budget_bytes == 32 * 1024 * 1024
    assert limits.max_concurrent_loads == 2
    assert limits.max_concurrent_inferences == 3
    assert limits.unload_after_inference is False


def test_local_limits_load_from_environment_with_strict_values() -> None:
    settings = Settings.from_env(
        {
            "JARVIS_LOCAL_MAX_CONTEXT_TOKENS": "256",
            "JARVIS_LOCAL_RAM_BUDGET_MB": "64",
            "JARVIS_LOCAL_VRAM_BUDGET_MB": "32",
            "JARVIS_LOCAL_MAX_SYSTEM_LOAD": "0.75",
            "JARVIS_LOCAL_MAX_CONCURRENT_LOADS": "2",
            "JARVIS_LOCAL_MAX_CONCURRENT_INFERENCES": "3",
            "JARVIS_LOCAL_UNLOAD_AFTER_INFERENCE": "false",
        }
    )
    limits = LocalModelLimits.from_settings(settings)
    assert limits.max_context_tokens == 256
    assert limits.ram_budget_bytes == 64 * 1024 * 1024
    assert limits.vram_budget_bytes == 32 * 1024 * 1024
    assert limits.max_system_load == 0.75
    assert limits.max_concurrent_loads == 2
    assert limits.max_concurrent_inferences == 3
    assert limits.unload_after_inference is False


@pytest.mark.parametrize(
    "settings",
    (
        {"local_max_context_tokens": -1},
        {"local_ram_budget_mb": 0},
        {"local_vram_budget_mb": 0},
        {"local_max_concurrent_loads": 0},
        {"local_max_concurrent_inferences": 17},
        {"local_max_system_load": 1.1},
        {"local_unload_after_inference": "sometimes"},
    ),
)
def test_invalid_local_resource_settings_are_rejected(settings) -> None:
    with pytest.raises(ValueError):
        Settings(**settings)


def test_model_selection_requires_declared_task_and_compatible_context() -> None:
    descriptor = _descriptor(
        tasks=frozenset({LocalModelTask.SUMMARIZE_TEXT}),
        max_context_tokens=64,
    )
    manager, backend, resources = _manager(descriptors=(descriptor,))

    with pytest.raises(LocalModelUnavailableError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    with pytest.raises(LocalModelResourceError):
        manager.run(
            task=LocalModelTask.SUMMARIZE_TEXT,
            content="x" * 65,
        )
    assert backend.load_calls == 0
    assert resources.calls == 0


def test_selection_prefers_the_lowest_estimated_ram_model_for_supported_task() -> None:
    descriptors = (
        _descriptor("large", ram_bytes=500),
        _descriptor("small", ram_bytes=100),
    )
    manager, backend, _ = _manager(descriptors=descriptors)

    manager.run(
        task=LocalModelTask.ANSWER_QUESTION,
        content="input",
    )

    assert backend.last_model_id == "small"


@pytest.mark.parametrize(
    "snapshot",
    (
        ResourceSnapshot(ResourceSource.UNKNOWN, available_ram_bytes=None),
        ResourceSnapshot(ResourceSource.SIMULATED, available_ram_bytes=None),
        ResourceSnapshot(ResourceSource.SIMULATED, available_ram_bytes=99),
    ),
)
def test_unknown_or_insufficient_ram_denies_admission(snapshot) -> None:
    manager, backend, _ = _manager(
        resource_provider=SimulatedResourceProvider(snapshot),
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=1_000,
            unload_after_inference=False,
        ),
    )

    with pytest.raises(LocalModelResourceError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert backend.load_calls == 0


def test_simulated_resource_source_cannot_authorize_non_test_manager_loads() -> None:
    manager = LocalModelManager(
        backend=SimulatedLocalModelBackend(),
        resource_provider=SimulatedResourceProvider(),
        token_counter=SimulatedContextTokenCounter(),
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=1_000,
        ),
    )
    manager.register(_descriptor())

    with pytest.raises(LocalModelResourceError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert manager.state("test-model") is ModelLifecycleState.UNLOADED


def test_stale_resource_snapshot_denies_admission() -> None:
    stale = ResourceSnapshot(
        ResourceSource.OPERATING_SYSTEM,
        available_ram_bytes=10_000,
        captured_at_monotonic=0,
    )
    manager, backend, _ = _manager(
        resource_provider=SimulatedResourceProvider(stale),
    )

    with pytest.raises(LocalModelResourceError, match="stale"):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert backend.load_calls == 0


@pytest.mark.parametrize(
    ("snapshot", "limits", "descriptors"),
    (
        (
            ResourceSnapshot(
                ResourceSource.SIMULATED,
                available_ram_bytes=100,
                available_vram_bytes=1_000,
            ),
            LocalModelLimits(
                128,
                1_000,
                vram_budget_bytes=1_000,
                unload_after_inference=False,
            ),
            (
                _descriptor(
                    "first",
                    tasks=frozenset({LocalModelTask.ANSWER_QUESTION}),
                    ram_bytes=70,
                ),
                _descriptor(
                    "second",
                    tasks=frozenset({LocalModelTask.SUMMARIZE_TEXT}),
                    ram_bytes=70,
                ),
            ),
        ),
        (
            ResourceSnapshot(
                ResourceSource.SIMULATED,
                available_ram_bytes=1_000,
                available_vram_bytes=100,
            ),
            LocalModelLimits(
                128,
                1_000,
                vram_budget_bytes=1_000,
                unload_after_inference=False,
            ),
            (
                _descriptor(
                    "first",
                    tasks=frozenset({LocalModelTask.ANSWER_QUESTION}),
                    ram_bytes=70,
                    vram_bytes=70,
                ),
                _descriptor(
                    "second",
                    tasks=frozenset({LocalModelTask.SUMMARIZE_TEXT}),
                    ram_bytes=70,
                    vram_bytes=70,
                ),
            ),
        ),
        (
            ResourceSnapshot(
                ResourceSource.SIMULATED,
                available_ram_bytes=1_000,
                available_vram_bytes=1_000,
            ),
            LocalModelLimits(
                128,
                1_000,
                vram_budget_bytes=100,
                unload_after_inference=False,
            ),
            (
                _descriptor(
                    "first",
                    tasks=frozenset({LocalModelTask.ANSWER_QUESTION}),
                    ram_bytes=70,
                    vram_bytes=70,
                ),
                _descriptor(
                    "second",
                    tasks=frozenset({LocalModelTask.SUMMARIZE_TEXT}),
                    ram_bytes=70,
                    vram_bytes=70,
                ),
            ),
        ),
    ),
)
def test_aggregate_physical_ram_and_vram_admission_is_enforced(
    snapshot,
    limits,
    descriptors,
) -> None:
    manager, backend, _ = _manager(
        resource_provider=SimulatedResourceProvider(snapshot),
        limits=limits,
        descriptors=descriptors,
    )
    manager.run(task=LocalModelTask.ANSWER_QUESTION, content="question")

    with pytest.raises(LocalModelResourceError):
        manager.run(task=LocalModelTask.SUMMARIZE_TEXT, content="summary")

    assert backend.load_calls == 1
    assert manager.state("first") is ModelLifecycleState.READY
    assert manager.state("second") is ModelLifecycleState.UNLOADED


def test_concurrent_load_reservation_prevents_overcommitting_physical_ram() -> None:
    backend = SimulatedLocalModelBackend()
    backend.load_entered = Event()
    backend.release_load = Event()
    descriptors = (
        _descriptor(
            "question",
            tasks=frozenset({LocalModelTask.ANSWER_QUESTION}),
            ram_bytes=70,
        ),
        _descriptor(
            "summary",
            tasks=frozenset({LocalModelTask.SUMMARIZE_TEXT}),
            ram_bytes=70,
        ),
    )
    resources = SimulatedResourceProvider(
        ResourceSnapshot(
            ResourceSource.SIMULATED,
            available_ram_bytes=100,
        )
    )
    manager, _, _ = _manager(
        backend=backend,
        resource_provider=resources,
        descriptors=descriptors,
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=1_000,
            max_concurrent_loads=2,
            max_concurrent_inferences=2,
            unload_after_inference=False,
        ),
    )
    errors: list[BaseException] = []

    def run_first() -> None:
        try:
            manager.run(
                task=LocalModelTask.ANSWER_QUESTION,
                content="question",
            )
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=run_first)
    worker.start()
    assert backend.load_entered.wait(timeout=2)
    with pytest.raises(LocalModelBusyError):
        manager.unload("question")
    with pytest.raises(LocalModelResourceError):
        manager.run(
            task=LocalModelTask.SUMMARIZE_TEXT,
            content="summary",
        )
    assert backend.load_calls == 1

    backend.release_load.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert errors == []


def test_resource_provider_error_is_sanitized_and_denies_load() -> None:
    class FailingResourceProvider:
        def snapshot(self) -> ResourceSnapshot:
            raise RuntimeError("PRIVATE_RESOURCE_FAILURE")

    backend = SimulatedLocalModelBackend()
    manager = LocalModelManager(
        backend=backend,
        resource_provider=FailingResourceProvider(),
        token_counter=SimulatedContextTokenCounter(),
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=1_000,
        ),
    )
    manager.register(_descriptor())

    with pytest.raises(LocalModelResourceError) as error:
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert "PRIVATE_RESOURCE_FAILURE" not in str(error.value)
    assert backend.load_calls == 0


def test_vram_and_system_load_require_known_values_and_fit_configured_limits() -> None:
    descriptor = _descriptor(vram_bytes=100)
    unknown_vram = SimulatedResourceProvider(
        ResourceSnapshot(ResourceSource.SIMULATED, available_ram_bytes=1_000)
    )
    manager, backend, _ = _manager(
        resource_provider=unknown_vram,
        descriptors=(descriptor,),
    )
    with pytest.raises(LocalModelResourceError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert backend.load_calls == 0

    overloaded = SimulatedResourceProvider(
        ResourceSnapshot(
            ResourceSource.SIMULATED,
            available_ram_bytes=1_000,
            available_vram_bytes=1_000,
            system_load=0.9,
        )
    )
    manager, backend, _ = _manager(
        resource_provider=overloaded,
        descriptors=(descriptor,),
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=1_000,
            vram_budget_bytes=1_000,
            max_system_load=0.8,
            unload_after_inference=False,
        ),
    )
    with pytest.raises(LocalModelResourceError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert backend.load_calls == 0


def test_aggregate_estimated_memory_budget_denies_additional_model_load() -> None:
    descriptors = (
        _descriptor(
            "question-model",
            tasks=frozenset({LocalModelTask.ANSWER_QUESTION}),
            ram_bytes=100,
        ),
        _descriptor(
            "summary-model",
            tasks=frozenset({LocalModelTask.SUMMARIZE_TEXT}),
            ram_bytes=100,
        ),
    )
    manager, backend, _ = _manager(
        descriptors=descriptors,
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=150,
            unload_after_inference=False,
        ),
    )
    manager.run(
        task=LocalModelTask.ANSWER_QUESTION,
        content="question",
    )

    with pytest.raises(LocalModelResourceError):
        manager.run(
            task=LocalModelTask.SUMMARIZE_TEXT,
            content="summary",
        )
    assert backend.load_calls == 1


def test_context_limit_is_enforced_before_resource_lookup_or_loading() -> None:
    manager, backend, resources = _manager(
        limits=LocalModelLimits(
            max_context_tokens=2,
            ram_budget_bytes=1_000,
            unload_after_inference=False,
        ),
    )
    with pytest.raises(LocalModelResourceError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="three",
        )
    assert resources.calls == 0
    assert backend.load_calls == 0


@pytest.mark.parametrize("count", (0, -1, True, "2"))
def test_invalid_context_token_counts_deny_admission(count) -> None:
    class InvalidCounter:
        def count_tokens(
            self,
            model_id: str,
            task: LocalModelTask,
            content: str,
        ) -> int:
            del model_id, task, content
            return count

    manager, backend, resources = _manager(token_counter=InvalidCounter())

    with pytest.raises(LocalModelResourceError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert resources.calls == 0
    assert backend.load_calls == 0


def test_context_token_counter_failure_is_sanitized_and_denies_admission() -> None:
    class FailingCounter:
        def count_tokens(
            self,
            model_id: str,
            task: LocalModelTask,
            content: str,
        ) -> int:
            del model_id, task, content
            raise RuntimeError("PRIVATE_TOKENIZER_FAILURE")

    manager, backend, resources = _manager(token_counter=FailingCounter())

    with pytest.raises(LocalModelResourceError) as error:
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="private input",
        )
    assert "PRIVATE_TOKENIZER_FAILURE" not in str(error.value)
    assert resources.calls == 0
    assert backend.load_calls == 0


def test_successful_run_loads_once_and_explicit_unload_releases_handle() -> None:
    manager, backend, _ = _manager()
    first = manager.run(
        task=LocalModelTask.ANSWER_QUESTION,
        content="first",
    )
    second = manager.run(
        task=LocalModelTask.ANSWER_QUESTION,
        content="second",
    )

    assert first == "simulated answer_question: first"
    assert second == "simulated answer_question: second"
    assert backend.last_context_limit == 128
    assert backend.last_context_tokens == len("second")
    assert backend.last_model_id == "test-model"
    assert backend.load_calls == 1
    assert manager.state("test-model") is ModelLifecycleState.READY

    manager.unload("test-model")
    assert backend.unload_calls == 1
    assert backend.unloaded_model_ids == ["test-model"]
    assert manager.state("test-model") is ModelLifecycleState.UNLOADED


def test_unload_after_inference_policy_releases_model_before_return() -> None:
    manager, backend, _ = _manager(
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=10_000,
            unload_after_inference=True,
        )
    )
    assert manager.run(
        task=LocalModelTask.ANSWER_QUESTION,
        content="input",
    ) == "simulated answer_question: input"
    assert manager.state("test-model") is ModelLifecycleState.UNLOADED
    assert backend.unload_calls == 1


def test_managed_local_provider_is_composed_for_informational_requests_only() -> None:
    settings = Settings(
        ai_enabled=True,
        ai_provider="local",
        local_max_context_tokens=128,
        local_ram_budget_mb=1,
        local_unload_after_inference=False,
    )
    backend = SimulatedLocalModelBackend()
    resources = SimulatedResourceProvider(
        ResourceSnapshot(
            source=ResourceSource.OPERATING_SYSTEM,
            available_ram_bytes=10_000,
            available_vram_bytes=10_000,
            system_load=0.1,
        )
    )
    manager = LocalModelManager(
        backend=backend,
        resource_provider=resources,
        token_counter=SimulatedContextTokenCounter(),
        limits=LocalModelLimits.from_settings(settings),
    )
    manager.register(_descriptor())
    app = create_application(settings, local_model_manager=manager)

    result = app.process("pregunta: información local")

    assert result.success is True
    assert result.authorized is False
    assert result.proposal is None
    assert result.tool_result is None
    assert result.informational_response == (
        "simulated answer_question: información local"
    )
    assert backend.infer_calls == 1

    provider = ManagedLocalAIProvider(manager)
    with pytest.raises(LocalModelUnavailableError):
        provider.propose_action("abre una aplicación")


def test_application_rejects_simulated_resource_test_mode_and_limit_mismatch() -> None:
    settings = Settings(
        ai_enabled=True,
        ai_provider="local",
        local_max_context_tokens=128,
        local_ram_budget_mb=1,
    )
    test_manager = LocalModelManager(
        backend=SimulatedLocalModelBackend(),
        resource_provider=SimulatedResourceProvider(),
        token_counter=SimulatedContextTokenCounter(),
        limits=LocalModelLimits.from_settings(settings),
        allow_simulated_resources_for_tests=True,
    )
    with pytest.raises(ValueError, match="simulated resources"):
        create_application(settings, local_model_manager=test_manager)

    mismatched_manager = LocalModelManager(
        backend=SimulatedLocalModelBackend(),
        resource_provider=SimulatedResourceProvider(
            ResourceSnapshot(
                ResourceSource.OPERATING_SYSTEM,
                available_ram_bytes=10_000,
            )
        ),
        token_counter=SimulatedContextTokenCounter(),
        limits=LocalModelLimits(
            max_context_tokens=64,
            ram_budget_bytes=1024,
        ),
    )
    with pytest.raises(ValueError, match="limits must match"):
        create_application(settings, local_model_manager=mismatched_manager)

    with pytest.raises(ValueError):
        Settings(
            ai_enabled=True,
            ai_provider="local",
            allow_simulated_resources_for_tests=True,
        )


def test_local_inference_failure_does_not_fall_back_to_remote_provider() -> None:
    settings = Settings(
        ai_enabled=True,
        ai_provider="local",
        local_max_context_tokens=128,
        local_ram_budget_mb=1,
        local_unload_after_inference=False,
    )
    backend = SimulatedLocalModelBackend()
    backend.fail_infer = True
    resources = SimulatedResourceProvider(
        ResourceSnapshot(
            ResourceSource.OPERATING_SYSTEM,
            available_ram_bytes=10_000,
        )
    )
    manager = LocalModelManager(
        backend=backend,
        resource_provider=resources,
        token_counter=SimulatedContextTokenCounter(),
        limits=LocalModelLimits.from_settings(settings),
    )
    manager.register(_descriptor())
    app = create_application(settings, local_model_manager=manager)
    calls = {"remote": 0}

    class RemoteSpy(SimulatedAIProvider):
        def respond_informational(self, request):
            del request
            calls["remote"] += 1
            raise AssertionError("Local failure invoked the remote provider.")

    app.provider_selector = AIProviderSelector(
        {
            "local": ManagedLocalAIProvider(manager),
            "remote": RemoteSpy(),
        }
    )

    result = app.process("pregunta: local request")

    assert result.success is False
    assert result.error == "Informative response could not be generated."
    assert result.proposal is None
    assert result.tool_result is None
    assert calls == {"remote": 0}


def test_configured_inference_concurrency_is_enforced_without_waiting() -> None:
    backend = SimulatedLocalModelBackend()
    backend.infer_entered = Event()
    backend.release_infer = Event()
    manager, _, _ = _manager(
        backend=backend,
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=10_000,
            max_concurrent_inferences=1,
            unload_after_inference=False,
        ),
    )
    errors: list[BaseException] = []

    def run_inference() -> None:
        try:
            manager.run(
                task=LocalModelTask.ANSWER_QUESTION,
                content="first",
            )
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=run_inference)
    worker.start()
    assert backend.infer_entered.wait(timeout=2)

    with pytest.raises(LocalModelBusyError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="second",
        )
    with pytest.raises(LocalModelBusyError):
        manager.unload("test-model")

    backend.release_infer.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert errors == []
    assert manager.state("test-model") is ModelLifecycleState.READY


def test_duplicate_load_is_rejected_while_first_load_is_in_progress() -> None:
    backend = SimulatedLocalModelBackend()
    backend.load_entered = Event()
    backend.release_load = Event()
    manager, _, _ = _manager(backend=backend)
    errors: list[BaseException] = []

    def run_inference() -> None:
        try:
            manager.run(
                task=LocalModelTask.ANSWER_QUESTION,
                content="first",
            )
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=run_inference)
    worker.start()
    assert backend.load_entered.wait(timeout=2)

    with pytest.raises(LocalModelBusyError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="second",
        )
    assert backend.load_calls == 1

    backend.release_load.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert errors == []


def test_load_concurrency_limit_applies_across_different_models() -> None:
    backend = SimulatedLocalModelBackend()
    backend.load_entered = Event()
    backend.release_load = Event()
    descriptors = (
        _descriptor(
            "question-model",
            tasks=frozenset({LocalModelTask.ANSWER_QUESTION}),
        ),
        _descriptor(
            "summary-model",
            tasks=frozenset({LocalModelTask.SUMMARIZE_TEXT}),
        ),
    )
    manager, _, _ = _manager(
        backend=backend,
        descriptors=descriptors,
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=10_000,
            max_concurrent_loads=1,
            max_concurrent_inferences=2,
            unload_after_inference=False,
        ),
    )
    errors: list[BaseException] = []

    def run_question() -> None:
        try:
            manager.run(
                task=LocalModelTask.ANSWER_QUESTION,
                content="question",
            )
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=run_question)
    worker.start()
    assert backend.load_entered.wait(timeout=2)

    with pytest.raises(LocalModelBusyError):
        manager.run(
            task=LocalModelTask.SUMMARIZE_TEXT,
            content="summary",
        )
    assert backend.load_calls == 1

    backend.release_load.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert errors == []


def test_load_failure_is_sanitized_and_next_request_can_recover() -> None:
    backend = SimulatedLocalModelBackend()
    backend.fail_load = True
    manager, _, _ = _manager(backend=backend)

    with pytest.raises(LocalModelLoadError) as error:
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert "PRIVATE_LOAD_FAILURE" not in str(error.value)
    assert manager.state("test-model") is ModelLifecycleState.ERROR

    backend.fail_load = False
    assert manager.run(
        task=LocalModelTask.ANSWER_QUESTION,
        content="recovered",
    ) == "simulated answer_question: recovered"


def test_uncertain_failed_load_keeps_reservation_and_blocks_retry() -> None:
    backend = SimulatedLocalModelBackend()
    backend.fail_load = True
    backend.load_cleanup_confirmed = False
    manager, _, _ = _manager(backend=backend)

    with pytest.raises(LocalModelLoadError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert manager.state("test-model") is ModelLifecycleState.ERROR

    with pytest.raises(LocalModelResourceError, match="state is uncertain"):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="retry",
        )
    assert backend.load_calls == 1
    with pytest.raises(LocalModelResourceError, match="state is uncertain"):
        manager.unload("test-model")

    backend.load_cleanup_confirmed = True
    backend.fail_load = False
    assert manager.run(
        task=LocalModelTask.ANSWER_QUESTION,
        content="cleanup confirmed",
    ) == "simulated answer_question: cleanup confirmed"
    assert backend.load_calls == 2


def test_inference_failure_is_not_success_and_next_request_recovers() -> None:
    backend = SimulatedLocalModelBackend()
    backend.fail_infer = True
    manager, _, _ = _manager(backend=backend)

    with pytest.raises(LocalModelInferenceError) as error:
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert "PRIVATE_INFERENCE_FAILURE" not in str(error.value)
    assert manager.state("test-model") is ModelLifecycleState.ERROR

    backend.fail_infer = False
    assert manager.run(
        task=LocalModelTask.ANSWER_QUESTION,
        content="recovered",
    ) == "simulated answer_question: recovered"
    assert backend.unload_calls == 1
    assert backend.load_calls == 2


def test_failed_unload_keeps_model_unavailable_until_cleanup_succeeds() -> None:
    backend = SimulatedLocalModelBackend()
    manager, _, _ = _manager(backend=backend)
    manager.run(
        task=LocalModelTask.ANSWER_QUESTION,
        content="input",
    )
    backend.fail_unload = True

    with pytest.raises(LocalModelUnloadError) as error:
        manager.unload("test-model")
    assert "PRIVATE_UNLOAD_FAILURE" not in str(error.value)
    assert manager.state("test-model") is ModelLifecycleState.ERROR

    backend.fail_unload = False
    manager.unload("test-model")
    assert manager.state("test-model") is ModelLifecycleState.UNLOADED


def test_unavailable_descriptor_is_never_loaded() -> None:
    manager, backend, _ = _manager(
        descriptors=(
            _descriptor(availability=ModelAvailability.UNAVAILABLE),
        )
    )
    with pytest.raises(LocalModelUnavailableError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert backend.load_calls == 0


def test_inference_concurrency_limit_allows_only_configured_parallel_models() -> None:
    backend = SimulatedLocalModelBackend()
    backend.infer_entered = Event()
    backend.two_inferences_entered = Event()
    backend.release_infer = Event()
    descriptors = tuple(
        _descriptor(f"model-{index}", ram_bytes=100)
        for index in range(3)
    )
    manager, _, _ = _manager(
        backend=backend,
        descriptors=descriptors,
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=10_000,
            max_concurrent_loads=2,
            max_concurrent_inferences=2,
            unload_after_inference=False,
        ),
    )
    errors: list[BaseException] = []

    def run_inference(content: str) -> None:
        try:
            manager.run(
                task=LocalModelTask.ANSWER_QUESTION,
                content=content,
            )
        except BaseException as error:
            errors.append(error)

    workers = [
        Thread(target=run_inference, args=(f"request-{index}",))
        for index in range(2)
    ]
    for worker in workers:
        worker.start()
    assert backend.two_inferences_entered.wait(timeout=2)

    with pytest.raises(LocalModelBusyError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="over-limit",
        )

    backend.release_infer.set()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert errors == []
    assert backend.infer_calls == 2


def test_invalid_backend_response_is_not_reported_as_success() -> None:
    backend = SimulatedLocalModelBackend()
    backend.invalid_response = True
    manager, _, _ = _manager(backend=backend)

    with pytest.raises(LocalModelInferenceError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )


def test_manager_rejects_non_contract_components() -> None:
    with pytest.raises(TypeError):
        LocalModelManager(
            backend=object(),
            resource_provider=SimulatedResourceProvider(),
            token_counter=SimulatedContextTokenCounter(),
            limits=LocalModelLimits(128, 1000),
        )
    with pytest.raises(TypeError):
        LocalModelManager(
            backend=SimulatedLocalModelBackend(),
            resource_provider=object(),
            token_counter=SimulatedContextTokenCounter(),
            limits=LocalModelLimits(128, 1000),
        )
