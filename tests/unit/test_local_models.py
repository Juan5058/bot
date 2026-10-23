from __future__ import annotations

import ctypes
from dataclasses import asdict
import sqlite3
from threading import Barrier, Event, Thread
import time
from types import SimpleNamespace

import pytest

import jarvis.ai.local_models as local_models_module
from jarvis.ai.local_models import (
    ContextTokenCounter,
    LocalModelBusyError,
    LocalModelDescriptor,
    LocalModelInferenceError,
    LocalModelLimits,
    LocalModelLoadError,
    LocalModelManager,
    LocalModelOperationError,
    LocalModelResourceError,
    LocalModelTask,
    LocalModelUnloadError,
    LocalModelUnavailableError,
    ManagedLocalAIProvider,
    ModelAvailability,
    ModelLifecycleState,
    ModelResidencyEstimate,
    ModelRequirements,
    RemoteOperationEvidence,
    RemoteOperationEvidenceKind,
    RemoteOperationLedger,
    RemoteOperationOutcome,
    RemoteOperationRecord,
    RemoteOperationSendPhase,
    RemoteOperationType,
    ResourceSnapshot,
    ResourceSource,
    WindowsResourceProvider,
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
            total_ram_bytes=20_000,
            available_vram_bytes=10_000,
            system_load=0.1,
        )
        self.calls = 0

    def snapshot(self) -> ResourceSnapshot:
        self.calls += 1
        return self.value


class _MemoryStatusExABIReference(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


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


class _OperationEvidenceVerifier:
    def __init__(self) -> None:
        self.valid_evidence: set[
            tuple[str, RemoteOperationEvidenceKind]
        ] = set()

    def allow(
        self,
        operation_id: str,
        kind: RemoteOperationEvidenceKind,
    ) -> None:
        self.valid_evidence.add((operation_id, kind))

    def verify(
        self,
        record: RemoteOperationRecord,
        evidence: RemoteOperationEvidence,
    ) -> bool:
        return (
            record.operation_id == evidence.operation_id
            and (record.operation_id, evidence.kind) in self.valid_evidence
        )


class _BarrierEvidenceVerifier:
    def __init__(self, barrier: Barrier) -> None:
        self.barrier = barrier

    def verify(
        self,
        record: RemoteOperationRecord,
        evidence: RemoteOperationEvidence,
    ) -> bool:
        self.barrier.wait(timeout=2)
        return (
            record.operation_id == evidence.operation_id
            and evidence.kind
            is RemoteOperationEvidenceKind.VERIFIED_SERVER_PROCESS_TERMINATION
        )


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
        ResourceSnapshot(ResourceSource.SIMULATED, available_ram_bytes=0)
    with pytest.raises(ValueError):
        ResourceSnapshot(
            ResourceSource.OPERATING_SYSTEM,
            available_ram_bytes=1,
            total_ram_bytes=0,
        )
    with pytest.raises(ValueError):
        ResourceSnapshot(
            ResourceSource.OPERATING_SYSTEM,
            available_ram_bytes=1,
            total_ram_bytes=-1,
        )
    with pytest.raises(ValueError):
        ResourceSnapshot(ResourceSource.SIMULATED, available_ram_bytes=1, system_load=2)
    with pytest.raises(ValueError):
        ResourceSnapshot(
            ResourceSource.OPERATING_SYSTEM,
            available_ram_bytes=11,
            total_ram_bytes=10,
        )
    with pytest.raises(ValueError):
        ResourceSnapshot(
            ResourceSource.OPERATING_SYSTEM,
            available_ram_bytes=True,
            total_ram_bytes=10,
        )
    with pytest.raises(ValueError):
        ResourceSnapshot(
            ResourceSource.OPERATING_SYSTEM,
            available_ram_bytes=1,
            total_ram_bytes=False,
        )
    with pytest.raises(TypeError):
        ResourceSnapshot("operating_system", available_ram_bytes=1)
    with pytest.raises(TypeError):
        ResourceSnapshot("", available_ram_bytes=1)
    with pytest.raises(ValueError):
        ResourceSnapshot(
            ResourceSource.OPERATING_SYSTEM,
            available_ram_bytes=1,
            total_ram_bytes=2,
            captured_at_monotonic=float("inf"),
        )
    with pytest.raises(ValueError):
        ResourceSnapshot(
            ResourceSource.OPERATING_SYSTEM,
            available_ram_bytes=1,
            total_ram_bytes=2,
            captured_at_monotonic=True,
        )
    with pytest.raises(ValueError):
        LocalModelLimits(
            max_context_tokens=-1,
            ram_budget_bytes=100,
        )
    with pytest.raises(ValueError):
        ModelRequirements(100, vram_bytes=0)


def test_resource_snapshot_preserves_ram_bytes_and_unknown_metrics() -> None:
    snapshot = ResourceSnapshot(
        source=ResourceSource.OPERATING_SYSTEM,
        available_ram_bytes=4_000,
        total_ram_bytes=8_000,
        captured_at_monotonic=1.0,
    )
    unknown = ResourceSnapshot(
        source=ResourceSource.OPERATING_SYSTEM,
        available_ram_bytes=None,
        total_ram_bytes=None,
        available_vram_bytes=None,
    )

    assert snapshot.total_ram_bytes == 8_000
    assert snapshot.available_ram_bytes == 4_000
    assert snapshot.available_vram_bytes is None
    assert unknown.total_ram_bytes is None
    assert unknown.available_ram_bytes is None


def test_windows_resource_provider_reads_native_memory_status(monkeypatch) -> None:
    expected_fields = (
        ("dwLength", ctypes.c_uint32, 0, 4),
        ("dwMemoryLoad", ctypes.c_uint32, 4, 4),
        ("ullTotalPhys", ctypes.c_uint64, 8, 8),
        ("ullAvailPhys", ctypes.c_uint64, 16, 8),
        ("ullTotalPageFile", ctypes.c_uint64, 24, 8),
        ("ullAvailPageFile", ctypes.c_uint64, 32, 8),
        ("ullTotalVirtual", ctypes.c_uint64, 40, 8),
        ("ullAvailVirtual", ctypes.c_uint64, 48, 8),
        ("ullAvailExtendedVirtual", ctypes.c_uint64, 56, 8),
    )
    implementation_type = local_models_module._MemoryStatusEx
    reference_type = _MemoryStatusExABIReference
    assert ctypes.sizeof(ctypes.c_void_p) in {4, 8}
    assert ctypes.sizeof(reference_type) == 64
    assert ctypes.sizeof(implementation_type) == 64
    for index, (name, field_type, expected_offset, expected_size) in enumerate(
        expected_fields
    ):
        assert reference_type._fields_[index] == (name, field_type)
        assert implementation_type._fields_[index] == (name, field_type)
        assert ctypes.sizeof(field_type) == expected_size
        assert getattr(reference_type, name).offset == expected_offset
        assert getattr(implementation_type, name).offset == expected_offset

    observed: dict[str, int] = {}

    class FakeGlobalMemoryStatusEx:
        argtypes = None
        restype = None

        def __call__(self, status_pointer) -> int:
            status = ctypes.cast(
                status_pointer,
                ctypes.POINTER(reference_type),
            ).contents
            observed["length"] = status.dwLength
            observed["size"] = ctypes.sizeof(reference_type)
            status.ullTotalPhys = 16_000_000_000
            status.ullAvailPhys = 6_000_000_000
            return 1

    function = FakeGlobalMemoryStatusEx()

    class FakeKernel32:
        GlobalMemoryStatusEx = function

    monkeypatch.setattr(
        local_models_module,
        "os",
        SimpleNamespace(name="nt"),
    )
    monkeypatch.setattr(
        local_models_module.ctypes,
        "WinDLL",
        lambda name, use_last_error: FakeKernel32(),
        raising=False,
    )

    snapshot = WindowsResourceProvider().snapshot()

    assert observed["length"] == 64
    assert observed["size"] == 64
    assert function.argtypes == [
        ctypes.POINTER(implementation_type)
    ]
    assert function.restype is ctypes.c_int
    assert snapshot.source is ResourceSource.OPERATING_SYSTEM
    assert snapshot.total_ram_bytes == 16_000_000_000
    assert snapshot.available_ram_bytes == 6_000_000_000
    assert snapshot.available_vram_bytes is None
    assert snapshot.captured_at_monotonic <= time.monotonic()


@pytest.mark.parametrize(
    "failure",
    (
        "api_load",
        "api_missing",
        "api_failure",
        "api_exception",
        "incoherent_memory",
        "zero_memory",
    ),
)
def test_windows_resource_provider_fails_closed(monkeypatch, failure) -> None:
    class FakeGlobalMemoryStatusEx:
        def __call__(self, status_pointer) -> int:
            if failure == "api_exception":
                raise OSError("private Windows API error")
            if failure == "api_failure":
                return 0
            status = ctypes.cast(
                status_pointer,
                ctypes.POINTER(local_models_module._MemoryStatusEx),
            ).contents
            if failure == "incoherent_memory":
                status.ullTotalPhys = 100
                status.ullAvailPhys = 101
            elif failure == "zero_memory":
                status.ullTotalPhys = 0
                status.ullAvailPhys = 0
            else:
                status.ullTotalPhys = 100
                status.ullAvailPhys = 50
            return 1

    class FakeKernel32:
        GlobalMemoryStatusEx = FakeGlobalMemoryStatusEx()

    def load_dll(*args, **kwargs):
        if failure == "api_load":
            raise OSError("private DLL load error")
        if failure == "api_missing":
            return object()
        return FakeKernel32()

    monkeypatch.setattr(
        local_models_module,
        "os",
        SimpleNamespace(name="nt"),
    )
    monkeypatch.setattr(
        local_models_module.ctypes,
        "WinDLL",
        load_dll,
        raising=False,
    )

    with pytest.raises(LocalModelResourceError) as error:
        WindowsResourceProvider().snapshot()
    assert "private" not in str(error.value)


def test_windows_resource_provider_does_not_load_api_off_windows(monkeypatch) -> None:
    def unexpected_load(*args, **kwargs):
        raise AssertionError("Windows API must not be loaded on another OS.")

    monkeypatch.setattr(
        local_models_module,
        "os",
        SimpleNamespace(name="posix"),
    )
    monkeypatch.setattr(
        local_models_module.ctypes,
        "WinDLL",
        unexpected_load,
        raising=False,
    )

    with pytest.raises(LocalModelResourceError, match="Windows RAM measurement"):
        WindowsResourceProvider().snapshot()


def test_remote_operation_completion_does_not_change_another_record() -> None:
    ledger = RemoteOperationLedger(max_concurrent_operations=2)
    operation_a = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    operation_b = ledger.reserve(
        model_id="model-b",
        operation_type=RemoteOperationType.INFER,
    )
    ledger.mark_dispatch_attempted(operation_a.operation_id)
    ledger.mark_dispatch_attempted(operation_b.operation_id)
    before_b = ledger.get(operation_b.operation_id)

    completed_a = ledger.complete(operation_a.operation_id, content_valid=True)

    assert completed_a.outcome is RemoteOperationOutcome.TERMINAL_VALID
    assert ledger.get(operation_b.operation_id) is before_b
    assert ledger.active_operation_count == 1
    assert ledger.available_operation_slots == 1


def test_uncertain_remote_operation_continues_to_count_against_limit() -> None:
    ledger = RemoteOperationLedger(max_concurrent_operations=1)
    operation = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    ledger.mark_dispatch_attempted(operation.operation_id)
    uncertain = ledger.mark_uncertain(operation.operation_id)

    assert uncertain.outcome is RemoteOperationOutcome.UNCERTAIN
    assert ledger.active_operation_count == 1
    assert ledger.available_operation_slots == 0
    with pytest.raises(LocalModelBusyError):
        ledger.reserve(
            model_id="model-b",
            operation_type=RemoteOperationType.INFER,
        )


def test_invalid_remote_operation_transition_leaves_records_and_slots_unchanged() -> None:
    ledger = RemoteOperationLedger(max_concurrent_operations=1)
    operation = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    before_records = ledger.records()
    before_slots = ledger.available_operation_slots

    with pytest.raises(LocalModelOperationError):
        ledger.complete(operation.operation_id, content_valid=True)

    assert ledger.records() == before_records
    assert ledger.available_operation_slots == before_slots
    assert ledger.active_operation_count == 1


def test_recovery_changes_only_operation_with_verified_evidence() -> None:
    verifier = _OperationEvidenceVerifier()
    ledger = RemoteOperationLedger(
        max_concurrent_operations=2,
        evidence_verifier=verifier,
    )
    operation_a = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    operation_b = ledger.reserve(
        model_id="model-b",
        operation_type=RemoteOperationType.INFER,
    )
    for operation in (operation_a, operation_b):
        ledger.mark_dispatch_attempted(operation.operation_id)
        ledger.mark_uncertain(operation.operation_id)
    before_b = ledger.get(operation_b.operation_id)
    verifier.allow(
        operation_a.operation_id,
        RemoteOperationEvidenceKind.VERIFIED_SERVER_PROCESS_TERMINATION,
    )

    with pytest.raises(LocalModelResourceError):
        ledger.recover(
            operation_b.operation_id,
            RemoteOperationEvidence(
                operation_id=operation_b.operation_id,
                kind=RemoteOperationEvidenceKind.VERIFIED_SERVER_PROCESS_TERMINATION,
            ),
        )
    assert ledger.get(operation_b.operation_id) is before_b
    assert ledger.active_operation_count == 2

    recovered_a = ledger.recover(
        operation_a.operation_id,
        RemoteOperationEvidence(
            operation_id=operation_a.operation_id,
            kind=RemoteOperationEvidenceKind.VERIFIED_SERVER_PROCESS_TERMINATION,
        ),
    )

    assert recovered_a.outcome is RemoteOperationOutcome.RECOVERED
    assert recovered_a.completion_evidence is (
        RemoteOperationEvidenceKind.VERIFIED_SERVER_PROCESS_TERMINATION
    )
    assert (
        recovered_a.send_phase
        is RemoteOperationSendPhase.DISPATCH_ATTEMPTED
    )
    assert ledger.get(operation_b.operation_id) is before_b
    assert ledger.active_operation_count == 1


def test_uncertain_operation_recovers_from_verified_correlated_terminal_response() -> None:
    verifier = _OperationEvidenceVerifier()
    ledger = RemoteOperationLedger(
        max_concurrent_operations=2,
        evidence_verifier=verifier,
    )
    operation_a = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    operation_b = ledger.reserve(
        model_id="model-b",
        operation_type=RemoteOperationType.INFER,
    )
    for operation in (operation_a, operation_b):
        ledger.mark_dispatch_attempted(operation.operation_id)
        ledger.mark_uncertain(operation.operation_id)
    verifier.allow(
        operation_a.operation_id,
        RemoteOperationEvidenceKind.CORRELATED_TERMINAL_RESPONSE,
    )

    recovered = ledger.recover(
        operation_a.operation_id,
        RemoteOperationEvidence(
            operation_id=operation_a.operation_id,
            kind=RemoteOperationEvidenceKind.CORRELATED_TERMINAL_RESPONSE,
        ),
    )

    assert recovered.outcome is RemoteOperationOutcome.RECOVERED
    assert recovered.send_phase is RemoteOperationSendPhase.RESPONSE_RECEIVED
    assert recovered.completion_evidence is (
        RemoteOperationEvidenceKind.CORRELATED_TERMINAL_RESPONSE
    )
    assert ledger.get(operation_a.operation_id) is recovered
    assert ledger.get(operation_b.operation_id).outcome is (
        RemoteOperationOutcome.UNCERTAIN
    )
    assert ledger.active_operation_count == 1
    assert ledger.available_operation_slots == 1


def test_rejected_correlated_recovery_preserves_record_and_active_slot() -> None:
    ledger = RemoteOperationLedger(
        max_concurrent_operations=1,
        evidence_verifier=_OperationEvidenceVerifier(),
    )
    operation = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    ledger.mark_dispatch_attempted(operation.operation_id)
    uncertain = ledger.mark_uncertain(operation.operation_id)
    evidence = RemoteOperationEvidence(
        operation_id=operation.operation_id,
        kind=RemoteOperationEvidenceKind.CORRELATED_TERMINAL_RESPONSE,
    )

    with pytest.raises(LocalModelResourceError, match="could not be verified"):
        ledger.recover(operation.operation_id, evidence)

    assert ledger.get(operation.operation_id) is uncertain
    assert ledger.active_operation_count == 1
    assert ledger.available_operation_slots == 0


@pytest.mark.parametrize(
    ("outcome", "send_phase", "evidence"),
    (
        (
            RemoteOperationOutcome.RESERVED,
            RemoteOperationSendPhase.DISPATCH_ATTEMPTED,
            None,
        ),
        (
            RemoteOperationOutcome.NOT_SENT,
            RemoteOperationSendPhase.RESERVED,
            RemoteOperationEvidenceKind.NOT_DISPATCHED,
        ),
        (
            RemoteOperationOutcome.PENDING,
            RemoteOperationSendPhase.DISPATCH_ATTEMPTED,
            RemoteOperationEvidenceKind.CORRELATED_TERMINAL_RESPONSE,
        ),
        (
            RemoteOperationOutcome.UNCERTAIN,
            RemoteOperationSendPhase.RESPONSE_RECEIVED,
            None,
        ),
        (
            RemoteOperationOutcome.TERMINAL_VALID,
            RemoteOperationSendPhase.RESPONSE_RECEIVED,
            None,
        ),
        (
            RemoteOperationOutcome.RECOVERED,
            RemoteOperationSendPhase.DISPATCH_ATTEMPTED,
            RemoteOperationEvidenceKind.CORRELATED_TERMINAL_RESPONSE,
        ),
        (
            RemoteOperationOutcome.RECOVERED,
            RemoteOperationSendPhase.RESPONSE_RECEIVED,
            RemoteOperationEvidenceKind.VERIFIED_SERVER_PROCESS_TERMINATION,
        ),
    ),
)
def test_remote_operation_record_rejects_inconsistent_phase_outcome_evidence(
    outcome: RemoteOperationOutcome,
    send_phase: RemoteOperationSendPhase,
    evidence: RemoteOperationEvidenceKind | None,
) -> None:
    with pytest.raises(ValueError):
        RemoteOperationRecord(
            operation_id="local-operation",
            model_id="model-a",
            operation_type=RemoteOperationType.INFER,
            send_phase=send_phase,
            outcome=outcome,
            completion_evidence=evidence,
        )


def test_recovery_is_denied_without_a_trusted_verifier() -> None:
    ledger = RemoteOperationLedger(max_concurrent_operations=1)
    operation = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    ledger.mark_dispatch_attempted(operation.operation_id)
    ledger.mark_uncertain(operation.operation_id)
    before_records = ledger.records()
    evidence = RemoteOperationEvidence(
        operation_id=operation.operation_id,
        kind=RemoteOperationEvidenceKind.VERIFIED_SERVER_PROCESS_TERMINATION,
    )

    with pytest.raises(LocalModelResourceError, match="No trusted"):
        ledger.recover(operation.operation_id, evidence)

    assert ledger.records() == before_records
    assert ledger.active_operation_count == 1
    assert ledger.available_operation_slots == 0


def test_recovery_rejects_evidence_for_another_operation_without_mutation() -> None:
    verifier = _OperationEvidenceVerifier()
    ledger = RemoteOperationLedger(
        max_concurrent_operations=2,
        evidence_verifier=verifier,
    )
    operation_a = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    operation_b = ledger.reserve(
        model_id="model-b",
        operation_type=RemoteOperationType.INFER,
    )
    for operation in (operation_a, operation_b):
        ledger.mark_dispatch_attempted(operation.operation_id)
        ledger.mark_uncertain(operation.operation_id)
    before_records = ledger.records()
    mismatched_evidence = RemoteOperationEvidence(
        operation_id=operation_b.operation_id,
        kind=RemoteOperationEvidenceKind.VERIFIED_SERVER_PROCESS_TERMINATION,
    )

    with pytest.raises(LocalModelOperationError, match="does not resolve"):
        ledger.recover(operation_a.operation_id, mismatched_evidence)

    assert ledger.records() == before_records
    assert ledger.active_operation_count == 2


@pytest.mark.parametrize("verifier_failure", (False, RuntimeError("verifier error")))
def test_failed_or_raising_evidence_verifier_preserves_records_and_slots(
    verifier_failure: bool | RuntimeError,
) -> None:
    class FailingVerifier:
        def verify(
            self,
            record: RemoteOperationRecord,
            evidence: RemoteOperationEvidence,
        ) -> bool:
            del record, evidence
            if isinstance(verifier_failure, BaseException):
                raise verifier_failure
            return verifier_failure

    ledger = RemoteOperationLedger(
        max_concurrent_operations=1,
        evidence_verifier=FailingVerifier(),
    )
    operation = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    ledger.mark_dispatch_attempted(operation.operation_id)
    ledger.mark_uncertain(operation.operation_id)
    before_records = ledger.records()
    evidence = RemoteOperationEvidence(
        operation_id=operation.operation_id,
        kind=RemoteOperationEvidenceKind.VERIFIED_SERVER_PROCESS_TERMINATION,
    )

    with pytest.raises(LocalModelResourceError, match="could not be verified"):
        ledger.recover(operation.operation_id, evidence)

    assert ledger.records() == before_records
    assert ledger.active_operation_count == 1
    assert ledger.available_operation_slots == 0


def test_concurrent_recovery_of_same_uncertain_operation_commits_at_most_once() -> None:
    verification_barrier = Barrier(2)
    verifier = _BarrierEvidenceVerifier(verification_barrier)
    ledger = RemoteOperationLedger(
        max_concurrent_operations=2,
        evidence_verifier=verifier,
    )
    operation_a = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    operation_b = ledger.reserve(
        model_id="model-b",
        operation_type=RemoteOperationType.INFER,
    )
    for operation in (operation_a, operation_b):
        ledger.mark_dispatch_attempted(operation.operation_id)
        ledger.mark_uncertain(operation.operation_id)
    evidence = RemoteOperationEvidence(
        operation_id=operation_a.operation_id,
        kind=RemoteOperationEvidenceKind.VERIFIED_SERVER_PROCESS_TERMINATION,
    )
    recovered: list[RemoteOperationRecord] = []
    failures: list[LocalModelOperationError] = []

    def recover() -> None:
        try:
            recovered.append(ledger.recover(operation_a.operation_id, evidence))
        except LocalModelOperationError as error:
            failures.append(error)

    workers = [Thread(target=recover), Thread(target=recover)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)
        assert not worker.is_alive()

    assert len(recovered) == 1
    assert len(failures) == 1
    assert ledger.get(operation_a.operation_id) is recovered[0]
    assert recovered[0].outcome is RemoteOperationOutcome.RECOVERED
    assert ledger.get(operation_b.operation_id).outcome is (
        RemoteOperationOutcome.UNCERTAIN
    )
    assert ledger.active_operation_count == 1
    assert ledger.available_operation_slots == 1


def test_activity_and_residency_estimates_have_independent_bookkeeping() -> None:
    ledger = RemoteOperationLedger(max_concurrent_operations=1)
    estimate = ModelResidencyEstimate(
        model_id="model-a",
        estimated_ram_bytes=100,
        estimated_vram_bytes=50,
    )
    ledger.reserve_residency_estimate(estimate)
    operation = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    ledger.mark_dispatch_attempted(operation.operation_id)
    ledger.complete(operation.operation_id, content_valid=False)

    assert ledger.active_operation_count == 0
    assert ledger.residency_estimate("model-a") is estimate

    second_operation = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.UNLOAD,
    )
    ledger.mark_dispatch_attempted(second_operation.operation_id)
    discarded_estimate = ledger.discard_residency_estimate("model-a")

    assert discarded_estimate is estimate
    assert ledger.active_operation_count == 1
    assert ledger.residency_estimate("model-a") is None


def test_discarding_residency_estimate_does_not_resolve_uncertain_operation() -> None:
    ledger = RemoteOperationLedger(max_concurrent_operations=1)
    estimate = ModelResidencyEstimate(model_id="model-a", estimated_ram_bytes=100)
    ledger.reserve_residency_estimate(estimate)
    operation = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    ledger.mark_dispatch_attempted(operation.operation_id)
    ledger.mark_uncertain(operation.operation_id)

    discarded = ledger.discard_residency_estimate("model-a")

    assert discarded is estimate
    assert ledger.residency_estimate("model-a") is None
    assert ledger.get(operation.operation_id).outcome is (
        RemoteOperationOutcome.UNCERTAIN
    )
    assert ledger.active_operation_count == 1
    assert ledger.available_operation_slots == 0


@pytest.mark.parametrize("limit", (0, -1, 10_001, True))
def test_remote_operation_history_limit_is_validated(limit: int) -> None:
    with pytest.raises(ValueError, match="history limit"):
        RemoteOperationLedger(
            max_concurrent_operations=1,
            max_total_records=limit,
        )


def test_full_history_rejects_reservation_without_changing_records_or_slots() -> None:
    ledger = RemoteOperationLedger(
        max_concurrent_operations=2,
        max_total_records=2,
    )
    operation_a = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    operation_b = ledger.reserve(
        model_id="model-b",
        operation_type=RemoteOperationType.INFER,
    )
    ledger.mark_dispatch_attempted(operation_a.operation_id)
    ledger.mark_uncertain(operation_a.operation_id)
    ledger.mark_dispatch_attempted(operation_b.operation_id)
    ledger.complete(operation_b.operation_id, content_valid=False)
    before_records = ledger.records()
    before_slots = ledger.available_operation_slots

    with pytest.raises(LocalModelOperationError, match="history limit"):
        ledger.reserve(
            model_id="model-c",
            operation_type=RemoteOperationType.INFER,
        )

    assert ledger.records() == before_records
    assert ledger.total_record_count == 2
    assert ledger.available_operation_slots == before_slots
    assert ledger.get(operation_a.operation_id).outcome is (
        RemoteOperationOutcome.UNCERTAIN
    )


def test_only_one_thread_can_reserve_the_last_operation_slot() -> None:
    ledger = RemoteOperationLedger(max_concurrent_operations=1)
    barrier = Barrier(3)
    successes: list[RemoteOperationRecord] = []
    failures: list[LocalModelBusyError] = []

    def reserve_for(model_id: str) -> None:
        barrier.wait()
        try:
            successes.append(
                ledger.reserve(
                    model_id=model_id,
                    operation_type=RemoteOperationType.INFER,
                )
            )
        except LocalModelBusyError as error:
            failures.append(error)

    workers = [
        Thread(target=reserve_for, args=("model-a",)),
        Thread(target=reserve_for, args=("model-b",)),
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()

    assert len(successes) == 1
    assert len(failures) == 1
    assert ledger.active_operation_count == 1


@pytest.mark.parametrize(
    "expected_outcome",
    (
        RemoteOperationOutcome.RESERVED,
        RemoteOperationOutcome.PENDING,
        RemoteOperationOutcome.UNCERTAIN,
    ),
)
def test_sqlite_ledger_reconstructs_active_records_after_restart(
    tmp_path,
    expected_outcome: RemoteOperationOutcome,
) -> None:
    database = tmp_path / "remote-operations.sqlite3"
    first_instance = RemoteOperationLedger(
        max_concurrent_operations=1,
        sqlite_path=database,
    )
    operation = first_instance.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    if expected_outcome in {
        RemoteOperationOutcome.PENDING,
        RemoteOperationOutcome.UNCERTAIN,
    }:
        first_instance.mark_dispatch_attempted(operation.operation_id)
    if expected_outcome is RemoteOperationOutcome.UNCERTAIN:
        first_instance.mark_uncertain(operation.operation_id)
    expected = first_instance.get(operation.operation_id)
    first_instance.close()

    restarted_instance = RemoteOperationLedger(
        max_concurrent_operations=1,
        sqlite_path=database,
    )

    assert restarted_instance.get(operation.operation_id) == expected
    assert restarted_instance.active_operation_count == 1
    assert restarted_instance.available_operation_slots == 0
    with pytest.raises(LocalModelBusyError):
        restarted_instance.reserve(
            model_id="model-b",
            operation_type=RemoteOperationType.INFER,
        )
    restarted_instance.close()


@pytest.mark.parametrize(
    "expected_outcome",
    (
        RemoteOperationOutcome.RESERVED,
        RemoteOperationOutcome.PENDING,
        RemoteOperationOutcome.UNCERTAIN,
    ),
)
def test_active_operation_blocks_same_model_after_restart_but_not_other_model(
    tmp_path,
    expected_outcome: RemoteOperationOutcome,
) -> None:
    database = tmp_path / "remote-operations.sqlite3"
    first = RemoteOperationLedger(
        max_concurrent_operations=2,
        sqlite_path=database,
    )
    operation = first.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    if expected_outcome in {
        RemoteOperationOutcome.PENDING,
        RemoteOperationOutcome.UNCERTAIN,
    }:
        first.mark_dispatch_attempted(operation.operation_id)
    if expected_outcome is RemoteOperationOutcome.UNCERTAIN:
        first.mark_uncertain(operation.operation_id)
    active_record = first.get(operation.operation_id)
    first.close()

    restarted = RemoteOperationLedger(
        max_concurrent_operations=2,
        sqlite_path=database,
    )

    with pytest.raises((LocalModelBusyError, LocalModelResourceError)):
        restarted.reserve(
            model_id="model-a",
            operation_type=RemoteOperationType.UNLOAD,
        )

    other_model = restarted.reserve(
        model_id="model-b",
        operation_type=RemoteOperationType.INFER,
    )
    assert restarted.get(operation.operation_id) == active_record
    assert restarted.active_operation_count == 2
    assert restarted.available_operation_slots == 0
    assert other_model.model_id == "model-b"
    restarted.close()


def test_sqlite_connections_atomically_compete_for_last_slot(tmp_path) -> None:
    database = tmp_path / "remote-operations.sqlite3"
    first = RemoteOperationLedger(
        max_concurrent_operations=1,
        sqlite_path=database,
    )
    second = RemoteOperationLedger(
        max_concurrent_operations=1,
        sqlite_path=database,
    )
    barrier = Barrier(3)
    successes: list[RemoteOperationRecord] = []
    failures: list[LocalModelBusyError] = []

    def reserve(ledger: RemoteOperationLedger, model_id: str) -> None:
        barrier.wait()
        try:
            successes.append(
                ledger.reserve(
                    model_id=model_id,
                    operation_type=RemoteOperationType.INFER,
                )
            )
        except LocalModelBusyError as error:
            failures.append(error)

    workers = [
        Thread(target=reserve, args=(first, "model-a")),
        Thread(target=reserve, args=(second, "model-b")),
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=3)
        assert not worker.is_alive()

    assert len(successes) == 1
    assert len(failures) == 1
    assert first.active_operation_count == second.active_operation_count == 1
    first.close()
    second.close()


def test_sqlite_connections_cannot_reserve_same_active_model_concurrently(
    tmp_path,
) -> None:
    database = tmp_path / "remote-operations.sqlite3"
    first = RemoteOperationLedger(max_concurrent_operations=3, sqlite_path=database)
    second = RemoteOperationLedger(max_concurrent_operations=3, sqlite_path=database)
    barrier = Barrier(3)
    successes: list[RemoteOperationRecord] = []
    failures: list[LocalModelBusyError] = []

    def reserve(ledger: RemoteOperationLedger) -> None:
        barrier.wait()
        try:
            successes.append(
                ledger.reserve(
                    model_id="shared-model",
                    operation_type=RemoteOperationType.UNLOAD,
                )
            )
        except LocalModelBusyError as error:
            failures.append(error)

    workers = [
        Thread(target=reserve, args=(first,)),
        Thread(target=reserve, args=(second,)),
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=3)
        assert not worker.is_alive()

    assert len(successes) == 1
    assert len(failures) == 1
    assert first.active_operation_count == second.active_operation_count == 1
    first.close()
    second.close()


def test_sqlite_connections_cannot_complete_one_operation_twice(tmp_path) -> None:
    database = tmp_path / "remote-operations.sqlite3"
    first = RemoteOperationLedger(max_concurrent_operations=1, sqlite_path=database)
    second = RemoteOperationLedger(max_concurrent_operations=1, sqlite_path=database)
    operation = first.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    first.mark_dispatch_attempted(operation.operation_id)
    barrier = Barrier(3)
    successes: list[RemoteOperationRecord] = []
    failures: list[LocalModelOperationError] = []

    def complete(ledger: RemoteOperationLedger) -> None:
        barrier.wait()
        try:
            successes.append(
                ledger.complete(operation.operation_id, content_valid=True)
            )
        except LocalModelOperationError as error:
            failures.append(error)

    workers = [Thread(target=complete, args=(first,)), Thread(target=complete, args=(second,))]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=3)
        assert not worker.is_alive()

    assert len(successes) == len(failures) == 1
    assert first.get(operation.operation_id).outcome is (
        RemoteOperationOutcome.TERMINAL_VALID
    )
    assert first.active_operation_count == 0
    first.close()
    second.close()


def test_sqlite_transition_failure_rolls_back_without_releasing_slot(
    tmp_path,
) -> None:
    database = tmp_path / "remote-operations.sqlite3"
    ledger = RemoteOperationLedger(max_concurrent_operations=1, sqlite_path=database)
    operation = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TRIGGER reject_dispatch BEFORE UPDATE ON remote_operations "
        "WHEN NEW.outcome = 'pending' BEGIN "
        "SELECT RAISE(ABORT, 'PRIVATE_SQLITE_FAILURE'); END"
    )
    connection.commit()
    connection.close()

    with pytest.raises(LocalModelResourceError) as error:
        ledger.mark_dispatch_attempted(operation.operation_id)

    assert "PRIVATE_SQLITE_FAILURE" not in str(error.value)
    assert ledger.get(operation.operation_id).outcome is (
        RemoteOperationOutcome.RESERVED
    )
    assert ledger.active_operation_count == 1
    ledger.close()


@pytest.mark.parametrize(
    "tamper",
    ("invalid_state", "corrupt_file"),
)
def test_sqlite_ledger_fails_closed_on_invalid_or_corrupt_database(
    tmp_path,
    tamper: str,
) -> None:
    database = tmp_path / "remote-operations.sqlite3"
    if tamper == "invalid_state":
        ledger = RemoteOperationLedger(
            max_concurrent_operations=1,
            sqlite_path=database,
        )
        ledger.reserve(
            model_id="model-a",
            operation_type=RemoteOperationType.INFER,
        )
        ledger.close()
        connection = sqlite3.connect(database)
        connection.execute(
            "UPDATE remote_operations SET outcome = 'invalid_state'"
        )
        connection.commit()
        connection.close()
        before = database.read_bytes()
    else:
        before = b"not a sqlite database"
        database.write_bytes(before)

    with pytest.raises(LocalModelResourceError):
        RemoteOperationLedger(
            max_concurrent_operations=1,
            sqlite_path=database,
        )

    assert database.read_bytes() == before


def test_remote_operation_records_do_not_store_request_content_or_credentials() -> None:
    ledger = RemoteOperationLedger(max_concurrent_operations=1)
    record = ledger.reserve(
        model_id="model-a",
        operation_type=RemoteOperationType.INFER,
    )
    record_fields = asdict(record)

    assert set(record_fields) == {
        "operation_id",
        "model_id",
        "operation_type",
        "send_phase",
        "outcome",
        "completion_evidence",
    }
    assert len(record.operation_id) == 32
    assert "PRIVATE_PROMPT_MARKER" not in repr(record)
    assert "PRIVATE_CREDENTIAL_MARKER" not in repr(record)


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
        ResourceSnapshot(
            ResourceSource.SIMULATED,
            available_ram_bytes=99,
            total_ram_bytes=100,
        ),
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


def test_operating_system_source_label_does_not_trust_resource_provider() -> None:
    class UnverifiedOperatingSystemProvider:
        def snapshot(self) -> ResourceSnapshot:
            return ResourceSnapshot(
                ResourceSource.OPERATING_SYSTEM,
                available_ram_bytes=10_000,
                total_ram_bytes=20_000,
            )

    class MarkedTestCounter(SimulatedContextTokenCounter):
        is_production_ready = True

    backend = SimulatedLocalModelBackend()
    manager = LocalModelManager(
        backend=backend,
        resource_provider=UnverifiedOperatingSystemProvider(),
        token_counter=MarkedTestCounter(),
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=10_000,
        ),
    )
    manager.register(_descriptor())

    assert manager.is_production_ready is False
    with pytest.raises(
        LocalModelResourceError,
        match="Production local model dependencies are unavailable",
    ):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )

    assert backend.load_calls == 0


def test_unverified_token_counter_cannot_authorize_production_admission() -> None:
    class MarkedTestResourceProvider:
        is_production_ready = True

        def snapshot(self) -> ResourceSnapshot:
            return ResourceSnapshot(
                ResourceSource.OPERATING_SYSTEM,
                available_ram_bytes=10_000,
                total_ram_bytes=20_000,
            )

    backend = SimulatedLocalModelBackend()
    manager = LocalModelManager(
        backend=backend,
        resource_provider=MarkedTestResourceProvider(),
        token_counter=SimulatedContextTokenCounter(),
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=10_000,
        ),
    )
    manager.register(_descriptor())

    assert manager.is_production_ready is False
    with pytest.raises(
        LocalModelResourceError,
        match="Production local model dependencies are unavailable",
    ):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )

    assert backend.load_calls == 0


def test_windows_ram_provider_and_readiness_claims_do_not_enable_production() -> None:
    class MarkedReadyBackend(SimulatedLocalModelBackend):
        def validate_descriptor(self, descriptor: LocalModelDescriptor) -> None:
            del descriptor

    class MarkedReadyCounter(SimulatedContextTokenCounter):
        is_production_ready = True

    backend = MarkedReadyBackend()
    resources = WindowsResourceProvider()
    manager = LocalModelManager(
        backend=backend,
        resource_provider=resources,
        token_counter=MarkedReadyCounter(),
        limits=LocalModelLimits(
            max_context_tokens=128,
            ram_budget_bytes=10_000,
            vram_budget_bytes=10_000,
        ),
    )
    manager.register(_descriptor(vram_bytes=100))

    assert not hasattr(resources, "is_production_ready")
    assert manager.is_production_ready is False
    with pytest.raises(
        LocalModelResourceError,
        match="Production local model dependencies are unavailable",
    ):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert backend.load_calls == 0

    settings = Settings(
        ai_enabled=True,
        ai_provider="local",
        local_max_context_tokens=128,
        local_ram_budget_mb=1,
        local_vram_budget_mb=1,
    )
    manager = LocalModelManager(
        backend=backend,
        resource_provider=resources,
        token_counter=MarkedReadyCounter(),
        limits=LocalModelLimits.from_settings(settings),
    )
    manager.register(_descriptor(vram_bytes=100))
    with pytest.raises(ValueError, match="not verified for production"):
        create_application(settings, local_model_manager=manager)


def test_stale_resource_snapshot_denies_admission() -> None:
    stale = ResourceSnapshot(
        ResourceSource.OPERATING_SYSTEM,
        available_ram_bytes=10_000,
        total_ram_bytes=20_000,
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


def test_future_resource_snapshot_denies_admission() -> None:
    future = ResourceSnapshot(
        ResourceSource.OPERATING_SYSTEM,
        available_ram_bytes=10_000,
        total_ram_bytes=20_000,
        captured_at_monotonic=time.monotonic() + 60,
    )
    manager, backend, _ = _manager(
        resource_provider=SimulatedResourceProvider(future),
    )

    with pytest.raises(LocalModelResourceError, match="stale"):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="input",
        )
    assert backend.load_calls == 0


def test_missing_total_ram_denies_admission() -> None:
    incomplete = ResourceSnapshot(
        ResourceSource.OPERATING_SYSTEM,
        available_ram_bytes=10_000,
    )
    manager, backend, _ = _manager(
        resource_provider=SimulatedResourceProvider(incomplete),
    )

    with pytest.raises(LocalModelResourceError, match="Total RAM"):
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
                total_ram_bytes=200,
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
                total_ram_bytes=2_000,
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
                total_ram_bytes=2_000,
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
            total_ram_bytes=200,
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
        allow_simulated_resources_for_tests=True,
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
        ResourceSnapshot(
            ResourceSource.SIMULATED,
            available_ram_bytes=1_000,
            total_ram_bytes=2_000,
        )
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
            total_ram_bytes=2_000,
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


def test_application_blocks_local_managers_without_verified_production_dependencies() -> None:
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
            total_ram_bytes=20_000,
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
    with pytest.raises(ValueError, match="not verified for production"):
        create_application(settings, local_model_manager=manager)
    assert backend.infer_calls == 0

    provider = ManagedLocalAIProvider(manager)
    with pytest.raises(LocalModelUnavailableError):
        provider.propose_action("abre una aplicaciÃƒÂ³n")


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
                total_ram_bytes=20_000,
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


def test_missing_local_production_dependencies_are_unavailable_without_remote_fallback() -> None:
    settings = Settings(
        ai_enabled=True,
        ai_provider="local",
        local_max_context_tokens=128,
        local_ram_budget_mb=1,
        local_unload_after_inference=False,
    )
    calls = {"remote": 0}

    class RemoteSpy(SimulatedAIProvider):
        def respond_informational(self, request):
            del request
            calls["remote"] += 1
            raise AssertionError("Local failure invoked the remote provider.")

    app = create_application(settings)
    app.provider_selector = AIProviderSelector({"remote": RemoteSpy()})

    result = app.process("pregunta: local request")

    assert result.success is False
    assert result.error == "Informative processing is unavailable."
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
    assert manager._active_loads == 0
    assert manager._active_inferences == 0



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

    # El intento de carga terminÃƒÂ³, aunque la reserva siga retenida.
    assert manager._active_loads == 0
    assert manager._active_inferences == 0
    assert manager._entries["test-model"].resource_reservation_held is True

    with pytest.raises(LocalModelResourceError, match="state is uncertain"):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="retry",
        )

    assert backend.load_calls == 1
    assert manager._active_loads == 0
    assert manager._active_inferences == 0
    assert manager._entries["test-model"].resource_reservation_held is True

    with pytest.raises(LocalModelResourceError, match="state is uncertain"):
        manager.unload("test-model")

    assert manager._active_loads == 0
    assert manager._active_inferences == 0
    assert manager._entries["test-model"].resource_reservation_held is True

    # El backend confirma la limpieza; ahora puede intentarse otra carga.
    backend.load_cleanup_confirmed = True
    backend.fail_load = False

    assert manager.run(
        task=LocalModelTask.ANSWER_QUESTION,
        content="cleanup confirmed",
    ) == "simulated answer_question: cleanup confirmed"

    assert backend.load_calls == 2
    assert manager._active_loads == 0
    assert manager._active_inferences == 0
    assert manager._entries["test-model"].resource_reservation_held is True




def test_concurrent_failed_load_cleanup_confirmation_is_safe() -> None:
    cleanup_calls_lock = __import__("threading").Lock()
    cleanup_calls = 0
    cleanup_entered = Event()
    release_cleanup = Event()

    class ConcurrentCleanupBackend(SimulatedLocalModelBackend):
        block_cleanup = False

        def failed_load_released(self, model_id: str) -> bool:
            nonlocal cleanup_calls

            if not self.block_cleanup:
                return super().failed_load_released(model_id)

            with cleanup_calls_lock:
                cleanup_calls += 1

            cleanup_entered.set()
            if not release_cleanup.wait(timeout=3):
                raise TimeoutError("simulated cleanup wait expired")
            return True

    backend = ConcurrentCleanupBackend()
    backend.fail_load = True
    backend.load_cleanup_confirmed = False
    manager, _, _ = _manager(backend=backend)

    # El fallo inicial debe dejar la reserva retenida:
    # aquÃƒÂ­ la barrera todavÃƒÂ­a estÃƒÂ¡ desactivada.
    with pytest.raises(LocalModelLoadError):
        manager.run(
            task=LocalModelTask.ANSWER_QUESTION,
            content="initial failure",
        )

    assert manager.state("test-model") is ModelLifecycleState.ERROR
    assert backend.load_calls == 1

    # La confirmaciÃƒÂ³n debe reservar el estado de limpieza antes de consultar.
    backend.fail_load = False
    backend.load_cleanup_confirmed = True
    backend.block_cleanup = True

    results: list[str] = []
    errors: list[BaseException] = []

    def recover() -> None:
        try:
            result = manager.run(
                task=LocalModelTask.ANSWER_QUESTION,
                content="recovery",
            )
            results.append(result)
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=recover)

    try:
        worker.start()
        assert cleanup_entered.wait(timeout=3)

        with pytest.raises(LocalModelBusyError):
            manager.run(
                task=LocalModelTask.ANSWER_QUESTION,
                content="competing recovery",
            )
        assert cleanup_calls == 1
        assert backend.load_calls == 1

        release_cleanup.set()
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert len(results) == 1
        assert errors == []
        assert backend.load_calls == 2
        assert manager.state("test-model") is ModelLifecycleState.READY
    finally:
        release_cleanup.set()
        if worker.is_alive():
            worker.join(timeout=4)



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
    assert manager._active_loads == 0
    assert manager._active_inferences == 0

    # El modelo sigue cargado; el gestor conserva la reserva de recursos.
    assert manager._entries["test-model"].resource_reservation_held is True

    backend.fail_infer = False
    assert manager.run(
        task=LocalModelTask.ANSWER_QUESTION,
        content="recovered",
    ) == "simulated answer_question: recovered"

    assert backend.unload_calls == 1
    assert backend.load_calls == 2
    assert manager._active_loads == 0
    assert manager._active_inferences == 0
    assert manager._entries["test-model"].resource_reservation_held is True



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
