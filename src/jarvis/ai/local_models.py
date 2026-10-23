from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
from enum import Enum
import math
import os
from pathlib import Path
import re
import sqlite3
from threading import Lock
import time
from typing import Iterator, Protocol
from uuid import uuid4

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


class RemoteOperationType(str, Enum):
    INSPECT = "inspect"
    INFER = "infer"
    UNLOAD = "unload"


class RemoteOperationSendPhase(str, Enum):
    RESERVED = "reserved"
    NOT_DISPATCHED = "not_dispatched"
    DISPATCH_ATTEMPTED = "dispatch_attempted"
    RESPONSE_RECEIVED = "response_received"


class RemoteOperationOutcome(str, Enum):
    RESERVED = "reserved"
    PENDING = "pending"
    NOT_SENT = "not_sent"
    UNCERTAIN = "uncertain"
    TERMINAL_VALID = "terminal_valid"
    TERMINAL_INVALID = "terminal_invalid"
    RECOVERED = "recovered"


class RemoteOperationEvidenceKind(str, Enum):
    NOT_DISPATCHED = "not_dispatched"
    CORRELATED_TERMINAL_RESPONSE = "correlated_terminal_response"
    VERIFIED_SERVER_PROCESS_TERMINATION = "verified_server_process_termination"


class LocalModelError(RuntimeError):
    pass


class LocalModelOperationError(LocalModelError):
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


def _secure_ledger_directory(path: Path) -> None:
    try:
        if os.name == "nt":
            _set_windows_user_only_acl(path, create_directory=True)
        else:
            if path.is_symlink():
                raise OSError("Ledger directory must not be redirected.")
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not path.is_dir():
                raise OSError("Ledger path is not a directory.")
            os.chmod(path, 0o700)
    except Exception:
        raise LocalModelResourceError(
            "Persistent operation ledger permissions are unavailable."
        ) from None


def _secure_ledger_file(path: Path) -> None:
    try:
        if os.name == "nt":
            _set_windows_user_only_acl(path, create_directory=False)
        else:
            if path.is_symlink() or not path.is_file():
                raise OSError("Ledger database must be a regular file.")
            os.chmod(path, 0o600)
    except Exception:
        raise LocalModelResourceError(
            "Persistent operation ledger permissions are unavailable."
        ) from None


def _set_windows_user_only_acl(path: Path, *, create_directory: bool) -> None:
    import ctypes.wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    advapi32.OpenProcessToken.argtypes = (
        ctypes.wintypes.HANDLE,
        ctypes.wintypes.DWORD,
        ctypes.POINTER(ctypes.wintypes.HANDLE),
    )
    advapi32.OpenProcessToken.restype = ctypes.wintypes.BOOL
    kernel32.GetFileAttributesW.argtypes = (ctypes.wintypes.LPCWSTR,)
    kernel32.GetFileAttributesW.restype = ctypes.wintypes.DWORD
    kernel32.CreateDirectoryW.argtypes = (
        ctypes.wintypes.LPCWSTR,
        ctypes.c_void_p,
    )
    kernel32.CreateDirectoryW.restype = ctypes.wintypes.BOOL
    kernel32.LocalFree.argtypes = (ctypes.c_void_p,)
    kernel32.LocalFree.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = (ctypes.wintypes.HANDLE,)
    kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = (
        ctypes.wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.wintypes.DWORD,
        ctypes.POINTER(ctypes.wintypes.DWORD),
    )
    advapi32.GetTokenInformation.restype = ctypes.wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.wintypes.LPWSTR),
    )
    advapi32.ConvertSidToStringSidW.restype = ctypes.wintypes.BOOL
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.wintypes.DWORD),
    )
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
        ctypes.wintypes.BOOL
    )
    advapi32.SetFileSecurityW.argtypes = (
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.DWORD,
        ctypes.c_void_p,
    )
    advapi32.SetFileSecurityW.restype = ctypes.wintypes.BOOL

    class SidAndAttributes(ctypes.Structure):
        _fields_ = [("sid", ctypes.c_void_p), ("attributes", ctypes.wintypes.DWORD)]

    class TokenUser(ctypes.Structure):
        _fields_ = [("user", SidAndAttributes)]

    token = ctypes.wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(),
        0x0008,
        ctypes.byref(token),
    ):
        raise OSError("Could not inspect the current Windows user.")
    try:
        required = ctypes.wintypes.DWORD()
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(required))
        token_buffer = ctypes.create_string_buffer(required.value)
        if not advapi32.GetTokenInformation(
            token,
            1,
            ctypes.cast(token_buffer, ctypes.c_void_p),
            required.value,
            ctypes.byref(required),
        ):
            raise OSError("Could not inspect the current Windows user.")
        token_user = ctypes.cast(
            token_buffer,
            ctypes.POINTER(TokenUser),
        ).contents
        sid_text = ctypes.c_wchar_p()
        if not advapi32.ConvertSidToStringSidW(
            token_user.user.sid,
            ctypes.byref(sid_text),
        ):
            raise OSError("Could not identify the current Windows user.")
        try:
            descriptor = ctypes.c_void_p()
            if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                f"D:P(A;;FA;;;{sid_text.value})",
                1,
                ctypes.byref(descriptor),
                None,
            ):
                raise OSError("Could not create a private security descriptor.")
            try:
                if create_directory and not path.exists():
                    class SecurityAttributes(ctypes.Structure):
                        _fields_ = [
                            ("length", ctypes.wintypes.DWORD),
                            ("security_descriptor", ctypes.c_void_p),
                            ("inherit_handle", ctypes.wintypes.BOOL),
                        ]

                    security_attributes = SecurityAttributes(
                        ctypes.sizeof(SecurityAttributes),
                        descriptor,
                        False,
                    )
                    if not kernel32.CreateDirectoryW(
                        str(path),
                        ctypes.byref(security_attributes),
                    ):
                        error_code = ctypes.get_last_error()
                        if error_code != 183:
                            raise OSError("Could not create a private data directory.")
                attributes = kernel32.GetFileAttributesW(str(path))
                if attributes == 0xFFFFFFFF or attributes & 0x400:
                    raise OSError("Ledger path is unavailable or redirected.")
                if create_directory and not attributes & 0x10:
                    raise OSError("Ledger data path is not a directory.")
                if not create_directory and attributes & 0x10:
                    raise OSError("Ledger database path is not a file.")
                if not advapi32.SetFileSecurityW(
                    str(path),
                    0x00000004 | 0x80000000,
                    descriptor,
                ):
                    raise OSError("Could not restrict ledger permissions.")
            finally:
                kernel32.LocalFree(descriptor)
        finally:
            kernel32.LocalFree(ctypes.cast(sid_text, ctypes.c_void_p))
    finally:
        kernel32.CloseHandle(token)


@dataclass(frozen=True, slots=True)
class ModelRequirements:
    """Legacy estimate fields; strict backends must bind them to catalog bounds."""

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
class ModelResidencyEstimate:
    """Internal estimates, not physical reservations or proof of model unload."""

    model_id: str
    estimated_ram_bytes: int
    estimated_vram_bytes: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.model_id) is not str
            or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", self.model_id)
            is None
        ):
            raise ValueError("Model identifier is invalid.")
        if type(self.estimated_ram_bytes) is not int or self.estimated_ram_bytes <= 0:
            raise ValueError("Estimated RAM must be a positive integer.")
        if self.estimated_vram_bytes is not None and (
            type(self.estimated_vram_bytes) is not int or self.estimated_vram_bytes <= 0
        ):
            raise ValueError("Estimated VRAM must be a positive integer.")


@dataclass(frozen=True, slots=True)
class RemoteOperationEvidence:
    operation_id: str
    kind: RemoteOperationEvidenceKind

    def __post_init__(self) -> None:
        if type(self.operation_id) is not str or not self.operation_id:
            raise ValueError("Local operation identifier is invalid.")
        if type(self.kind) is not RemoteOperationEvidenceKind:
            raise TypeError("Operation evidence kind is invalid.")


@dataclass(frozen=True, slots=True)
class RemoteOperationRecord:
    """Local bookkeeping only; operation_id is not an Ollama operation ID."""

    operation_id: str
    model_id: str
    operation_type: RemoteOperationType
    send_phase: RemoteOperationSendPhase
    outcome: RemoteOperationOutcome
    completion_evidence: RemoteOperationEvidenceKind | None = None

    def __post_init__(self) -> None:
        if type(self.operation_id) is not str or not self.operation_id:
            raise ValueError("Local operation identifier is invalid.")
        if (
            type(self.model_id) is not str
            or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", self.model_id)
            is None
        ):
            raise ValueError("Model identifier is invalid.")
        if type(self.operation_type) is not RemoteOperationType:
            raise TypeError("Remote operation type is invalid.")
        if type(self.send_phase) is not RemoteOperationSendPhase:
            raise TypeError("Remote operation send phase is invalid.")
        if type(self.outcome) is not RemoteOperationOutcome:
            raise TypeError("Remote operation outcome is invalid.")
        if self.completion_evidence is not None and (
            type(self.completion_evidence) is not RemoteOperationEvidenceKind
        ):
            raise TypeError("Remote operation completion evidence is invalid.")
        expected_evidence = {
            RemoteOperationOutcome.RESERVED: None,
            RemoteOperationOutcome.PENDING: None,
            RemoteOperationOutcome.UNCERTAIN: None,
            RemoteOperationOutcome.NOT_SENT: (
                RemoteOperationEvidenceKind.NOT_DISPATCHED
            ),
            RemoteOperationOutcome.TERMINAL_VALID: (
                RemoteOperationEvidenceKind.CORRELATED_TERMINAL_RESPONSE
            ),
            RemoteOperationOutcome.TERMINAL_INVALID: (
                RemoteOperationEvidenceKind.CORRELATED_TERMINAL_RESPONSE
            ),
        }
        expected_recovered_phase: RemoteOperationSendPhase | None = None
        if self.outcome is RemoteOperationOutcome.RECOVERED:
            if self.completion_evidence not in {
                RemoteOperationEvidenceKind.CORRELATED_TERMINAL_RESPONSE,
                RemoteOperationEvidenceKind.VERIFIED_SERVER_PROCESS_TERMINATION,
            }:
                raise ValueError("Recovered operation evidence is invalid.")
            expected_recovered_phase = (
                RemoteOperationSendPhase.RESPONSE_RECEIVED
                if self.completion_evidence
                is RemoteOperationEvidenceKind.CORRELATED_TERMINAL_RESPONSE
                else RemoteOperationSendPhase.DISPATCH_ATTEMPTED
            )
            if self.send_phase is not expected_recovered_phase:
                raise ValueError("Recovered operation phase and evidence do not match.")
        elif self.completion_evidence is not expected_evidence[self.outcome]:
            raise ValueError("Operation outcome and evidence do not match.")
        expected_phase = {
            RemoteOperationOutcome.RESERVED: RemoteOperationSendPhase.RESERVED,
            RemoteOperationOutcome.NOT_SENT: (
                RemoteOperationSendPhase.NOT_DISPATCHED
            ),
            RemoteOperationOutcome.PENDING: (
                RemoteOperationSendPhase.DISPATCH_ATTEMPTED
            ),
            RemoteOperationOutcome.UNCERTAIN: (
                RemoteOperationSendPhase.DISPATCH_ATTEMPTED
            ),
            RemoteOperationOutcome.TERMINAL_VALID: (
                RemoteOperationSendPhase.RESPONSE_RECEIVED
            ),
            RemoteOperationOutcome.TERMINAL_INVALID: (
                RemoteOperationSendPhase.RESPONSE_RECEIVED
            ),
            RemoteOperationOutcome.RECOVERED: expected_recovered_phase,
        }
        if self.send_phase is not expected_phase[self.outcome]:
            raise ValueError("Operation outcome and send phase do not match.")


class RemoteOperationEvidenceVerifier(Protocol):
    def verify(
        self,
        record: RemoteOperationRecord,
        evidence: RemoteOperationEvidence,
    ) -> bool:
        """Validate typed evidence using a trusted, operation-specific source."""
        ...


class RemoteOperationLedger:
    """SQLite-capable metadata ledger; estimates are not physical resources."""

    def __init__(
        self,
        *,
        max_concurrent_operations: int,
        max_total_records: int = 10_000,
        evidence_verifier: RemoteOperationEvidenceVerifier | None = None,
        sqlite_path: str | Path | None = None,
    ) -> None:
        if (
            type(max_concurrent_operations) is not int
            or not 1 <= max_concurrent_operations <= 16
        ):
            raise ValueError("Operation concurrency must be from 1 to 16.")
        if type(max_total_records) is not int or not 1 <= max_total_records <= 10_000:
            raise ValueError("Operation history limit must be from 1 to 10000.")
        if evidence_verifier is not None and not callable(
            getattr(evidence_verifier, "verify", None)
        ):
            raise TypeError("Operation evidence verifier is invalid.")
        self._max_concurrent_operations = max_concurrent_operations
        self._max_total_records = max_total_records
        self._evidence_verifier = evidence_verifier
        self._database_path = None if sqlite_path is None else Path(sqlite_path)
        self._records: dict[str, RemoteOperationRecord] = {}
        self._residency_estimates: dict[str, ModelResidencyEstimate] = {}
        self._lock = Lock()
        self._closed = False
        allow_initialization = self._database_path is None
        if self._database_path is None:
            database_name = ":memory:"
        else:
            if not self._database_path.is_absolute():
                raise ValueError("Operation ledger database path must be absolute.")
            if self._database_path.name in {"", ".", ".."}:
                raise ValueError("Operation ledger database path is invalid.")
            _secure_ledger_directory(self._database_path.parent)
            if self._database_path.is_symlink():
                raise LocalModelResourceError(
                    "Persistent operation ledger is unavailable."
                )
            allow_initialization = not self._database_path.exists()
            if not allow_initialization:
                _secure_ledger_file(self._database_path)
            database_name = str(self._database_path)
        try:
            self._connection = sqlite3.connect(
                database_name,
                timeout=5.0,
                check_same_thread=False,
                isolation_level=None,
            )
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._initialize_database(allow_initialization=allow_initialization)
            if self._database_path is not None:
                _secure_ledger_file(self._database_path)
        except LocalModelError:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise
        except Exception:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise LocalModelResourceError(
                "Persistent operation ledger is unavailable."
            ) from None

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> RemoteOperationLedger:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    @property
    def max_concurrent_operations(self) -> int:
        return self._max_concurrent_operations

    @property
    def max_total_records(self) -> int:
        return self._max_total_records

    @property
    def is_persistent(self) -> bool:
        return self._database_path is not None

    @property
    def total_record_count(self) -> int:
        with self._lock:
            self._refresh_cache_locked()
            return len(self._records)

    @property
    def active_operation_count(self) -> int:
        with self._lock:
            return self._active_count_locked()

    @property
    def available_operation_slots(self) -> int:
        return self._max_concurrent_operations - self.active_operation_count

    def records(self) -> tuple[RemoteOperationRecord, ...]:
        with self._lock:
            self._refresh_cache_locked()
            return tuple(self._records.values())

    def get(self, operation_id: str) -> RemoteOperationRecord:
        if type(operation_id) is not str:
            raise TypeError("Local operation identifier must be text.")
        with self._lock:
            self._refresh_cache_locked()
            record = self._records.get(operation_id)
            if record is None:
                raise LocalModelUnavailableError("Local operation is unavailable.")
            return record

    def reserve(
        self,
        *,
        model_id: str,
        operation_type: RemoteOperationType,
    ) -> RemoteOperationRecord:
        if (
            type(model_id) is not str
            or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", model_id)
            is None
        ):
            raise ValueError("Model identifier is invalid.")
        if type(operation_type) is not RemoteOperationType:
            raise TypeError("Remote operation type is invalid.")
        with self._lock, self._transaction() as connection:
            existing = connection.execute(
                "SELECT outcome FROM remote_operations "
                "WHERE model_id = ? AND outcome IN (?, ?, ?) LIMIT 1",
                (
                    model_id,
                    RemoteOperationOutcome.RESERVED.value,
                    RemoteOperationOutcome.PENDING.value,
                    RemoteOperationOutcome.UNCERTAIN.value,
                ),
            ).fetchone()
            if existing is not None:
                if existing[0] == RemoteOperationOutcome.UNCERTAIN.value:
                    raise LocalModelResourceError(
                        "Local model operation state is uncertain."
                    )
                raise LocalModelBusyError(
                    "A remote operation for this model is already active."
                )
            if self._record_count_locked(connection) >= self._max_total_records:
                raise LocalModelOperationError(
                    "Remote operation history limit was reached."
                )
            if self._active_count_locked(connection) >= self._max_concurrent_operations:
                raise LocalModelBusyError(
                    "Remote operation concurrency limit was reached."
                )
            operation_id = uuid4().hex
            record = RemoteOperationRecord(
                operation_id=operation_id,
                model_id=model_id,
                operation_type=operation_type,
                send_phase=RemoteOperationSendPhase.RESERVED,
                outcome=RemoteOperationOutcome.RESERVED,
            )
            connection.execute(
                "INSERT INTO remote_operations "
                "(operation_id, model_id, operation_type, send_phase, outcome, "
                "completion_evidence) VALUES (?, ?, ?, ?, ?, NULL)",
                (
                    record.operation_id,
                    record.model_id,
                    record.operation_type.value,
                    record.send_phase.value,
                    record.outcome.value,
                ),
            )
        with self._lock:
            self._records[record.operation_id] = record
        return record

    def mark_dispatch_attempted(self, operation_id: str) -> RemoteOperationRecord:
        return self._transition(
            operation_id,
            expected=RemoteOperationOutcome.RESERVED,
            send_phase=RemoteOperationSendPhase.DISPATCH_ATTEMPTED,
            outcome=RemoteOperationOutcome.PENDING,
        )

    def mark_not_sent(self, operation_id: str) -> RemoteOperationRecord:
        return self._transition(
            operation_id,
            expected=RemoteOperationOutcome.RESERVED,
            send_phase=RemoteOperationSendPhase.NOT_DISPATCHED,
            outcome=RemoteOperationOutcome.NOT_SENT,
            evidence=RemoteOperationEvidenceKind.NOT_DISPATCHED,
        )

    def mark_not_sent_after_dispatch(
        self,
        operation_id: str,
    ) -> RemoteOperationRecord:
        """Release only when the transport proves dispatch did not begin."""
        return self._transition(
            operation_id,
            expected=RemoteOperationOutcome.PENDING,
            send_phase=RemoteOperationSendPhase.NOT_DISPATCHED,
            outcome=RemoteOperationOutcome.NOT_SENT,
            evidence=RemoteOperationEvidenceKind.NOT_DISPATCHED,
        )

    def mark_uncertain(self, operation_id: str) -> RemoteOperationRecord:
        return self._transition(
            operation_id,
            expected=RemoteOperationOutcome.PENDING,
            send_phase=RemoteOperationSendPhase.DISPATCH_ATTEMPTED,
            outcome=RemoteOperationOutcome.UNCERTAIN,
        )

    def complete(
        self,
        operation_id: str,
        *,
        content_valid: bool,
    ) -> RemoteOperationRecord:
        if type(content_valid) is not bool:
            raise TypeError("Terminal content validity must be a boolean.")
        return self._transition(
            operation_id,
            expected=RemoteOperationOutcome.PENDING,
            send_phase=RemoteOperationSendPhase.RESPONSE_RECEIVED,
            outcome=(
                RemoteOperationOutcome.TERMINAL_VALID
                if content_valid
                else RemoteOperationOutcome.TERMINAL_INVALID
            ),
            evidence=RemoteOperationEvidenceKind.CORRELATED_TERMINAL_RESPONSE,
        )

    def recover(
        self,
        operation_id: str,
        evidence: RemoteOperationEvidence,
    ) -> RemoteOperationRecord:
        if type(evidence) is not RemoteOperationEvidence:
            raise TypeError("Typed operation evidence is required.")
        if evidence.operation_id != operation_id or (
            evidence.kind is RemoteOperationEvidenceKind.NOT_DISPATCHED
        ):
            raise LocalModelOperationError(
                "Evidence does not resolve this remote operation."
            )
        if self._evidence_verifier is None:
            raise LocalModelResourceError(
                "No trusted operation evidence verifier is configured."
            )
        with self._lock:
            self._refresh_cache_locked()
            current = self._records.get(operation_id)
            if current is None:
                raise LocalModelUnavailableError("Local operation is unavailable.")
            if current.outcome is not RemoteOperationOutcome.UNCERTAIN:
                raise LocalModelOperationError(
                    "Only uncertain operations can be recovered."
                )
        try:
            verified = self._evidence_verifier.verify(current, evidence) is True
        except Exception:
            raise LocalModelResourceError(
                "Operation recovery evidence could not be verified."
            ) from None
        if not verified:
            raise LocalModelResourceError(
                "Operation recovery evidence could not be verified."
            )
        with self._lock, self._transaction() as connection:
            latest = self._select_record_locked(connection, operation_id)
            if latest != current:
                raise LocalModelOperationError(
                    "Remote operation changed during recovery."
                )
            recovered = RemoteOperationRecord(
                operation_id=current.operation_id,
                model_id=current.model_id,
                operation_type=current.operation_type,
                send_phase=(
                    RemoteOperationSendPhase.RESPONSE_RECEIVED
                    if evidence.kind
                    is RemoteOperationEvidenceKind.CORRELATED_TERMINAL_RESPONSE
                    else current.send_phase
                ),
                outcome=RemoteOperationOutcome.RECOVERED,
                completion_evidence=evidence.kind,
            )
            connection.execute(
                "UPDATE remote_operations SET send_phase = ?, outcome = ?, "
                "completion_evidence = ? WHERE operation_id = ? AND outcome = ?",
                (
                    recovered.send_phase.value,
                    recovered.outcome.value,
                    recovered.completion_evidence.value,
                    operation_id,
                    RemoteOperationOutcome.UNCERTAIN.value,
                ),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                raise LocalModelOperationError(
                    "Remote operation changed during recovery."
                )
        with self._lock:
            self._records[operation_id] = recovered
        return recovered

    def reserve_residency_estimate(
        self,
        estimate: ModelResidencyEstimate,
    ) -> None:
        if type(estimate) is not ModelResidencyEstimate:
            raise TypeError("A validated model residency estimate is required.")
        with self._lock:
            if estimate.model_id in self._residency_estimates:
                raise LocalModelOperationError(
                    "Model residency estimate is already held."
                )
            self._residency_estimates[estimate.model_id] = estimate

    def discard_residency_estimate(self, model_id: str) -> ModelResidencyEstimate:
        """Discard bookkeeping only; this does not confirm unload or operation end."""
        if type(model_id) is not str:
            raise TypeError("Model identifier must be text.")
        with self._lock:
            estimate = self._residency_estimates.pop(model_id, None)
            if estimate is None:
                raise LocalModelOperationError(
                    "Model residency estimate is not held."
                )
            return estimate

    def residency_estimate(self, model_id: str) -> ModelResidencyEstimate | None:
        if type(model_id) is not str:
            raise TypeError("Model identifier must be text.")
        with self._lock:
            return self._residency_estimates.get(model_id)

    def _transition(
        self,
        operation_id: str,
        *,
        expected: RemoteOperationOutcome,
        send_phase: RemoteOperationSendPhase,
        outcome: RemoteOperationOutcome,
        evidence: RemoteOperationEvidenceKind | None = None,
    ) -> RemoteOperationRecord:
        if type(operation_id) is not str:
            raise TypeError("Local operation identifier must be text.")
        with self._lock, self._transaction() as connection:
            current = self._select_record_locked(connection, operation_id)
            if current is None:
                raise LocalModelUnavailableError("Local operation is unavailable.")
            if current.outcome is not expected:
                raise LocalModelOperationError(
                    "Remote operation transition is invalid."
                )
            updated = RemoteOperationRecord(
                operation_id=current.operation_id,
                model_id=current.model_id,
                operation_type=current.operation_type,
                send_phase=send_phase,
                outcome=outcome,
                completion_evidence=evidence,
            )
            connection.execute(
                "UPDATE remote_operations SET send_phase = ?, outcome = ?, "
                "completion_evidence = ? WHERE operation_id = ? AND outcome = ?",
                (
                    updated.send_phase.value,
                    updated.outcome.value,
                    None if evidence is None else evidence.value,
                    operation_id,
                    expected.value,
                ),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                raise LocalModelOperationError(
                    "Remote operation transition is invalid."
                )
        with self._lock:
            self._records[operation_id] = updated
        return updated

    def _initialize_database(self, *, allow_initialization: bool) -> None:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                version = self._connection.execute("PRAGMA user_version").fetchone()[0]
                tables = {
                    row[0]
                    for row in self._connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' "
                        "AND name NOT LIKE 'sqlite_%'"
                    )
                }
                if version == 0 and not tables and allow_initialization:
                    self._connection.execute(
                        "CREATE TABLE ledger_config ("
                        "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                        "max_concurrent_operations INTEGER NOT NULL, "
                        "max_total_records INTEGER NOT NULL)"
                    )
                    self._connection.execute(
                        "CREATE TABLE remote_operations ("
                        "operation_id TEXT PRIMARY KEY NOT NULL, "
                        "model_id TEXT NOT NULL, "
                        "operation_type TEXT NOT NULL, "
                        "send_phase TEXT NOT NULL, "
                        "outcome TEXT NOT NULL, "
                        "completion_evidence TEXT)"
                    )
                    self._connection.execute(
                        "INSERT INTO ledger_config VALUES (1, ?, ?)",
                        (self._max_concurrent_operations, self._max_total_records),
                    )
                    self._connection.execute("PRAGMA user_version = 1")
                elif version != 1 or tables != {
                    "ledger_config",
                    "remote_operations",
                }:
                    raise LocalModelResourceError(
                        "Persistent operation ledger schema is incompatible."
                    )
                self._validate_database_locked()
                self._connection.commit()
            except LocalModelError:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise
            except Exception:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise LocalModelResourceError(
                    "Persistent operation ledger is unavailable."
                ) from None

    def _validate_database_locked(self) -> None:
        result = self._connection.execute("PRAGMA integrity_check").fetchone()
        if result is None or result[0] != "ok":
            raise LocalModelResourceError(
                "Persistent operation ledger integrity check failed."
            )
        config = self._connection.execute(
            "SELECT max_concurrent_operations, max_total_records "
            "FROM ledger_config WHERE singleton = 1"
        ).fetchall()
        if (
            len(config) != 1
            or type(config[0][0]) is not int
            or type(config[0][1]) is not int
            or config[0]
            != (self._max_concurrent_operations, self._max_total_records)
        ):
            raise LocalModelResourceError(
                "Persistent operation ledger configuration is incompatible."
            )
        expected_columns = {
            "ledger_config": (
                ("singleton", "INTEGER", 0, 1),
                ("max_concurrent_operations", "INTEGER", 1, 0),
                ("max_total_records", "INTEGER", 1, 0),
            ),
            "remote_operations": (
                ("operation_id", "TEXT", 1, 1),
                ("model_id", "TEXT", 1, 0),
                ("operation_type", "TEXT", 1, 0),
                ("send_phase", "TEXT", 1, 0),
                ("outcome", "TEXT", 1, 0),
                ("completion_evidence", "TEXT", 0, 0),
            ),
        }
        for table, expected in expected_columns.items():
            columns = tuple(
                (row[1], row[2].upper(), row[3], row[5])
                for row in self._connection.execute(
                    f"PRAGMA table_info({table})"
                )
            )
            if columns != expected:
                raise LocalModelResourceError(
                    "Persistent operation ledger schema is incompatible."
                )
        self._refresh_cache_locked()
        if len(self._records) > self._max_total_records:
            raise LocalModelResourceError(
                "Persistent operation ledger exceeds its configured history limit."
            )
        if self._active_count_locked() > self._max_concurrent_operations:
            raise LocalModelResourceError(
                "Persistent operation ledger exceeds its concurrency limit."
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._ensure_open()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield self._connection
            self._connection.commit()
        except LocalModelError:
            if self._connection.in_transaction:
                self._connection.rollback()
            raise
        except Exception:
            if self._connection.in_transaction:
                self._connection.rollback()
            raise LocalModelResourceError(
                "Persistent operation ledger transaction failed."
            ) from None

    def _ensure_open(self) -> None:
        if self._closed:
            raise LocalModelResourceError(
                "Persistent operation ledger is unavailable."
            )

    @staticmethod
    def _record_from_row(row: tuple[object, ...]) -> RemoteOperationRecord:
        try:
            return RemoteOperationRecord(
                operation_id=row[0],
                model_id=row[1],
                operation_type=RemoteOperationType(row[2]),
                send_phase=RemoteOperationSendPhase(row[3]),
                outcome=RemoteOperationOutcome(row[4]),
                completion_evidence=(
                    None
                    if row[5] is None
                    else RemoteOperationEvidenceKind(row[5])
                ),
            )
        except Exception:
            raise LocalModelResourceError(
                "Persistent operation ledger contains invalid records."
            ) from None

    def _select_record_locked(
        self,
        connection: sqlite3.Connection,
        operation_id: str,
    ) -> RemoteOperationRecord | None:
        row = connection.execute(
            "SELECT operation_id, model_id, operation_type, send_phase, outcome, "
            "completion_evidence FROM remote_operations WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        return None if row is None else self._record_from_row(row)

    def _refresh_cache_locked(self) -> None:
        self._ensure_open()
        try:
            rows = self._connection.execute(
                "SELECT operation_id, model_id, operation_type, send_phase, outcome, "
                "completion_evidence FROM remote_operations"
            ).fetchall()
        except sqlite3.Error:
            raise LocalModelResourceError(
                "Persistent operation ledger could not be read."
            ) from None
        refreshed: dict[str, RemoteOperationRecord] = {}
        for row in rows:
            record = self._record_from_row(row)
            cached = self._records.get(record.operation_id)
            refreshed[record.operation_id] = (
                cached if cached == record else record
            )
        self._records = refreshed

    def _record_count_locked(self, connection: sqlite3.Connection) -> int:
        return connection.execute(
            "SELECT COUNT(*) FROM remote_operations"
        ).fetchone()[0]

    def _active_count_locked(
        self,
        connection: sqlite3.Connection | None = None,
    ) -> int:
        selected = self._connection if connection is None else connection
        self._ensure_open()
        try:
            return selected.execute(
                "SELECT COUNT(*) FROM remote_operations WHERE outcome IN (?, ?, ?)",
                (
                    RemoteOperationOutcome.RESERVED.value,
                    RemoteOperationOutcome.PENDING.value,
                    RemoteOperationOutcome.UNCERTAIN.value,
                ),
            ).fetchone()[0]
        except sqlite3.Error:
            raise LocalModelResourceError(
                "Persistent operation ledger could not be read."
            ) from None

    @staticmethod
    def _holds_activity_slot(record: RemoteOperationRecord) -> bool:
        return record.outcome in {
            RemoteOperationOutcome.RESERVED,
            RemoteOperationOutcome.PENDING,
            RemoteOperationOutcome.UNCERTAIN,
        }


@dataclass(frozen=True, slots=True)
class LocalModelDescriptor:
    model_id: str
    tasks: frozenset[LocalModelTask]
    max_context_tokens: int
    requirements: ModelRequirements
    availability: ModelAvailability = ModelAvailability.AVAILABLE
    catalog_entry_id: str | None = None
    tokenizer_model_id: str | None = None
    tokenizer_id: str | None = None

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
        for value, field_name in (
            (self.catalog_entry_id, "catalog entry"),
            (self.tokenizer_model_id, "tokenizer model"),
            (self.tokenizer_id, "tokenizer"),
        ):
            if value is not None and (
                type(value) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}", value)
                is None
            ):
                raise ValueError(f"Model {field_name} identifier is invalid.")


@dataclass(frozen=True, slots=True, repr=False)
class ResourceSnapshot:
    source: ResourceSource
    available_ram_bytes: int | None
    available_vram_bytes: int | None = None
    system_load: float | None = None
    captured_at_monotonic: float = field(default_factory=time.monotonic)
    total_ram_bytes: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if type(self.source) is not ResourceSource:
            raise TypeError("Resource measurement source is invalid.")
        for value in (
            self.total_ram_bytes,
            self.available_ram_bytes,
            self.available_vram_bytes,
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("Known resource amounts must be positive integers.")
        if (
            self.total_ram_bytes is not None
            and self.available_ram_bytes is not None
            and self.available_ram_bytes > self.total_ram_bytes
        ):
            raise ValueError("Available RAM cannot exceed total RAM.")
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
                self.total_ram_bytes,
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


class _MemoryStatusEx(ctypes.Structure):
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


class WindowsResourceProvider:
    """Measures system-wide physical RAM on Windows; it reserves no memory."""

    def snapshot(self) -> ResourceSnapshot:
        if os.name != "nt":
            raise LocalModelResourceError(
                "Windows RAM measurement is unavailable."
            )
        try:
            win_dll = getattr(ctypes, "WinDLL")
            kernel32 = win_dll("kernel32", use_last_error=True)
            measure_memory = kernel32.GlobalMemoryStatusEx
            measure_memory.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
            measure_memory.restype = ctypes.c_int
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if not measure_memory(ctypes.byref(status)):
                raise LocalModelResourceError(
                    "Windows RAM measurement failed."
                )
            return ResourceSnapshot(
                source=ResourceSource.OPERATING_SYSTEM,
                available_ram_bytes=status.ullAvailPhys,
                total_ram_bytes=status.ullTotalPhys,
                captured_at_monotonic=time.monotonic(),
            )
        except LocalModelResourceError:
            raise
        except Exception:
            raise LocalModelResourceError(
                "Windows RAM measurement is unavailable."
            ) from None


class ContextTokenCounter(Protocol):
    tokenizer_id: str

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

    @property
    def is_production_ready(self) -> bool:
        """Production composition stays closed until trusted dependencies exist.

        Readiness attributes on injected objects are claims, not evidence.
        There is currently no validated VRAM provider or production tokenizer.
        """
        return False

    def register(self, descriptor: LocalModelDescriptor) -> None:
        if type(descriptor) is not LocalModelDescriptor:
            raise TypeError("Only validated local model descriptors can be registered.")
        descriptor_validator = getattr(self._backend, "validate_descriptor", None)
        if descriptor_validator is not None:
            if not callable(descriptor_validator):
                raise TypeError("Local model backend descriptor validation is invalid.")
            descriptor_validator(descriptor)
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
        if (
            not self._allow_simulated_resources_for_tests
            and not self.is_production_ready
        ):
            raise LocalModelResourceError(
                "Production local model dependencies are unavailable."
            )
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
            if entry.descriptor.tokenizer_id is not None and getattr(
                self._token_counter,
                "tokenizer_id",
                None,
            ) != entry.descriptor.tokenizer_id:
                raise LocalModelResourceError(
                    "Configured tokenizer does not match the model catalog."
                )
            try:
                context_tokens = self._token_counter.count_tokens(
                    entry.descriptor.tokenizer_model_id
                    or entry.descriptor.model_id,
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
        if snapshot.total_ram_bytes is None:
            raise LocalModelResourceError(
                "Total RAM could not be determined."
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
        with self._lock:
            if (
                entry.handle is not None
                or not entry.resource_reservation_held
                or entry.active
                or entry.state
                not in {ModelLifecycleState.ERROR, ModelLifecycleState.UNLOADED}
            ):
                raise LocalModelBusyError(
                    "Local model state changed during cleanup confirmation."
                )
            entry.state = ModelLifecycleState.UNLOADING
            entry.active = True

        released = False
        try:
            released = (
                self._backend.failed_load_released(entry.descriptor.model_id)
                is True
            )
        except Exception:
            released = False
        finally:
            with self._lock:
                if (
                    entry.handle is not None
                    or entry.state is not ModelLifecycleState.UNLOADING
                    or not entry.active
                ):
                    raise LocalModelBusyError(
                        "Local model state changed during cleanup confirmation."
                    )
                entry.active = False
                if released:
                    entry.resource_reservation_held = False
                    entry.state = ModelLifecycleState.UNLOADED
                else:
                    entry.state = ModelLifecycleState.ERROR
        if not released:
            raise LocalModelResourceError(
                "Local model resource state is uncertain."
            )

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
                entry.state = ModelLifecycleState.ERROR
            raise LocalModelResourceError(
                "Local model resource state is uncertain."
            )
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
