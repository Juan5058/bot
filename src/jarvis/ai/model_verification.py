from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Protocol

from jarvis.ai.model_catalog import ArtifactIdentity


_SAFE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_RESOURCE_EVIDENCE_AGE_SECONDS = 5.0


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    INDETERMINATE = "indeterminate"


class VerificationCheckId(str, Enum):
    ARTIFACT_IDENTITY = "artifact_identity"
    TRUSTED_PROVENANCE = "trusted_provenance"
    ARTIFACT_INTEGRITY = "artifact_integrity"
    RUNTIME_BINDING = "runtime_binding"
    ARCHITECTURE_AND_FORMAT = "architecture_and_format"
    RUNTIME_COMPATIBILITY = "runtime_compatibility"
    TOKENIZER_COMPATIBILITY = "tokenizer_compatibility"
    RAM_RESOURCES = "ram_resources"
    VRAM_RESOURCES = "vram_resources"
    EXECUTION_MODE = "execution_mode"


class VerificationReasonCode(str, Enum):
    VERIFIED = "verified"
    MISMATCH = "mismatch"
    UNTRUSTED_SOURCE = "untrusted_source"
    INCOMPATIBLE = "incompatible"
    INSUFFICIENT_RESOURCES = "insufficient_resources"
    EVIDENCE_MISSING = "evidence_missing"
    EVIDENCE_EXPIRED = "evidence_expired"
    VERIFIER_ERROR = "verifier_error"
    UNKNOWN = "unknown"


class AdmissionStatus(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"
    INDETERMINATE = "indeterminate"


class AdmissionReasonCode(str, Enum):
    ALL_CHECKS_VERIFIED = "all_checks_verified"
    CHECK_REJECTED = "check_rejected"
    CHECK_INDETERMINATE = "check_indeterminate"
    CHECKS_MISSING = "checks_missing"
    DUPLICATE_CHECKS = "duplicate_checks"
    CONTRADICTORY_CHECKS = "contradictory_checks"
    REPORT_EXPIRED = "report_expired"
    REPORT_TIME_INVALID = "report_time_invalid"
    EVIDENCE_MISSING = "evidence_missing"
    EVIDENCE_EXPIRED = "evidence_expired"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    AUTHORITY_UNAVAILABLE = "authority_unavailable"
    EVIDENCE_REJECTED = "evidence_rejected"
    AUTHORITY_ERROR = "authority_error"
    AUTHORITY_INVALID_RESULT = "authority_invalid_result"


REQUIRED_VERIFICATION_CHECKS = frozenset(VerificationCheckId)


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if (
        type(value) is not datetime
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be a timezone-aware datetime.")


@dataclass(frozen=True, slots=True, repr=False)
class VerificationEvidenceReference:
    """Non-secret pointer to evidence, not evidence authentication by itself."""

    source_id: str
    reference_id: str
    observed_at: datetime
    expires_at: datetime
    subject_digest: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("source", self.source_id),
            ("reference", self.reference_id),
        ):
            if (
                type(value) is not str
                or _SAFE_IDENTIFIER_PATTERN.fullmatch(value) is None
            ):
                raise ValueError(f"Evidence {name} identifier is invalid.")
        _require_aware_datetime(self.observed_at, "Evidence observed_at")
        _require_aware_datetime(self.expires_at, "Evidence expires_at")
        if self.expires_at <= self.observed_at:
            raise ValueError("Evidence expiration must follow observation.")
        if self.subject_digest is not None and (
            type(self.subject_digest) is not str
            or _SHA256_PATTERN.fullmatch(self.subject_digest) is None
        ):
            raise ValueError("Evidence subject digest is invalid.")

    def __repr__(self) -> str:
        return "VerificationEvidenceReference(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class VerificationCheck:
    check_id: VerificationCheckId
    status: VerificationStatus
    reason: VerificationReasonCode
    evidence: VerificationEvidenceReference | None = None

    def __post_init__(self) -> None:
        if type(self.check_id) is not VerificationCheckId:
            raise TypeError("Verification check identifier is invalid.")
        if type(self.status) is not VerificationStatus:
            raise TypeError("Verification status is invalid.")
        if type(self.reason) is not VerificationReasonCode:
            raise TypeError("Verification reason is invalid.")
        if self.evidence is not None and (
            type(self.evidence) is not VerificationEvidenceReference
        ):
            raise TypeError("Verification evidence reference is invalid.")
        if (
            self.status is VerificationStatus.VERIFIED
        ) != (self.reason is VerificationReasonCode.VERIFIED):
            raise ValueError("Verification status and reason do not match.")

    def __repr__(self) -> str:
        return (
            f"VerificationCheck(check_id={self.check_id.value!r}, "
            f"status={self.status.value!r}, <redacted>)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ModelVerificationReport:
    """Bounded verification summary; claims require trusted authority validation."""

    report_id: str
    identity: ArtifactIdentity
    runtime_version: str
    checked_at: datetime
    expires_at: datetime
    checks: tuple[VerificationCheck, ...]

    def __post_init__(self) -> None:
        if (
            type(self.report_id) is not str
            or _SAFE_IDENTIFIER_PATTERN.fullmatch(self.report_id) is None
        ):
            raise ValueError("Verification report identifier is invalid.")
        if type(self.identity) is not ArtifactIdentity:
            raise TypeError("Verification report identity is invalid.")
        if (
            type(self.runtime_version) is not str
            or re.fullmatch(
                r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)",
                self.runtime_version,
            )
            is None
        ):
            raise ValueError("Verification runtime version is invalid.")
        _require_aware_datetime(self.checked_at, "Report checked_at")
        _require_aware_datetime(self.expires_at, "Report expires_at")
        if self.expires_at <= self.checked_at:
            raise ValueError("Report expiration must follow verification.")
        if type(self.checks) is not tuple or any(
            type(check) is not VerificationCheck for check in self.checks
        ):
            raise TypeError("Verification report checks are invalid.")

    def __repr__(self) -> str:
        return f"ModelVerificationReport(report_id={self.report_id!r}, <redacted>)"


class TrustedVerificationEvidenceAuthority(Protocol):
    """Trust boundary for authenticating report evidence and its provenance.

    Implementations must validate the evidence against independently configured
    trust roots and bind artifact/runtime evidence to the loaded model. A report
    or a VERIFIED status is never sufficient without this authority.
    """

    def validate(self, report: ModelVerificationReport) -> bool:
        ...


@dataclass(frozen=True, slots=True, init=False, repr=False)
class ProductionAdmissionDecision:
    """Policy result only; not an authorization or a production credential."""

    status: AdmissionStatus
    reason: AdmissionReasonCode
    report_id: str | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Production admission decisions are policy-generated.")

    @property
    def admitted(self) -> bool:
        return self.status is AdmissionStatus.ADMITTED

    def __repr__(self) -> str:
        return (
            f"ProductionAdmissionDecision(status={self.status.value!r}, "
            f"reason={self.reason.value!r}, <redacted>)"
        )


def _decision(
    status: AdmissionStatus,
    reason: AdmissionReasonCode,
    report_id: str | None,
) -> ProductionAdmissionDecision:
    result = object.__new__(ProductionAdmissionDecision)
    object.__setattr__(result, "status", status)
    object.__setattr__(result, "reason", reason)
    object.__setattr__(result, "report_id", report_id)
    return result


class ProductionAdmissionPolicy:
    """Fail-closed report evaluator; not currently wired into app composition."""

    __slots__ = ()

    def evaluate(
        self,
        report: ModelVerificationReport,
        *,
        evidence_authority: TrustedVerificationEvidenceAuthority | None,
        now: datetime | None = None,
    ) -> ProductionAdmissionDecision:
        report_id = (
            report.report_id
            if type(report) is ModelVerificationReport
            else None
        )
        if type(report) is not ModelVerificationReport:
            return _decision(
                AdmissionStatus.INDETERMINATE,
                AdmissionReasonCode.CHECK_INDETERMINATE,
                None,
            )
        evaluation_time = datetime.now(timezone.utc) if now is None else now
        try:
            _require_aware_datetime(evaluation_time, "Evaluation time")
        except ValueError:
            return _decision(
                AdmissionStatus.INDETERMINATE,
                AdmissionReasonCode.REPORT_TIME_INVALID,
                report_id,
            )
        if report.checked_at > evaluation_time:
            return _decision(
                AdmissionStatus.INDETERMINATE,
                AdmissionReasonCode.REPORT_TIME_INVALID,
                report_id,
            )
        if report.expires_at <= evaluation_time:
            return _decision(
                AdmissionStatus.INDETERMINATE,
                AdmissionReasonCode.REPORT_EXPIRED,
                report_id,
            )

        checks_by_id: dict[VerificationCheckId, VerificationCheck] = {}
        for check in report.checks:
            previous = checks_by_id.get(check.check_id)
            if previous is not None:
                reason = (
                    AdmissionReasonCode.CONTRADICTORY_CHECKS
                    if previous != check
                    else AdmissionReasonCode.DUPLICATE_CHECKS
                )
                return _decision(AdmissionStatus.REJECTED, reason, report_id)
            checks_by_id[check.check_id] = check

        if not REQUIRED_VERIFICATION_CHECKS.issubset(checks_by_id):
            return _decision(
                AdmissionStatus.INDETERMINATE,
                AdmissionReasonCode.CHECKS_MISSING,
                report_id,
            )

        for check_id in REQUIRED_VERIFICATION_CHECKS:
            check = checks_by_id[check_id]
            if check.status is VerificationStatus.REJECTED:
                return _decision(
                    AdmissionStatus.REJECTED,
                    AdmissionReasonCode.CHECK_REJECTED,
                    report_id,
                )
            if check.status is not VerificationStatus.VERIFIED:
                return _decision(
                    AdmissionStatus.INDETERMINATE,
                    AdmissionReasonCode.CHECK_INDETERMINATE,
                    report_id,
                )
            evidence = check.evidence
            if evidence is None:
                return _decision(
                    AdmissionStatus.INDETERMINATE,
                    AdmissionReasonCode.EVIDENCE_MISSING,
                    report_id,
                )
            if evidence.observed_at > report.checked_at:
                return _decision(
                    AdmissionStatus.INDETERMINATE,
                    AdmissionReasonCode.REPORT_TIME_INVALID,
                    report_id,
                )
            if evidence.expires_at <= evaluation_time:
                return _decision(
                    AdmissionStatus.INDETERMINATE,
                    AdmissionReasonCode.EVIDENCE_EXPIRED,
                    report_id,
                )
            if check_id in {
                VerificationCheckId.ARTIFACT_IDENTITY,
                VerificationCheckId.ARTIFACT_INTEGRITY,
                VerificationCheckId.RUNTIME_BINDING,
            }:
                if (
                    report.identity.digest is None
                    or evidence.subject_digest is None
                ):
                    return _decision(
                        AdmissionStatus.INDETERMINATE,
                        AdmissionReasonCode.EVIDENCE_MISSING,
                        report_id,
                    )
                if evidence.subject_digest != report.identity.digest:
                    return _decision(
                        AdmissionStatus.REJECTED,
                        AdmissionReasonCode.EVIDENCE_MISMATCH,
                        report_id,
                    )
            if check_id in {
                VerificationCheckId.RAM_RESOURCES,
                VerificationCheckId.VRAM_RESOURCES,
            } and (
                evaluation_time - evidence.observed_at
            ).total_seconds() > _MAX_RESOURCE_EVIDENCE_AGE_SECONDS:
                return _decision(
                    AdmissionStatus.INDETERMINATE,
                    AdmissionReasonCode.EVIDENCE_EXPIRED,
                    report_id,
                )

        try:
            has_authority = evidence_authority is not None and callable(
                getattr(evidence_authority, "validate", None)
            )
        except Exception:
            has_authority = False
        if not has_authority:
            return _decision(
                AdmissionStatus.INDETERMINATE,
                AdmissionReasonCode.AUTHORITY_UNAVAILABLE,
                report_id,
            )
        try:
            validated = evidence_authority.validate(report)
        except Exception:
            return _decision(
                AdmissionStatus.INDETERMINATE,
                AdmissionReasonCode.AUTHORITY_ERROR,
                report_id,
            )
        if validated is False:
            return _decision(
                AdmissionStatus.REJECTED,
                AdmissionReasonCode.EVIDENCE_REJECTED,
                report_id,
            )
        if validated is not True:
            return _decision(
                AdmissionStatus.INDETERMINATE,
                AdmissionReasonCode.AUTHORITY_INVALID_RESULT,
                report_id,
            )
        return _decision(
            AdmissionStatus.ADMITTED,
            AdmissionReasonCode.ALL_CHECKS_VERIFIED,
            report_id,
        )
