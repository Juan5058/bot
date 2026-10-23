from datetime import datetime, timedelta, timezone

import pytest

from jarvis.ai.local_models import (
    LocalModelDescriptor,
    LocalModelLimits,
    LocalModelManager,
    LocalModelTask,
    ModelRequirements,
    ResourceSnapshot,
    ResourceSource,
)
from jarvis.ai.model_catalog import ArtifactIdentity, ModelRuntime
from jarvis.ai.model_verification import (
    AdmissionReasonCode,
    AdmissionStatus,
    ModelVerificationReport,
    ProductionAdmissionDecision,
    ProductionAdmissionPolicy,
    REQUIRED_VERIFICATION_CHECKS,
    VerificationCheck,
    VerificationCheckId,
    VerificationEvidenceReference,
    VerificationReasonCode,
    VerificationStatus,
)
from jarvis.app import create_application
from jarvis.config.settings import Settings


NOW = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
_DIGEST = "sha256:" + "a" * 64


class SyntheticEvidenceAuthority:
    """Test double for policy logic only; never authenticates real model bytes."""

    def __init__(self, outcome: object = True, error: Exception | None = None) -> None:
        self.outcome = outcome
        self.error = error
        self.calls = 0

    def validate(self, report: ModelVerificationReport) -> bool:
        del report
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.outcome


def evidence(
    *,
    observed_at: datetime = NOW - timedelta(seconds=1),
    expires_at: datetime = NOW + timedelta(minutes=5),
    reference_id: str = "fixture-reference",
    subject_digest: str | None = _DIGEST,
) -> VerificationEvidenceReference:
    return VerificationEvidenceReference(
        source_id="test-trust-source",
        reference_id=reference_id,
        observed_at=observed_at,
        expires_at=expires_at,
        subject_digest=subject_digest,
    )


def check(
    check_id: VerificationCheckId,
    status: VerificationStatus = VerificationStatus.VERIFIED,
    *,
    evidence_reference: VerificationEvidenceReference | None = None,
) -> VerificationCheck:
    reason = {
        VerificationStatus.VERIFIED: VerificationReasonCode.VERIFIED,
        VerificationStatus.REJECTED: VerificationReasonCode.MISMATCH,
        VerificationStatus.INDETERMINATE: VerificationReasonCode.UNKNOWN,
    }[status]
    return VerificationCheck(
        check_id=check_id,
        status=status,
        reason=reason,
        evidence=evidence_reference if evidence_reference is not None else evidence(),
    )


def report(
    checks: tuple[VerificationCheck, ...] | None = None,
    *,
    checked_at: datetime = NOW,
    expires_at: datetime = NOW + timedelta(minutes=10),
) -> ModelVerificationReport:
    return ModelVerificationReport(
        report_id="verification-fixture-1",
        identity=ArtifactIdentity(
            runtime=ModelRuntime.OLLAMA,
            artifact_id="fixture/model",
            variant="fixture-v1",
            quantization="Q4_K_M",
            digest=_DIGEST,
        ),
        runtime_version="0.40.6",
        checked_at=checked_at,
        expires_at=expires_at,
        checks=(
            tuple(check(check_id) for check_id in REQUIRED_VERIFICATION_CHECKS)
            if checks is None
            else checks
        ),
    )


def evaluate(
    candidate: ModelVerificationReport,
    authority: SyntheticEvidenceAuthority | None = None,
    *,
    now: datetime = NOW,
):
    return ProductionAdmissionPolicy().evaluate(
        candidate,
        evidence_authority=authority,
        now=now,
    )


def test_all_required_checks_with_current_evidence_are_admitted_by_policy_logic():
    authority = SyntheticEvidenceAuthority()

    decision = evaluate(report(), authority)

    assert decision.status is AdmissionStatus.ADMITTED
    assert decision.reason is AdmissionReasonCode.ALL_CHECKS_VERIFIED
    assert decision.admitted is True
    assert authority.calls == 1
    # The synthetic authority tests the decision algorithm, not authenticity.


@pytest.mark.parametrize("check_id", tuple(REQUIRED_VERIFICATION_CHECKS))
def test_each_rejected_required_check_denies_admission(check_id):
    authority = SyntheticEvidenceAuthority()
    checks = tuple(
        check(
            item,
            VerificationStatus.REJECTED
            if item is check_id
            else VerificationStatus.VERIFIED,
        )
        for item in REQUIRED_VERIFICATION_CHECKS
    )

    decision = evaluate(report(checks), authority)

    assert decision.status is AdmissionStatus.REJECTED
    assert decision.reason is AdmissionReasonCode.CHECK_REJECTED
    assert authority.calls == 0


@pytest.mark.parametrize("check_id", tuple(REQUIRED_VERIFICATION_CHECKS))
def test_each_indeterminate_required_check_prevents_admission(check_id):
    authority = SyntheticEvidenceAuthority()
    checks = tuple(
        check(
            item,
            VerificationStatus.INDETERMINATE
            if item is check_id
            else VerificationStatus.VERIFIED,
        )
        for item in REQUIRED_VERIFICATION_CHECKS
    )

    decision = evaluate(report(checks), authority)

    assert decision.status is AdmissionStatus.INDETERMINATE
    assert decision.reason is AdmissionReasonCode.CHECK_INDETERMINATE
    assert authority.calls == 0


def test_missing_duplicate_and_contradictory_checks_fail_closed():
    all_checks = tuple(check(check_id) for check_id in REQUIRED_VERIFICATION_CHECKS)
    changed_evidence_check = VerificationCheck(
        all_checks[0].check_id,
        all_checks[0].status,
        all_checks[0].reason,
        evidence(reference_id="different-fixture-reference"),
    )

    missing = evaluate(report(all_checks[:-1]), SyntheticEvidenceAuthority())
    duplicate = evaluate(
        report(all_checks + (all_checks[0],)),
        SyntheticEvidenceAuthority(),
    )
    contradictory = evaluate(
        report(
            all_checks
            + (changed_evidence_check,)
        ),
        SyntheticEvidenceAuthority(),
    )

    assert missing.status is AdmissionStatus.INDETERMINATE
    assert missing.reason is AdmissionReasonCode.CHECKS_MISSING
    assert duplicate.status is AdmissionStatus.REJECTED
    assert duplicate.reason is AdmissionReasonCode.DUPLICATE_CHECKS
    assert contradictory.status is AdmissionStatus.REJECTED
    assert contradictory.reason is AdmissionReasonCode.CONTRADICTORY_CHECKS

    status_conflict = evaluate(
        report(
            all_checks
            + (check(all_checks[0].check_id, VerificationStatus.REJECTED),)
        ),
        SyntheticEvidenceAuthority(),
    )
    assert status_conflict.reason is AdmissionReasonCode.CONTRADICTORY_CHECKS


def test_missing_expired_or_future_evidence_prevents_admission():
    all_checks = tuple(check(check_id) for check_id in REQUIRED_VERIFICATION_CHECKS)
    target = all_checks[0].check_id

    missing_evidence = tuple(
        VerificationCheck(item.check_id, item.status, item.reason, None)
        if item.check_id is target
        else item
        for item in all_checks
    )
    expired_evidence = tuple(
        VerificationCheck(
            item.check_id,
            item.status,
            item.reason,
            evidence(expires_at=NOW),
        )
        if item.check_id is target
        else item
        for item in all_checks
    )
    future_evidence = tuple(
        VerificationCheck(
            item.check_id,
            item.status,
            item.reason,
            evidence(observed_at=NOW + timedelta(seconds=1)),
        )
        if item.check_id is target
        else item
        for item in all_checks
    )
    stale_resource_evidence = tuple(
        VerificationCheck(
            item.check_id,
            item.status,
            item.reason,
            evidence(observed_at=NOW - timedelta(seconds=6)),
        )
        if item.check_id is VerificationCheckId.RAM_RESOURCES
        else item
        for item in all_checks
    )

    assert evaluate(report(missing_evidence), SyntheticEvidenceAuthority()).reason is (
        AdmissionReasonCode.EVIDENCE_MISSING
    )
    assert evaluate(report(expired_evidence), SyntheticEvidenceAuthority()).reason is (
        AdmissionReasonCode.EVIDENCE_EXPIRED
    )
    assert evaluate(report(future_evidence), SyntheticEvidenceAuthority()).reason is (
        AdmissionReasonCode.REPORT_TIME_INVALID
    )
    assert evaluate(
        report(stale_resource_evidence),
        SyntheticEvidenceAuthority(),
    ).reason is AdmissionReasonCode.EVIDENCE_EXPIRED
    assert evaluate(
        report(expires_at=NOW + timedelta(minutes=1)),
        SyntheticEvidenceAuthority(),
        now=NOW + timedelta(minutes=1),
    ).reason is AdmissionReasonCode.REPORT_EXPIRED


def test_evidence_authority_missing_rejecting_or_failing_fails_closed():
    candidate = report()

    assert evaluate(candidate).reason is AdmissionReasonCode.AUTHORITY_UNAVAILABLE
    assert evaluate(
        candidate,
        SyntheticEvidenceAuthority(outcome=False),
    ).status is AdmissionStatus.REJECTED
    invalid_result = evaluate(
        candidate,
        SyntheticEvidenceAuthority(outcome=1),
    )
    assert invalid_result.status is AdmissionStatus.INDETERMINATE
    assert invalid_result.reason is AdmissionReasonCode.AUTHORITY_INVALID_RESULT
    failed = evaluate(
        candidate,
        SyntheticEvidenceAuthority(error=RuntimeError("private detail")),
    )
    assert failed.status is AdmissionStatus.INDETERMINATE
    assert failed.reason is AdmissionReasonCode.AUTHORITY_ERROR


def test_artifact_evidence_must_match_report_digest():
    checks = tuple(
        VerificationCheck(
            item.check_id,
            item.status,
            item.reason,
            evidence(subject_digest="sha256:" + "b" * 64),
        )
        if item.check_id is VerificationCheckId.RUNTIME_BINDING
        else item
        for item in report().checks
    )

    decision = evaluate(report(checks), SyntheticEvidenceAuthority())

    assert decision.status is AdmissionStatus.REJECTED
    assert decision.reason is AdmissionReasonCode.EVIDENCE_MISMATCH


def test_report_and_decision_representations_do_not_expose_evidence_details():
    candidate = report()
    decision = evaluate(candidate, SyntheticEvidenceAuthority())

    assert "fixture-reference" not in repr(candidate)
    assert _DIGEST not in repr(candidate)
    assert "fixture-reference" not in repr(candidate.checks[0])
    with pytest.raises(TypeError, match="policy-generated"):
        ProductionAdmissionDecision(
            AdmissionStatus.ADMITTED,
            AdmissionReasonCode.ALL_CHECKS_VERIFIED,
            candidate.report_id,
        )
    assert decision.admitted is True


def test_synthetic_verified_report_cannot_enable_current_production_composition():
    class Backend:
        load_calls = 0
        infer_calls = 0

        def load(self, model_id):
            self.load_calls += 1
            return model_id

        def failed_load_released(self, model_id):
            del model_id
            return True

        def infer(self, handle, task, content, context_tokens, max_context_tokens):
            self.infer_calls += 1
            return "unexpected"

        def unload(self, handle):
            del handle

    class Resources:
        def snapshot(self):
            return ResourceSnapshot(
                ResourceSource.OPERATING_SYSTEM,
                available_ram_bytes=10_000,
                total_ram_bytes=20_000,
            )

    class Counter:
        tokenizer_id = "fixture-tokenizer"

        def count_tokens(self, model_id, task, content):
            del model_id, task, content
            return 1

    settings = Settings(
        ai_enabled=True,
        ai_provider="local",
        local_max_context_tokens=128,
        local_ram_budget_mb=1,
    )
    backend = Backend()
    manager = LocalModelManager(
        backend=backend,
        resource_provider=Resources(),
        token_counter=Counter(),
        limits=LocalModelLimits.from_settings(settings),
    )
    manager.register(
        LocalModelDescriptor(
            model_id="fixture-model",
            tasks=frozenset({LocalModelTask.ANSWER_QUESTION}),
            max_context_tokens=128,
            requirements=ModelRequirements(ram_bytes=100),
        )
    )
    synthetic_decision = evaluate(report(), SyntheticEvidenceAuthority())

    assert synthetic_decision.admitted is True
    assert manager.is_production_ready is False
    with pytest.raises(ValueError, match="not verified for production"):
        create_application(settings, local_model_manager=manager)
    assert "local" not in create_application(settings).provider_selector._providers
    assert backend.load_calls == 0
    assert backend.infer_calls == 0
