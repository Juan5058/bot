
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from jarvis.ai.model_catalog import (
    DEFAULT_MODEL_CATALOG,
    ArtifactIdentity,
    ModelCatalog,
    ModelCatalogEntry,
    ModelCatalogError,
    ModelRuntime,
    ResolvedModel,
    RequirementBasis,
    ResourceKind,
    ResourceRequirement,
    RuntimeCompatibility,
)


_SYNTHETIC_DIGEST = "sha256:" + "a" * 64


def _identity(
    *,
    artifact_id: str = "fixture/model",
    variant: str = "fixture-variant",
    quantization: str | None = "fixture-q",
    digest: str | None = _SYNTHETIC_DIGEST,
) -> ArtifactIdentity:
    return ArtifactIdentity(
        runtime=ModelRuntime.OLLAMA,
        artifact_id=artifact_id,
        variant=variant,
        quantization=quantization,
        digest=digest,
    )


def _requirement(
    resource: ResourceKind = ResourceKind.RAM,
    *,
    basis: RequirementBasis = RequirementBasis.ESTIMATE,
    bytes_required: int = 4096,
) -> ResourceRequirement:
    return ResourceRequirement(
        resource=resource,
        bytes_required=bytes_required,
        basis=basis,
        source="synthetic test fixture",
        method="deterministic unit-test value",
        conditions="test only; not a model measurement",
    )


def _entry(
    *,
    entry_id: str = "fixture-entry",
    identity: ArtifactIdentity | None = None,
    compatibility: RuntimeCompatibility | None = None,
    requirements: tuple[ResourceRequirement, ...] | None = None,
    required_resources: frozenset[ResourceKind] = frozenset(
        {ResourceKind.RAM}
    ),
    max_context_tokens: int | None = 256,
) -> ModelCatalogEntry:
    return ModelCatalogEntry(
        entry_id=entry_id,
        identity=identity or _identity(),
        compatibility=compatibility
        or RuntimeCompatibility(
            ModelRuntime.OLLAMA,
            "0.5.0",
            "0.5.9",
        ),
        requirements=(
            requirements
            if requirements is not None
            else (
                _requirement(
                    basis=RequirementBasis.CONSERVATIVE_BOUND,
                ),
            )
        ),
        required_resources=required_resources,
        max_context_tokens=max_context_tokens,
        tokenizer_id="fixture-tokenizer",
    )


class MatchingIdentityVerifier:
    """Test-only verifier returning explicit synthetic runtime metadata."""

    def __init__(self, observed: ArtifactIdentity) -> None:
        self.observed = observed
        self.calls: list[ArtifactIdentity] = []

    def verify(
        self,
        declared_identity: ArtifactIdentity,
    ) -> ArtifactIdentity:
        self.calls.append(declared_identity)
        return self.observed


@pytest.mark.parametrize(
    "basis",
    (
        RequirementBasis.MEASURED,
        RequirementBasis.ESTIMATE,
    ),
)
def test_required_resource_evidence_must_be_a_conservative_bound(
    basis: RequirementBasis,
) -> None:
    entry = _entry(
        requirements=(_requirement(basis=basis),),
    )

    assert not entry.is_strictly_eligible()


def test_conservative_bound_allows_strict_eligibility_without_guaranteeing_load(
) -> None:
    entry = _entry(
        requirements=(
            _requirement(
                basis=RequirementBasis.CONSERVATIVE_BOUND,
            ),
        ),
    )

    assert entry.is_strictly_eligible()
    assert entry.requirements[0].is_load_guarantee is False


@pytest.mark.parametrize(
    ("ram_basis", "vram_basis", "expected"),
    (
        (
            RequirementBasis.CONSERVATIVE_BOUND,
            RequirementBasis.CONSERVATIVE_BOUND,
            True,
        ),
        (
            RequirementBasis.ESTIMATE,
            RequirementBasis.CONSERVATIVE_BOUND,
            False,
        ),
        (
            RequirementBasis.CONSERVATIVE_BOUND,
            RequirementBasis.MEASURED,
            False,
        ),
    ),
)
def test_all_required_ram_and_vram_evidence_must_meet_policy(
    ram_basis: RequirementBasis,
    vram_basis: RequirementBasis,
    expected: bool,
) -> None:
    entry = _entry(
        requirements=(
            _requirement(ResourceKind.RAM, basis=ram_basis),
            _requirement(ResourceKind.VRAM, basis=vram_basis),
        ),
        required_resources=frozenset(
            {ResourceKind.RAM, ResourceKind.VRAM}
        ),
    )

    assert entry.is_strictly_eligible() is expected


def test_optional_resource_evidence_does_not_replace_required_ram_policy(
) -> None:
    entry = _entry(
        requirements=(
            _requirement(
                ResourceKind.RAM,
                basis=RequirementBasis.CONSERVATIVE_BOUND,
            ),
            _requirement(
                ResourceKind.VRAM,
                basis=RequirementBasis.ESTIMATE,
            ),
        ),
        required_resources=frozenset({ResourceKind.RAM}),
    )

    assert entry.is_strictly_eligible()


def test_insufficient_resource_evidence_blocks_strict_resolution() -> None:
    entry = _entry(
        requirements=(
            _requirement(basis=RequirementBasis.ESTIMATE),
        ),
    )
    verifier = MatchingIdentityVerifier(entry.identity)
    catalog = ModelCatalog(
        (entry,),
        identity_verifier=verifier,
    )

    with pytest.raises(
        ModelCatalogError,
        match="lacks identity, context, or required resource evidence",
    ):
        catalog.resolve(
            "fixture-entry",
            runtime_version="0.5.4",
        )

    # Rejection occurs before external identity verification.
    assert verifier.calls == []


def test_conservative_bound_provenance_fields_are_not_authentication(
) -> None:
    requirement = ResourceRequirement(
        resource=ResourceKind.RAM,
        bytes_required=4096,
        basis=RequirementBasis.CONSERVATIVE_BOUND,
        source="self-declared source",
        method="self-declared method",
        conditions="self-declared conditions",
    )

    # Descriptive strings do not authenticate the claim or prove
    # that the runtime has enough available memory.
    assert requirement.is_load_guarantee is False
    assert requirement.basis is RequirementBasis.CONSERVATIVE_BOUND


def test_catalog_resolves_exact_entry_after_independent_identity_verification(
) -> None:
    entry = _entry()
    verifier = MatchingIdentityVerifier(entry.identity)
    catalog = ModelCatalog(
        (entry,),
        identity_verifier=verifier,
    )

    resolved = catalog.resolve(
        "fixture-entry",
        runtime_version="0.5.4",
    )

    assert resolved.entry is entry
    assert resolved.runtime_version == "0.5.4"
    assert resolved.verified_identity == entry.identity
    assert verifier.calls == [entry.identity]
    assert catalog.entries == (entry,)


def test_catalog_rejects_context_outside_declared_configuration() -> None:
    entry = _entry(max_context_tokens=256)
    catalog = ModelCatalog(
        (entry,),
        identity_verifier=MatchingIdentityVerifier(entry.identity),
    )

    assert catalog.resolve(
        "fixture-entry",
        runtime_version="0.5.4",
        requested_context_tokens=128,
    ).entry is entry

    with pytest.raises(ModelCatalogError, match="context exceeds"):
        catalog.resolve(
            "fixture-entry",
            runtime_version="0.5.4",
            requested_context_tokens=257,
        )

    with pytest.raises(ValueError, match="positive integer"):
        catalog.resolve(
            "fixture-entry",
            runtime_version="0.5.4",
            requested_context_tokens=True,
        )


def test_unverified_declaration_is_inspectable_but_not_strictly_resolvable(
) -> None:
    entry = _entry()
    catalog = ModelCatalog((entry,))

    inspected = catalog.inspect("fixture-entry")

    assert inspected is entry
    assert not isinstance(inspected, ResolvedModel)
    assert not hasattr(inspected, "identity_verified")

    with pytest.raises(
        ModelCatalogError,
        match="Artifact identity is not verified",
    ):
        catalog.resolve(
            "fixture-entry",
            runtime_version="0.5.4",
        )


def test_resolve_rejects_non_strict_calls_explicitly() -> None:
    catalog = ModelCatalog((_entry(),))

    with pytest.raises(
        ModelCatalogError,
        match=r"Non-strict resolution.*inspect",
    ):
        catalog.resolve("fixture-entry", strict=False)


def test_resolved_model_cannot_be_constructed_as_verified_by_callers() -> None:
    entry = _entry()

    with pytest.raises(TypeError, match="created by strict resolution"):
        ResolvedModel(entry, "0.5.4", True)

    with pytest.raises(TypeError, match="created by strict resolution"):
        ResolvedModel(
            entry=entry,
            runtime_version="0.5.4",
            identity_verified=True,
        )


def test_strict_resolution_requires_a_configured_verifier() -> None:
    catalog = ModelCatalog((_entry(),))

    with pytest.raises(ModelCatalogError, match="not verified"):
        catalog.resolve(
            "fixture-entry",
            runtime_version="0.5.4",
        )


def test_verifier_failure_becomes_a_controlled_catalog_error() -> None:
    class FailingVerifier:
        def verify(
            self,
            declared_identity: ArtifactIdentity,
        ) -> ArtifactIdentity:
            raise RuntimeError("private verifier detail")

    catalog = ModelCatalog(
        (_entry(),),
        identity_verifier=FailingVerifier(),
    )

    with pytest.raises(
        ModelCatalogError,
        match="could not be verified",
    ) as error:
        catalog.resolve(
            "fixture-entry",
            runtime_version="0.5.4",
        )

    assert str(error.value) == "Artifact identity could not be verified."


@pytest.mark.parametrize(
    "conflicting_identity",
    (
        _identity(artifact_id="fixture/other"),
        _identity(variant="other-fixture-variant"),
        _identity(quantization="other-fixture-q"),
        _identity(digest="sha256:" + "b" * 64),
    ),
)
def test_verifier_must_return_the_exact_declared_identity(
    conflicting_identity: ArtifactIdentity,
) -> None:
    entry = _entry()
    verifier = MatchingIdentityVerifier(conflicting_identity)

    with pytest.raises(ModelCatalogError, match="does not match"):
        ModelCatalog(
            (entry,),
            identity_verifier=verifier,
        ).resolve(
            "fixture-entry",
            runtime_version="0.5.4",
        )


def test_catalog_rejects_duplicate_entry_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        ModelCatalog((_entry(), _entry()))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"artifact_id": ""},
        {"artifact_id": "../outside"},
        {"variant": ""},
        {"quantization": "qwen;run"},
        {"digest": "not-a-digest"},
        {"digest": "sha256:" + "A" * 64},
    ),
)
def test_artifact_identity_rejects_malformed_fields(kwargs) -> None:
    with pytest.raises(ValueError):
        _identity(**kwargs)


def test_identity_requires_a_known_runtime_enum() -> None:
    with pytest.raises(TypeError):
        ArtifactIdentity(
            runtime="ollama",
            artifact_id="fixture/model",
            variant="fixture",
            quantization="fixture-q",
            digest=_SYNTHETIC_DIGEST,
        )


def test_catalog_entry_requires_identity_and_compatibility() -> None:
    with pytest.raises(TypeError):
        ModelCatalogEntry(
            entry_id="fixture-entry",
            identity=None,
            compatibility=RuntimeCompatibility(
                ModelRuntime.OLLAMA,
                "0.5.0",
                "0.5.9",
            ),
            requirements=(_requirement(),),
            required_resources=frozenset({ResourceKind.RAM}),
            max_context_tokens=256,
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"bytes_required": 0},
        {"bytes_required": -1},
        {"bytes_required": True},
        {"source": ""},
        {"method": " "},
        {"conditions": ""},
    ),
)
def test_resource_requirement_rejects_invalid_or_missing_provenance(
    kwargs,
) -> None:
    values = {
        "resource": ResourceKind.RAM,
        "bytes_required": 4096,
        "basis": RequirementBasis.ESTIMATE,
        "source": "synthetic test fixture",
        "method": "deterministic unit-test value",
        "conditions": "test only",
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        ResourceRequirement(**values)


@pytest.mark.parametrize(
    "basis",
    (
        RequirementBasis.MEASURED,
        RequirementBasis.ESTIMATE,
        RequirementBasis.CONSERVATIVE_BOUND,
    ),
)
def test_requirement_basis_is_explicit_and_never_claims_load_guarantee(
    basis: RequirementBasis,
) -> None:
    requirement = _requirement(basis=basis)

    assert requirement.basis is basis
    assert requirement.is_estimate is (
        basis is RequirementBasis.ESTIMATE
    )
    assert requirement.is_load_guarantee is False


def test_catalog_rejects_duplicate_or_missing_resource_requirements() -> None:
    ram = _requirement()

    with pytest.raises(ValueError, match="duplicates"):
        _entry(requirements=(ram, ram))

    with pytest.raises(ValueError, match="no documented requirement"):
        _entry(
            required_resources=frozenset(
                {ResourceKind.RAM, ResourceKind.VRAM}
            )
        )


def test_empty_requirements_are_not_replaced_by_helper_defaults() -> None:
    with pytest.raises(ValueError, match="no documented requirement"):
        _entry(requirements=())


def test_runtime_compatibility_rejects_invalid_version_ranges() -> None:
    with pytest.raises(ValueError, match="inverted"):
        RuntimeCompatibility(
            ModelRuntime.OLLAMA,
            "0.6.0",
            "0.5.9",
        )

    with pytest.raises(ValueError, match="numeric major.minor.patch"):
        RuntimeCompatibility(
            ModelRuntime.OLLAMA,
            "latest",
            "1.0.0",
        )


@pytest.mark.parametrize(
    ("identity", "context"),
    (
        (_identity(digest=None), 256),
        (_identity(quantization=None), 256),
        (_identity(), None),
    ),
)
def test_strict_resolution_rejects_incomplete_identity_or_context(
    identity: ArtifactIdentity,
    context: int | None,
) -> None:
    catalog = ModelCatalog(
        (
            _entry(
                identity=identity,
                max_context_tokens=context,
            ),
        ),
    )

    with pytest.raises(
        ModelCatalogError,
        match="lacks identity, context, or required resource evidence",
    ):
        catalog.resolve(
            "fixture-entry",
            runtime_version="0.5.4",
        )


def test_catalog_rejects_compatibility_for_another_runtime() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        _entry(
            compatibility=RuntimeCompatibility(
                ModelRuntime.LLAMA_CPP,
                "0.5.0",
                "0.5.9",
            )
        )


@pytest.mark.parametrize(
    "runtime_version",
    ("0.4.9", "0.6.0", "latest", ""),
)
def test_strict_resolution_rejects_incompatible_runtime_versions(
    runtime_version: str,
) -> None:
    with pytest.raises(ModelCatalogError, match="Runtime version"):
        ModelCatalog(
            (_entry(),),
            identity_verifier=MatchingIdentityVerifier(_identity()),
        ).resolve(
            "fixture-entry",
            runtime_version=runtime_version,
        )


def test_external_identity_mismatch_does_not_mutate_catalog() -> None:
    entry = _entry()
    catalog = ModelCatalog((entry,))

    with pytest.raises(ModelCatalogError, match="does not match"):
        catalog.require_matching_external_identity(
            "fixture-entry",
            _identity(quantization="different-fixture-q"),
        )

    assert catalog.entries == (entry,)

    with pytest.raises(FrozenInstanceError):
        entry.identity = _identity(variant="mutated")

    with pytest.raises(FrozenInstanceError):
        catalog._entries = ()


def test_external_identity_match_returns_catalog_entry_without_replacing_it(
) -> None:
    entry = _entry()
    catalog = ModelCatalog((entry,))

    matched = catalog.require_matching_external_identity(
        "fixture-entry",
        _identity(),
    )

    assert matched is entry
    assert catalog.entries == (entry,)


def test_empty_catalog_and_unknown_entry_are_unavailable() -> None:
    assert DEFAULT_MODEL_CATALOG.entries == ()

    with pytest.raises(ModelCatalogError, match="unavailable"):
        DEFAULT_MODEL_CATALOG.resolve("qwen3-1.7b")

    with pytest.raises(ModelCatalogError, match="unavailable"):
        ModelCatalog().inspect("absent-entry")

    with pytest.raises(ModelCatalogError, match="Non-strict resolution"):
        ModelCatalog().resolve(
            "absent-entry",
            strict=False,
        )


def test_synthetic_test_entry_is_not_registered_as_a_production_model(
) -> None:
    synthetic_catalog = ModelCatalog((_entry(),))

    assert DEFAULT_MODEL_CATALOG.entries == ()
    assert synthetic_catalog.entries[0].entry_id == "fixture-entry"
    assert (
        synthetic_catalog.entries[0].identity.artifact_id
        == "fixture/model"
    )
