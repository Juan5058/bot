
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from re import fullmatch
from types import MappingProxyType
from typing import Mapping, Protocol


_IDENTIFIER_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}"
_CATALOG_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
_RUNTIME_VERSION_PATTERN = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
_SHA256_PATTERN = r"sha256:[0-9a-f]{64}"


class ModelCatalogError(ValueError):
    """A catalog entry is invalid, unavailable, or incompatible."""


class ModelRuntime(str, Enum):
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"


class ResourceKind(str, Enum):
    RAM = "ram"
    VRAM = "vram"


class RequirementBasis(str, Enum):
    MEASURED = "measured"
    ESTIMATE = "estimate"
    CONSERVATIVE_BOUND = "conservative_bound"


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    runtime: ModelRuntime
    artifact_id: str
    variant: str
    quantization: str | None
    digest: str | None

    def __post_init__(self) -> None:
        if type(self.runtime) is not ModelRuntime:
            raise TypeError("Artifact runtime is invalid.")

        if (
            type(self.artifact_id) is not str
            or fullmatch(_IDENTIFIER_PATTERN, self.artifact_id) is None
        ):
            raise ValueError("Artifact identifier is invalid.")

        if (
            type(self.variant) is not str
            or fullmatch(_IDENTIFIER_PATTERN, self.variant) is None
        ):
            raise ValueError("Artifact variant is invalid.")

        if self.quantization is not None and (
            type(self.quantization) is not str
            or fullmatch(_IDENTIFIER_PATTERN, self.quantization) is None
        ):
            raise ValueError("Artifact quantization is invalid.")

        if self.digest is not None and (
            type(self.digest) is not str
            or fullmatch(_SHA256_PATTERN, self.digest) is None
        ):
            raise ValueError("Artifact digest is invalid.")


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    """Documented resource value; provenance is not a load guarantee."""

    resource: ResourceKind
    bytes_required: int
    basis: RequirementBasis
    source: str
    method: str
    conditions: str

    def __post_init__(self) -> None:
        if type(self.resource) is not ResourceKind:
            raise TypeError("Resource kind is invalid.")

        if (
            type(self.bytes_required) is not int
            or self.bytes_required <= 0
        ):
            raise ValueError(
                "Resource requirement must be a positive byte count."
            )

        if type(self.basis) is not RequirementBasis:
            raise TypeError("Resource requirement basis is invalid.")

        for name, value in (
            ("source", self.source),
            ("method", self.method),
            ("conditions", self.conditions),
        ):
            if (
                type(value) is not str
                or not value.strip()
                or len(value) > 1000
            ):
                raise ValueError(
                    f"Resource requirement {name} is required."
                )

    @property
    def is_estimate(self) -> bool:
        return self.basis is RequirementBasis.ESTIMATE

    @property
    def is_load_guarantee(self) -> bool:
        """No evidence category alone guarantees a successful model load."""
        return False


@dataclass(frozen=True, slots=True)
class RuntimeCompatibility:
    runtime: ModelRuntime
    minimum_version: str
    maximum_version: str

    def __post_init__(self) -> None:
        if type(self.runtime) is not ModelRuntime:
            raise TypeError("Compatible runtime is invalid.")

        minimum = _parse_version(self.minimum_version)
        maximum = _parse_version(self.maximum_version)

        if minimum > maximum:
            raise ValueError("Runtime compatibility range is inverted.")

    def accepts(self, runtime: ModelRuntime, version: str) -> bool:
        if type(runtime) is not ModelRuntime or runtime is not self.runtime:
            return False

        parsed = _parse_version(version)

        return (
            _parse_version(self.minimum_version)
            <= parsed
            <= _parse_version(self.maximum_version)
        )


@dataclass(frozen=True, slots=True)
class ModelCatalogEntry:
    entry_id: str
    identity: ArtifactIdentity
    compatibility: RuntimeCompatibility
    requirements: tuple[ResourceRequirement, ...]
    required_resources: frozenset[ResourceKind]
    max_context_tokens: int | None
    tokenizer_id: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.entry_id) is not str
            or fullmatch(_CATALOG_ID_PATTERN, self.entry_id) is None
        ):
            raise ValueError("Catalog entry identifier is invalid.")

        if type(self.identity) is not ArtifactIdentity:
            raise TypeError("Catalog artifact identity is invalid.")

        if type(self.compatibility) is not RuntimeCompatibility:
            raise TypeError("Catalog runtime compatibility is invalid.")

        if self.compatibility.runtime is not self.identity.runtime:
            raise ValueError("Catalog identity and runtime are incompatible.")

        if type(self.requirements) is not tuple or any(
            type(requirement) is not ResourceRequirement
            for requirement in self.requirements
        ):
            raise TypeError("Catalog resource requirements are invalid.")

        resources = tuple(
            requirement.resource for requirement in self.requirements
        )

        if len(resources) != len(set(resources)):
            raise ValueError(
                "Catalog resource requirements contain duplicates."
            )

        if (
            type(self.required_resources) is not frozenset
            or ResourceKind.RAM not in self.required_resources
            or any(
                type(resource) is not ResourceKind
                for resource in self.required_resources
            )
        ):
            raise ValueError(
                "Catalog required resource dimensions are invalid."
            )

        if not self.required_resources.issubset(resources):
            raise ValueError(
                "A required resource has no documented requirement."
            )

        if self.max_context_tokens is not None and (
            type(self.max_context_tokens) is not int
            or self.max_context_tokens <= 0
        ):
            raise ValueError("Catalog context limit is invalid.")
        if self.tokenizer_id is not None and (
            type(self.tokenizer_id) is not str
            or fullmatch(_IDENTIFIER_PATTERN, self.tokenizer_id) is None
        ):
            raise ValueError("Catalog tokenizer identifier is invalid.")

    def is_strictly_eligible(self) -> bool:
        """Check identity, context, and required resource evidence.

        CONSERVATIVE_BOUND is a catalog policy, not authenticated provenance
        and not a guarantee that the model can be loaded.
        """
        if (
            self.identity.digest is None
            or self.identity.quantization is None
            or self.max_context_tokens is None
            or self.tokenizer_id is None
        ):
            return False

        requirements_by_resource = {
            requirement.resource: requirement
            for requirement in self.requirements
        }

        for resource in self.required_resources:
            requirement = requirements_by_resource.get(resource)

            if (
                requirement is None
                or requirement.basis
                is not RequirementBasis.CONSERVATIVE_BOUND
            ):
                return False

        return True


class ArtifactIdentityVerifier(Protocol):
    def verify(
        self,
        declared_identity: ArtifactIdentity,
    ) -> ArtifactIdentity | None:
        """Obtain identity independently from the configured runtime."""
        ...


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Evidence of strict catalog resolution, not authorization or load approval.

    Instances are produced by ``ModelCatalog.resolve`` after successful identity
    verification. Python-level construction controls are API safeguards, not a
    security boundary; future composition must trust its configured verifier
    and perform authorization and resource admission separately.
    """

    entry: ModelCatalogEntry
    runtime_version: str
    verified_identity: ArtifactIdentity

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "ResolvedModel instances are created by strict resolution."
        )


@dataclass(frozen=True, slots=True, init=False)
class ModelCatalog:
    """Immutable application-owned metadata.

    Identity claims need an injected verifier. ``inspect`` returns only the
    declared entry. ``resolve`` is always strict; passing ``strict=False`` is
    rejected. Neither operation authorizes or loads a model.
    """

    _entries: tuple[ModelCatalogEntry, ...]
    _by_id: Mapping[str, ModelCatalogEntry]
    _identity_verifier: ArtifactIdentityVerifier | None

    def __init__(
        self,
        entries: tuple[ModelCatalogEntry, ...] = (),
        *,
        identity_verifier: ArtifactIdentityVerifier | None = None,
    ) -> None:
        if type(entries) is not tuple or any(
            type(entry) is not ModelCatalogEntry for entry in entries
        ):
            raise TypeError(
                "Catalog entries must be a tuple of validated entries."
            )

        if identity_verifier is not None and not callable(
            getattr(identity_verifier, "verify", None)
        ):
            raise TypeError("Catalog identity verifier is invalid.")

        by_id: dict[str, ModelCatalogEntry] = {}

        for entry in entries:
            if entry.entry_id in by_id:
                raise ValueError(
                    "Catalog entry identifiers must be unique."
                )

            by_id[entry.entry_id] = entry

        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_entries", entries)
        object.__setattr__(
            self,
            "_identity_verifier",
            identity_verifier,
        )

    @property
    def entries(self) -> tuple[ModelCatalogEntry, ...]:
        return self._entries

    @property
    def has_identity_verifier(self) -> bool:
        """Whether a verifier was configured; this does not establish its trust."""
        return self._identity_verifier is not None

    @property
    def has_authenticated_identity_verifier(self) -> bool:
        """Whether a trusted verifier authenticates the runtime artifact bytes.

        The current verifier contract only compares declared identity metadata;
        no implementation independently authenticates the bytes Ollama loaded.
        """
        return False

    def inspect(self, entry_id: str) -> ModelCatalogEntry:
        """Return the declaration without asserting its identity."""
        if type(entry_id) is not str:
            raise TypeError("Catalog entry identifier must be text.")

        entry = self._by_id.get(entry_id)

        if entry is None:
            raise ModelCatalogError("Catalog entry is unavailable.")

        return entry

    def resolve(
        self,
        entry_id: str,
        *,
        strict: bool = True,
        runtime_version: str | None = None,
        requested_context_tokens: int | None = None,
    ) -> ResolvedModel:
        if strict is not True:
            if type(strict) is not bool:
                raise TypeError(
                    "Strict resolution setting must be a boolean."
                )

            raise ModelCatalogError(
                "Non-strict resolution is unsupported; use inspect() instead."
            )

        entry = self.inspect(entry_id)

        if requested_context_tokens is not None:
            if (
                type(requested_context_tokens) is not int
                or requested_context_tokens <= 0
            ):
                raise ValueError(
                    "Requested context must be a positive integer."
                )

            if (
                entry.max_context_tokens is None
                or requested_context_tokens > entry.max_context_tokens
            ):
                raise ModelCatalogError(
                    "Requested context exceeds catalog limits."
                )

        if not entry.is_strictly_eligible():
            raise ModelCatalogError(
                "Catalog entry lacks identity, context, "
                "or required resource evidence."
            )

        if (
            runtime_version is None
            or not _accepts_runtime_version(entry, runtime_version)
        ):
            raise ModelCatalogError(
                "Runtime version is incompatible with catalog entry."
            )

        verifier = self._identity_verifier

        if verifier is None:
            raise ModelCatalogError(
                "Artifact identity is not verified; "
                "a configured verifier is required."
            )

        try:
            verified_identity = verifier.verify(entry.identity)
        except Exception:
            raise ModelCatalogError(
                "Artifact identity could not be verified."
            ) from None

        if (
            type(verified_identity) is not ArtifactIdentity
            or verified_identity != entry.identity
        ):
            raise ModelCatalogError(
                "Verified artifact identity does not match catalog entry."
            )

        return _create_resolved_model(
            entry,
            runtime_version,
            verified_identity,
            _token=_RESOLVED_MODEL_CREATION_TOKEN,
        )

    def require_matching_external_identity(
        self,
        entry_id: str,
        external_identity: ArtifactIdentity,
    ) -> ModelCatalogEntry:
        """Check metadata consistency; this does not authenticate the artifact."""
        if type(entry_id) is not str:
            raise TypeError("Catalog entry identifier must be text.")

        if type(external_identity) is not ArtifactIdentity:
            raise TypeError("External artifact identity is invalid.")

        entry = self._by_id.get(entry_id)

        if entry is None:
            raise ModelCatalogError("Catalog entry is unavailable.")

        if external_identity != entry.identity:
            raise ModelCatalogError(
                "External artifact identity does not match."
            )

        return entry


def _parse_version(version: str) -> tuple[int, int, int]:
    if type(version) is not str or fullmatch(
        _RUNTIME_VERSION_PATTERN,
        version,
    ) is None:
        raise ValueError(
            "Runtime version must use numeric major.minor.patch form."
        )

    major, minor, patch = (int(part) for part in version.split("."))

    return major, minor, patch


def _accepts_runtime_version(
    entry: ModelCatalogEntry,
    version: str,
) -> bool:
    try:
        return entry.compatibility.accepts(
            entry.identity.runtime,
            version,
        )
    except (TypeError, ValueError):
        return False


DEFAULT_MODEL_CATALOG = ModelCatalog()


_RESOLVED_MODEL_CREATION_TOKEN = object()


def _create_resolved_model(
    entry: ModelCatalogEntry,
    runtime_version: str,
    verified_identity: ArtifactIdentity,
    *,
    _token: object,
) -> ResolvedModel:
    if _token is not _RESOLVED_MODEL_CREATION_TOKEN:
        raise TypeError(
            "ResolvedModel instances require strict resolution."
        )

    resolved = object.__new__(ResolvedModel)
    object.__setattr__(resolved, "entry", entry)
    object.__setattr__(resolved, "runtime_version", runtime_version)
    object.__setattr__(resolved, "verified_identity", verified_identity)

    return resolved
