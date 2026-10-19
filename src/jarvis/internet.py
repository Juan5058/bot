from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

class InternetServiceCapability(str, Enum):
    WEB_SEARCH = "web_search"


class InternetServiceUnavailableReason(str, Enum):
    POLICY_DENIED = "policy_denied"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"


@dataclass(frozen=True, slots=True, repr=False)
class InternetServiceRequest:
    capability: InternetServiceCapability
    query: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.capability) is not InternetServiceCapability:
            raise TypeError("Internet service capability is invalid.")
        if type(self.query) is not str or not self.query.strip():
            raise ValueError("Internet service query must be non-empty text.")
        if len(self.query) > 8000 or "\x00" in self.query:
            raise ValueError("Internet service query is outside supported limits.")

    def __repr__(self) -> str:
        return "InternetServiceRequest(<redacted>)"


@dataclass(frozen=True, slots=True)
class InternetAccessPolicy:
    allowed_capabilities: frozenset[InternetServiceCapability] = frozenset()

    def __post_init__(self) -> None:
        if type(self.allowed_capabilities) is not frozenset or any(
            type(capability) is not InternetServiceCapability
            for capability in self.allowed_capabilities
        ):
            raise TypeError("Internet access policy is invalid.")

    def allows(self, capability: InternetServiceCapability) -> bool:
        return (
            type(capability) is InternetServiceCapability
            and capability in self.allowed_capabilities
        )


@dataclass(frozen=True, slots=True)
class InternetServiceSelection:
    capability: InternetServiceCapability
    reason: InternetServiceUnavailableReason

    def __post_init__(self) -> None:
        if type(self.capability) is not InternetServiceCapability:
            raise TypeError("Internet service selection capability is invalid.")
        if type(self.reason) is not InternetServiceUnavailableReason:
            raise TypeError("Internet service selection reason is invalid.")


class InternetServiceSelector:
    __slots__ = ("_policy",)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_policy" and hasattr(self, name):
            raise AttributeError("Internet access policy is immutable.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_policy":
            raise AttributeError("Internet access policy cannot be deleted.")
        super().__delattr__(name)

    def __init__(self, policy: InternetAccessPolicy | None = None) -> None:
        if policy is None:
            policy = InternetAccessPolicy()
        if type(policy) is not InternetAccessPolicy:
            raise TypeError("Internet service selector requires its own policy.")
        self._policy = policy

    def select(
        self,
        request: InternetServiceRequest,
    ) -> InternetServiceSelection:
        if type(request) is not InternetServiceRequest:
            raise TypeError("Internet service selection requires a validated request.")
        if not self._policy.allows(request.capability):
            reason = InternetServiceUnavailableReason.POLICY_DENIED
        else:
            reason = InternetServiceUnavailableReason.ADAPTER_UNAVAILABLE
        return InternetServiceSelection(request.capability, reason)
