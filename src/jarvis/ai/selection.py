from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from jarvis.ai.contracts import AIProviderContract
from jarvis.config.settings import Settings
from jarvis.conversation.router import (
    Capability,
    InterpretationStatus,
    RouteCategory,
    RouteDecision,
)
from jarvis.tools.registry import ToolRegistry


class SelectionTarget(str, Enum):
    DETERMINISTIC_ROUTER = "deterministic_router"
    DETERMINISTIC_TOOL = "deterministic_tool"
    AI_PROVIDER = "ai_provider"
    TERMINAL = "terminal"
    UNAVAILABLE = "unavailable"


class SelectionReason(str, Enum):
    DETERMINISTIC_ROUTE = "deterministic_route"
    REGISTERED_TOOL = "registered_tool"
    CLARIFICATION_OR_REJECTION = "clarification_or_rejection"
    AI_DISABLED = "ai_disabled"
    PROVIDER_NOT_AVAILABLE = "provider_not_available"
    PROVIDER_SELECTED = "provider_selected"
    REMOTE_PRIVACY_POLICY = "remote_privacy_policy"
    NON_AI_CAPABILITY = "non_ai_capability"


@dataclass(frozen=True, slots=True, repr=False)
class ProviderSelection:
    target: SelectionTarget
    reason: SelectionReason
    provider_name: str | None = None
    provider: AIProviderContract | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.target) is not SelectionTarget:
            raise TypeError("Selection target is invalid.")
        if type(self.reason) is not SelectionReason:
            raise TypeError("Selection reason is invalid.")
        if self.provider_name is not None and self.provider_name not in {
            "simulated",
            "local",
            "remote",
        }:
            raise ValueError("Selected provider name is not supported.")
        expected_reasons = {
            SelectionTarget.DETERMINISTIC_ROUTER: {
                SelectionReason.DETERMINISTIC_ROUTE
            },
            SelectionTarget.DETERMINISTIC_TOOL: {SelectionReason.REGISTERED_TOOL},
            SelectionTarget.AI_PROVIDER: {SelectionReason.PROVIDER_SELECTED},
            SelectionTarget.TERMINAL: {
                SelectionReason.CLARIFICATION_OR_REJECTION,
                SelectionReason.NON_AI_CAPABILITY,
            },
            SelectionTarget.UNAVAILABLE: {
                SelectionReason.AI_DISABLED,
                SelectionReason.PROVIDER_NOT_AVAILABLE,
                SelectionReason.REMOTE_PRIVACY_POLICY,
            },
        }
        if self.reason not in expected_reasons[self.target]:
            raise ValueError("Selection reason does not match its target.")
        if self.target is SelectionTarget.AI_PROVIDER:
            if (
                self.provider_name is None
                or not isinstance(self.provider, AIProviderContract)
            ):
                raise ValueError("Provider selections require a registered provider.")
        elif self.provider_name is not None or self.provider is not None:
            raise ValueError("Non-provider selections cannot contain a provider.")


class AIProviderSelector:
    __slots__ = ("_providers",)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_providers" and hasattr(self, name):
            raise AttributeError("The provider catalog is immutable.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_providers":
            raise AttributeError("The provider catalog cannot be deleted.")
        super().__delattr__(name)

    def __init__(
        self,
        providers: Mapping[str, AIProviderContract] | None = None,
    ) -> None:
        if providers is None:
            providers = {}
        if type(providers) is not dict and not isinstance(providers, Mapping):
            raise TypeError("Provider catalog must be a mapping.")

        supported_providers = {"simulated", "local", "remote"}
        validated: dict[str, AIProviderContract] = {}
        for name, provider in providers.items():
            if type(name) is not str or name not in supported_providers:
                raise ValueError("Provider catalog contains an unsupported provider.")
            if not isinstance(provider, AIProviderContract):
                raise TypeError("Provider catalog contains an invalid provider.")
            validated[name] = provider
        self._providers = MappingProxyType(validated)

    def select(
        self,
        decision: RouteDecision,
        registry: ToolRegistry,
        settings: Settings,
    ) -> ProviderSelection:
        if type(decision) is not RouteDecision:
            raise TypeError("Provider selection requires a validated route.")
        if type(registry) is not ToolRegistry:
            raise TypeError("Provider selection requires the trusted tool registry.")
        if type(settings) is not Settings:
            raise TypeError("Provider selection requires validated settings.")

        if decision.category is RouteCategory.DETERMINISTIC:
            if decision.status is InterpretationStatus.UNDERSTOOD:
                return ProviderSelection(
                    SelectionTarget.DETERMINISTIC_ROUTER,
                    SelectionReason.DETERMINISTIC_ROUTE,
                )
            return ProviderSelection(
                SelectionTarget.UNAVAILABLE,
                SelectionReason.PROVIDER_NOT_AVAILABLE,
            )

        if decision.category is RouteCategory.SPECIALIZED:
            if (
                decision.capability is Capability.ECHO_MESSAGE
                and registry.has("echo_message")
            ):
                return ProviderSelection(
                    SelectionTarget.DETERMINISTIC_TOOL,
                    SelectionReason.REGISTERED_TOOL,
                )
            return ProviderSelection(
                SelectionTarget.UNAVAILABLE,
                SelectionReason.PROVIDER_NOT_AVAILABLE,
            )

        if decision.category in {RouteCategory.CLARIFY, RouteCategory.REJECT}:
            return ProviderSelection(
                SelectionTarget.TERMINAL,
                SelectionReason.CLARIFICATION_OR_REJECTION,
            )

        if decision.category is RouteCategory.EXTERNAL_SERVICE:
            return ProviderSelection(
                SelectionTarget.TERMINAL,
                SelectionReason.NON_AI_CAPABILITY,
            )

        if (
            decision.category is not RouteCategory.GENERATIVE
            or decision.status is not InterpretationStatus.UNDERSTOOD
            or decision.capability
            not in {Capability.ANSWER_QUESTION, Capability.SUMMARIZE_TEXT}
        ):
            return ProviderSelection(
                SelectionTarget.UNAVAILABLE,
                SelectionReason.PROVIDER_NOT_AVAILABLE,
            )

        if not settings.ai_enabled:
            return ProviderSelection(
                SelectionTarget.UNAVAILABLE,
                SelectionReason.AI_DISABLED,
            )

        provider_name = settings.ai_provider
        if provider_name == "remote":
            if not settings.remote_ai_enabled:
                return ProviderSelection(
                    SelectionTarget.UNAVAILABLE,
                    SelectionReason.PROVIDER_NOT_AVAILABLE,
                )
            return ProviderSelection(
                SelectionTarget.UNAVAILABLE,
                SelectionReason.REMOTE_PRIVACY_POLICY,
            )

        provider = self._providers.get(provider_name)
        if provider is None:
            return ProviderSelection(
                SelectionTarget.UNAVAILABLE,
                SelectionReason.PROVIDER_NOT_AVAILABLE,
            )
        return ProviderSelection(
            SelectionTarget.AI_PROVIDER,
            SelectionReason.PROVIDER_SELECTED,
            provider_name=provider_name,
            provider=provider,
        )
