from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from jarvis.config.validation import validate_permission_name
from jarvis.core.context import ConversationContext
from jarvis.core.errors import ToolNotRegisteredError
from jarvis.tools.base import ToolDefinition

_TOOL_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_RISK_LEVELS = frozenset({"low", "medium", "high"})
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class TrustedToolDefinition:
    name: str
    implementation_type: type[ToolDefinition]
    required_permission: str
    risk_level: str = "low"
    requires_confirmation: bool = False
    identity_required: bool = False
    execution_mode: str = "simulated"

    def __post_init__(self) -> None:
        if type(self.name) is not str or not _TOOL_NAME_PATTERN.fullmatch(self.name):
            raise ValueError("Trusted tool name is invalid.")
        if (
            not isinstance(self.implementation_type, type)
            or not issubclass(self.implementation_type, ToolDefinition)
        ):
            raise TypeError("Trusted tool implementation must be a ToolDefinition type.")
        validate_permission_name(self.required_permission)
        if type(self.risk_level) is not str or self.risk_level not in _RISK_LEVELS:
            raise ValueError("Trusted tool risk level is invalid.")
        if type(self.requires_confirmation) is not bool or type(self.identity_required) is not bool:
            raise TypeError("Trusted tool identity and confirmation requirements must be booleans.")
        if type(self.execution_mode) is not str or self.execution_mode != "simulated":
            raise ValueError("Only simulated trusted tools are supported.")


class EchoTool(ToolDefinition):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()
        self.name = "echo_message"
        self.description = "Return a deterministic simulated message for tests and local flows."
        self.required_permission = "tool.echo_message"
        self.risk_level = "low"
        self.requires_confirmation = False
        self.execution_mode = "simulated"

    def execute(self, context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        message = kwargs.get("message", "")
        return {
            "kind": "simulated_echo",
            "message": message,
            "status": "ok",
            "execution_mode": "simulated",
        }


class ToolRegistry:
    __slots__ = ("_trusted_catalog", "_tools")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_trusted_catalog", "_tools"} and hasattr(self, name):
            if name == "_trusted_catalog" or self.is_frozen:
                raise AttributeError("Trusted tool registry state cannot be modified.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in {"_trusted_catalog", "_tools"}:
            raise AttributeError("Tool registry state cannot be deleted.")
        super().__delattr__(name)

    def __init__(self) -> None:
        self._initialize(_CORE_TRUSTED_CATALOG)

    def _initialize(self, catalog: Mapping[str, TrustedToolDefinition]) -> None:
        if type(catalog) not in {dict, _MAPPING_PROXY_TYPE}:
            raise TypeError("Trusted tool catalog must be a mapping.")
        copied_catalog: dict[str, TrustedToolDefinition] = {}
        implementation_types: set[type[ToolDefinition]] = set()
        for name, definition in catalog.items():
            if type(name) is not str or type(definition) is not TrustedToolDefinition:
                raise TypeError("Trusted tool catalog entries are malformed.")
            validated_definition = TrustedToolDefinition(
                name=definition.name,
                implementation_type=definition.implementation_type,
                required_permission=definition.required_permission,
                risk_level=definition.risk_level,
                requires_confirmation=definition.requires_confirmation,
                identity_required=definition.identity_required,
                execution_mode=definition.execution_mode,
            )
            if name != validated_definition.name:
                raise ValueError("Trusted tool catalog key does not match its definition.")
            if validated_definition.implementation_type in implementation_types:
                raise ValueError("A tool implementation type cannot have multiple trusted identities.")
            implementation_types.add(validated_definition.implementation_type)
            copied_catalog[name] = validated_definition
        self._trusted_catalog = MappingProxyType(copied_catalog)
        self._tools: dict[str, ToolDefinition] = {}

    @classmethod
    def for_trusted_composition(
        cls,
        trusted_catalog: Mapping[str, TrustedToolDefinition],
    ) -> ToolRegistry:
        if cls is not ToolRegistry:
            raise TypeError("Trusted composition only creates core tool registries.")
        if type(trusted_catalog) is not dict:
            raise TypeError("Trusted composition requires a plain tool catalog.")
        registry = cls.__new__(cls)
        registry._initialize(trusted_catalog)
        return registry

    def register(self, tool: ToolDefinition) -> None:
        if self.is_frozen:
            raise RuntimeError("Tool registry is sealed and cannot be modified.")
        if not isinstance(tool, ToolDefinition):
            raise TypeError("Tool registry only accepts ToolDefinition instances.")
        if type(tool.name) is not str or not _TOOL_NAME_PATTERN.fullmatch(tool.name):
            raise ValueError("Tool name is required and must be a string.")
        trusted_definition = self._trusted_catalog.get(tool.name)
        if trusted_definition is None:
            raise ValueError(f"Tool '{tool.name}' is not in the trusted catalog.")
        if type(tool) is not trusted_definition.implementation_type:
            raise TypeError("Tool implementation type does not match the trusted catalog.")
        if (
            tool.required_permission != trusted_definition.required_permission
            or tool.risk_level != trusted_definition.risk_level
            or tool.requires_confirmation != trusted_definition.requires_confirmation
            or tool.execution_mode != trusted_definition.execution_mode
        ):
            raise ValueError("Tool security metadata does not match the trusted catalog.")
        if type(tool.description) is not str or not tool.description.strip() or len(tool.description) > 512:
            raise ValueError("Tool description must be a non-empty string of at most 512 characters.")
        if type(tool.required_permission) is not str:
            raise TypeError("Tool permission metadata must be a string.")
        validate_permission_name(tool.required_permission)
        if type(tool.risk_level) is not str or tool.risk_level not in _RISK_LEVELS:
            raise ValueError("Tool risk level is invalid.")
        if type(tool.requires_confirmation) is not bool:
            raise TypeError("Tool confirmation requirement must be a boolean.")
        if type(tool.execution_mode) is not str or tool.execution_mode != "simulated":
            raise ValueError("Only simulated execution mode is supported.")
        if type(tool._frozen) is not bool:
            raise TypeError("Tool frozen state must be a boolean.")
        if not callable(getattr(tool, "execute", None)):
            raise ValueError("Tool must provide an executable implementation.")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        if not tool._frozen:
            tool._frozen = True
        self._tools[tool.name] = tool

    def has(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def get(self, tool_name: str) -> ToolDefinition:
        if tool_name not in self._tools:
            raise ToolNotRegisteredError(f"Tool '{tool_name}' is not registered.")
        return self._tools[tool_name]

    def get_trusted_definition(self, tool_name: str) -> TrustedToolDefinition:
        if tool_name not in self._tools:
            raise ToolNotRegisteredError(f"Tool '{tool_name}' is not registered.")
        return self._trusted_catalog[tool_name]

    def list_names(self) -> list[str]:
        return sorted(self._tools)

    @property
    def is_frozen(self) -> bool:
        return type(self._tools) is _MAPPING_PROXY_TYPE

    def freeze(self) -> None:
        if self.is_frozen:
            return
        self._tools = MappingProxyType(dict(self._tools))


_CORE_TRUSTED_CATALOG = MappingProxyType(
    {
        "echo_message": TrustedToolDefinition(
            name="echo_message",
            implementation_type=EchoTool,
            required_permission="tool.echo_message",
            risk_level="low",
            requires_confirmation=False,
            identity_required=False,
            execution_mode="simulated",
        )
    }
)


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.freeze()
    return registry
