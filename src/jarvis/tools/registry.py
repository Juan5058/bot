from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from jarvis.config.validation import validate_permission_name
from jarvis.conversation.models import validate_tool_parameters
from jarvis.core.errors import ToolNotRegisteredError
from jarvis.tools.base import ToolDefinition

_TOOL_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_RISK_LEVELS = frozenset({"low", "medium", "high"})
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class ToolArgumentDefinition:
    name: str
    value_type: type
    required: bool = True
    min_length: int | None = None
    max_length: int | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("Tool argument name must be a non-empty string.")
        if type(self.value_type) is not type or self.value_type not in {
            str,
            bool,
            int,
            float,
            dict,
            list,
        }:
            raise TypeError("Tool argument type is not supported.")
        if type(self.required) is not bool:
            raise TypeError("Tool argument required flag must be a boolean.")
        if self.value_type is str:
            if self.min_length is not None and (
                type(self.min_length) is not int or self.min_length < 0
            ):
                raise ValueError("String argument minimum length is invalid.")
            if self.max_length is not None and (
                type(self.max_length) is not int
                or self.max_length < (self.min_length or 0)
            ):
                raise ValueError("String argument maximum length is invalid.")
        elif self.min_length is not None or self.max_length is not None:
            raise ValueError("Length constraints only apply to string arguments.")


@dataclass(frozen=True, slots=True)
class TrustedToolDefinition:
    name: str
    implementation_type: type[ToolDefinition]
    required_permission: str
    risk_level: str = "low"
    requires_confirmation: bool = False
    identity_required: bool = False
    execution_mode: str = "simulated"
    argument_schema: tuple[ToolArgumentDefinition, ...] = ()
    allow_additional_arguments: bool = False

    def __post_init__(self) -> None:
        if type(self.name) is not str or not _TOOL_NAME_PATTERN.fullmatch(self.name):
            raise ValueError("Trusted tool name is invalid.")
        if (
            not isinstance(self.implementation_type, type)
            or not issubclass(self.implementation_type, ToolDefinition)
        ):
            raise TypeError("Trusted tool implementation must be a ToolDefinition type.")
        validate_permission_name(self.required_permission)
        if type(self.argument_schema) is not tuple or any(
            type(argument) is not ToolArgumentDefinition
            for argument in self.argument_schema
        ):
            raise TypeError("Trusted tool argument schema is malformed.")
        argument_names = tuple(argument.name for argument in self.argument_schema)
        if len(argument_names) != len(set(argument_names)):
            raise ValueError("Trusted tool argument names must be unique.")
        if type(self.allow_additional_arguments) is not bool:
            raise TypeError("Additional tool arguments flag must be a boolean.")
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


class ToolRegistry:
    __slots__ = ("_trusted_catalog", "_registered_tools")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_trusted_catalog", "_registered_tools"} and hasattr(self, name):
            if name == "_trusted_catalog" or self.is_frozen:
                raise AttributeError("Trusted tool registry state cannot be modified.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in {"_trusted_catalog", "_registered_tools"}:
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
                argument_schema=definition.argument_schema,
                allow_additional_arguments=definition.allow_additional_arguments,
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
        self._registered_tools: dict[str, type[ToolDefinition]] = {}

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
        if tool.name in self._registered_tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        if not tool._frozen:
            tool._frozen = True
        self._registered_tools[tool.name] = type(tool)

    def has(self, tool_name: str) -> bool:
        return tool_name in self._registered_tools

    def get_trusted_definition(self, tool_name: str) -> TrustedToolDefinition:
        registered_type = self._registered_tools.get(tool_name)
        if registered_type is None:
            raise ToolNotRegisteredError(f"Tool '{tool_name}' is not registered.")
        definition = self._trusted_catalog[tool_name]
        if registered_type is not definition.implementation_type:
            raise ToolNotRegisteredError(f"Tool '{tool_name}' is not registered.")
        return definition

    def validate_tool_arguments(self, tool_name: str, parameters: Any) -> None:
        validate_tool_parameters(parameters)
        definition = self.get_trusted_definition(tool_name)
        fields = {argument.name: argument for argument in definition.argument_schema}
        if not definition.allow_additional_arguments and any(
            key not in fields for key in parameters
        ):
            raise ValueError("Tool arguments contain unsupported fields.")
        for argument in definition.argument_schema:
            if argument.required and argument.name not in parameters:
                raise ValueError("A required tool argument is missing.")
        for name, value in parameters.items():
            argument = fields.get(name)
            if argument is None:
                continue
            if type(value) is not argument.value_type:
                raise TypeError("Tool argument has an invalid type.")
            if argument.value_type is str and (
                (argument.min_length is not None and len(value) < argument.min_length)
                or (argument.max_length is not None and len(value) > argument.max_length)
            ):
                raise ValueError("Tool string argument length is invalid.")

    def list_names(self) -> list[str]:
        return sorted(self._registered_tools)

    @property
    def is_frozen(self) -> bool:
        return type(self._registered_tools) is _MAPPING_PROXY_TYPE

    def freeze(self) -> None:
        if self.is_frozen:
            return
        self._registered_tools = MappingProxyType(dict(self._registered_tools))


_CORE_TRUSTED_CATALOG = MappingProxyType(
    {
        "echo_message": TrustedToolDefinition(
            name="echo_message",
            implementation_type=EchoTool,
            required_permission="tool.echo_message",
            argument_schema=(
                ToolArgumentDefinition(
                    name="message",
                    value_type=str,
                    required=True,
                    min_length=0,
                    max_length=8000,
                ),
            ),
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
