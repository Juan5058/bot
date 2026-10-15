from __future__ import annotations

from typing import Any

from jarvis.core.context import ConversationContext
from jarvis.core.errors import ToolNotRegisteredError
from jarvis.tools.base import ToolDefinition


class EchoTool(ToolDefinition):
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
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._sealed = False

    def register(self, tool: ToolDefinition) -> None:
        if self._sealed:
            raise RuntimeError("Tool registry is sealed and cannot be modified.")
        if not isinstance(tool, ToolDefinition):
            raise TypeError("Tool registry only accepts ToolDefinition instances.")
        if not tool.name or not isinstance(tool.name, str):
            raise ValueError("Tool name is required and must be a string.")
        if not tool.required_permission or not isinstance(tool.required_permission, str):
            raise ValueError("Each tool must declare a required permission.")
        if tool.execution_mode not in {"simulated"}:
            raise ValueError("Only simulated execution mode is supported in Fase 1.")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        tool._frozen = True
        self._tools[tool.name] = tool

    def has(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def get(self, tool_name: str) -> ToolDefinition:
        if tool_name not in self._tools:
            raise ToolNotRegisteredError(f"Tool '{tool_name}' is not registered.")
        return self._tools[tool_name]

    def list_names(self) -> list[str]:
        return sorted(self._tools)

    def freeze(self) -> None:
        self._sealed = True


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.freeze()
    return registry
