from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from jarvis.core.context import ConversationContext


class ToolDefinition(ABC):
    __slots__ = ("name", "description", "required_permission", "risk_level", "requires_confirmation", "execution_mode", "_frozen")

    def __init__(self) -> None:
        self._frozen = False
        self.name = ""
        self.description = ""
        self.required_permission = ""
        self.risk_level = "low"
        self.requires_confirmation = False
        self.execution_mode = "simulated"

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False) and name in self.__slots__ and name != "_frozen":
            raise AttributeError(f"Tool '{self.__class__.__name__}' is immutable after registration.")
        super().__setattr__(name, value)

    @abstractmethod
    def execute(self, context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError
