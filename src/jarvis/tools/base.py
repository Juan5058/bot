from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from jarvis.core.context import ConversationContext


class ToolDefinition(ABC):
    __slots__ = ("name", "description", "required_permission", "risk_level", "requires_confirmation", "execution_mode", "_frozen")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dictoffset__ != 0:
            raise TypeError("ToolDefinition subclasses must use slots; instance dictionaries can shadow execute.")
        reserved = {
            "name",
            "description",
            "required_permission",
            "risk_level",
            "requires_confirmation",
            "execution_mode",
            "_frozen",
        }
        if reserved.intersection(cls.__dict__) or {
            "__setattr__",
            "__delattr__",
            "__getattribute__",
        }.intersection(cls.__dict__):
            raise TypeError("ToolDefinition subclasses cannot override frozen security attributes.")

    def __init__(self) -> None:
        self._frozen = False
        self.name = ""
        self.description = ""
        self.required_permission = ""
        self.risk_level = "low"
        self.requires_confirmation = False
        self.execution_mode = "simulated"

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "execute":
            raise AttributeError("Tool execution methods cannot be replaced on an instance.")
        if getattr(self, "_frozen", False):
            raise AttributeError(f"Tool '{self.__class__.__name__}' is immutable after registration.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name == "execute" or getattr(self, "_frozen", False):
            raise AttributeError(f"Tool '{self.__class__.__name__}' cannot be modified after registration.")
        super().__delattr__(name)

    @abstractmethod
    def execute(self, context: ConversationContext, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError
