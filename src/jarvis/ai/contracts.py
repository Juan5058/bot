from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal


MAX_INFORMATIONAL_RESPONSE_LENGTH = 16_000


@dataclass(frozen=True, slots=True, repr=False)
class InformationalRequest:
    capability: Literal["answer_question", "summarize_text"]
    content: str

    def __post_init__(self) -> None:
        if type(self.capability) is not str or self.capability not in {
            "answer_question",
            "summarize_text",
        }:
            raise ValueError("Informational capability is not supported.")
        if type(self.content) is not str or not self.content.strip():
            raise ValueError("Informational request content must be non-empty text.")
        if len(self.content) > 8000:
            raise ValueError("Informational request content exceeds the supported size.")
        if "\x00" in self.content:
            raise ValueError("Informational request contains unsupported characters.")

    def __repr__(self) -> str:
        return "InformationalRequest(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class InformationalResponse:
    text: str

    def __post_init__(self) -> None:
        if type(self.text) is not str or not self.text.strip():
            raise ValueError("Informational response must be non-empty plain text.")
        if len(self.text) > MAX_INFORMATIONAL_RESPONSE_LENGTH:
            raise ValueError("Informational response exceeds the supported size.")
        if any(
            ord(character) < 32 and character not in "\t\n\r"
            for character in self.text
        ) or "\x7f" in self.text:
            raise ValueError("Informational response contains unsupported characters.")

    def __repr__(self) -> str:
        return "InformationalResponse(<redacted>)"


@dataclass(frozen=True, slots=True)
class ActionSuggestion:
    """Untrusted provider output containing only a tool name and arguments."""

    tool_name: str
    parameters: dict[str, Any]


class AIProviderContract(ABC):
    @abstractmethod
    def classify_intent(self, user_text: str) -> dict[str, Any]:
        """Return a classification whose actionable intent is ``text_request``."""

    @abstractmethod
    def propose_action(self, user_text: str) -> ActionSuggestion:
        """Suggest a tool and arguments without security or execution metadata."""

    @abstractmethod
    def respond(self, user_text: str) -> str:
        """Return a response to the user."""

    @abstractmethod
    def respond_informational(
        self,
        request: InformationalRequest,
    ) -> InformationalResponse:
        """Return informational text only; never a tool or execution proposal."""
