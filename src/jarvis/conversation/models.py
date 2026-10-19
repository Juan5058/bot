from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


_MAX_PARAMETER_DEPTH = 32
_MAX_CONTAINER_ITEMS = 256
_MAX_PARAMETER_NODES = 2048
_MAX_PARAMETER_STRING_LENGTH = 8000
MAX_USER_INPUT_LENGTH = 8000
MAX_INFORMATIONAL_RESPONSE_LENGTH = 16000


@dataclass(frozen=True, slots=True, repr=False)
class RequestEnvelope:
    text: str = field(repr=False)
    request_id: str = field(default_factory=lambda: uuid4().hex, init=False)

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise TypeError("Request text must be a string.")
        if len(self.text) > MAX_USER_INPUT_LENGTH:
            raise ValueError(
                f"Request text cannot exceed {MAX_USER_INPUT_LENGTH} characters."
            )
        if not self.text.strip():
            raise ValueError("Request text cannot be empty.")

    def __repr__(self) -> str:
        return "RequestEnvelope(<redacted>)"


def validate_tool_parameters(parameters: Any) -> None:
    """Validate bounded JSON-compatible tool parameters without changing them."""
    if type(parameters) is not dict:
        raise TypeError("Tool parameters must be a dictionary.")

    node_count = 0
    active_containers: set[int] = set()
    pending: list[tuple[Any, int, bool]] = [(parameters, 0, False)]

    while pending:
        value, depth, exiting = pending.pop()
        if exiting:
            active_containers.remove(id(value))
            continue

        node_count += 1
        if node_count > _MAX_PARAMETER_NODES:
            raise ValueError("Proposal parameters exceed the supported size.")

        if type(value) is str:
            if len(value) > _MAX_PARAMETER_STRING_LENGTH:
                raise ValueError(
                    "Proposal parameter strings exceed the supported length."
                )
            continue
        if value is None or type(value) in {bool, int}:
            continue
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError(
                    "Proposal parameters cannot contain non-finite numbers."
                )
            continue

        if type(value) not in {list, dict}:
            raise TypeError(
                "Proposal parameters must contain only JSON-compatible values."
            )
        if depth >= _MAX_PARAMETER_DEPTH:
            raise ValueError(
                "Proposal parameters exceed the supported nesting depth."
            )
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError(
                "Proposal parameter containers exceed the supported size."
            )

        container_id = id(value)
        if container_id in active_containers:
            raise ValueError("Proposal parameters cannot contain cycles.")
        active_containers.add(container_id)
        pending.append((value, depth, True))

        if type(value) is list:
            children = value
        else:
            children = []
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(
                        "Proposal parameter keys must be strings."
                    )
                if len(key) > _MAX_PARAMETER_STRING_LENGTH:
                    raise ValueError(
                        "Proposal parameter keys exceed the supported length."
                    )
                children.append(item)
        pending.extend((item, depth + 1, False) for item in reversed(children))


@dataclass(slots=True)
class UserInput:
    text: str


@dataclass(slots=True, repr=False)
class ToolProposal:
    tool_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    permission: str = ""
    risk_level: str = "low"
    requires_confirmation: bool = False
    confirmation_token: str | None = field(
        default=None,
        repr=False,
    )
    action_ref: str = ""

    def __repr__(self) -> str:
        return "ToolProposal(<redacted>)"


@dataclass(slots=True, repr=False)
class ConversationResult:
    user_text: str = field(repr=False)
    intent: str
    proposal: ToolProposal | None = field(repr=False)
    authorized: bool
    success: bool
    tool_result: dict[str, Any] | None = field(
        default=None,
        repr=False,
    )
    informational_response: str | None = field(
        default=None,
        repr=False,
    )
    error: str | None = field(
        default=None,
        repr=False,
    )
    audit_id: str | None = None

    def __repr__(self) -> str:
        return "ConversationResult(<redacted>)"

    def to_dict(self) -> dict[str, Any]:
        if type(self.authorized) is not bool:
            raise TypeError("Conversation result authorization status is invalid.")
        if type(self.success) is not bool:
            raise TypeError("Conversation result status is invalid.")
        if type(self.intent) is not str:
            raise TypeError("Conversation result intent is invalid.")
        if self.intent not in {"text_request", "unknown"}:
            raise ValueError("Conversation result intent is unsupported.")
        if self.proposal is not None:
            if type(self.proposal) is not ToolProposal:
                raise TypeError(
                    "Conversation result proposal has an invalid type."
                )

            proposal = self.proposal

            if (
                type(proposal.tool_name) is not str
                or type(proposal.permission) is not str
                or type(proposal.risk_level) is not str
                or type(proposal.requires_confirmation) is not bool
                or type(proposal.action_ref) is not str
                or type(proposal.parameters) is not dict
            ):
                raise TypeError(
                    "Conversation result proposal has invalid fields."
                )
            validate_tool_parameters(proposal.parameters)

        if (
            type(self.user_text) is not str
            or len(self.user_text) > MAX_USER_INPUT_LENGTH
        ):
            raise TypeError("Conversation result user text is invalid.")
        if self.success and self.error is not None:
            raise ValueError("Successful conversation results cannot contain an error.")
        if self.success and self.proposal is not None and self.tool_result is None:
            raise ValueError(
                "Successful tool proposals must include an execution result."
            )
        if self.informational_response is not None:
            if (
                type(self.informational_response) is not str
                or not self.informational_response.strip()
                or len(self.informational_response)
                > MAX_INFORMATIONAL_RESPONSE_LENGTH
                or any(
                    ord(character) < 32 and character not in "\t\n\r"
                    for character in self.informational_response
                )
                or "\x7f" in self.informational_response
            ):
                raise ValueError("Informational response is invalid.")
            if (
                not self.success
                or self.authorized
                or self.proposal is not None
                or self.tool_result is not None
                or self.error is not None
            ):
                raise ValueError(
                    "Informational responses cannot represent authorized tool execution."
                )
        error_messages = {
            "Request could not be processed.",
            "Request was denied.",
            "Tool execution failed.",
            "Please provide a message after 'repite:' or 'echo:'.",
            "Please provide text after 'pregunta:' or 'resumen:'.",
            "Use 'repite: <message>' or 'echo: <message>' for a supported request.",
            "This request requires generative processing, which is unavailable.",
            "Informative processing is unavailable.",
            "Informative response could not be generated.",
            "Please provide search text after 'web:' or 'busca en Internet:'.",
            "Internet search is unavailable.",
        }
        if self.error is not None and type(self.error) is not str:
            raise TypeError("Conversation result error is invalid.")
        error = (
            self.error
            if self.error in error_messages
            else "Request could not be processed."
            if self.error is not None
            else None
        )

        public_tool_result = None
        if self.tool_result is not None:
            if (
                type(self.tool_result) is not dict
                or any(type(key) is not str for key in self.tool_result)
                or set(self.tool_result)
                != {"tool_name", "status", "result"}
            ):
                raise ValueError(
                    "Conversation result contains an invalid public tool result."
                )
            if (
                type(self.tool_result["tool_name"]) is not str
                or self.tool_result["tool_name"] != "echo_message"
                or type(self.tool_result["status"]) is not str
                or self.tool_result["status"] != "simulated_ok"
                or type(self.tool_result["result"]) is not dict
            ):
                raise ValueError(
                    "Conversation result contains an invalid public tool result."
                )
            echo_result = self.tool_result["result"]
            if (
                any(type(key) is not str for key in echo_result)
                or set(echo_result)
                != {"kind", "message", "status", "execution_mode"}
                or type(echo_result["kind"]) is not str
                or echo_result["kind"] != "simulated_echo"
                or type(echo_result["message"]) is not str
                or len(echo_result["message"]) > _MAX_PARAMETER_STRING_LENGTH
                or type(echo_result["status"]) is not str
                or echo_result["status"] != "ok"
                or type(echo_result["execution_mode"]) is not str
                or echo_result["execution_mode"] != "simulated"
            ):
                raise ValueError(
                    "Conversation result contains an invalid public tool result."
                )
            if not self.success:
                raise ValueError(
                    "Failed conversation results cannot contain a successful tool result."
                )
            public_tool_result = {
                "status": "simulated_ok",
                "message": echo_result["message"],
            }

        public_result = {
            "user_text": self.user_text,
            "intent": self.intent,
            "success": self.success,
            "tool_result": public_tool_result,
            "error": error,
        }
        if self.informational_response is not None:
            public_result["response"] = self.informational_response
        return public_result
