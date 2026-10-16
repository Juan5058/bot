from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


_SENSITIVE_PARAMETER_KEY_PARTS = (
    "secret",
    "token",
    "password",
    "api_key",
    "access_key",
    "credential",
    "biometric",
)

_MAX_PARAMETER_DEPTH = 32
_MAX_CONTAINER_ITEMS = 256
_MAX_PARAMETER_NODES = 2048
MAX_USER_INPUT_LENGTH = 8000


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

        if value is None or type(value) in {str, bool, int}:
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
                children.append(item)
        pending.extend((item, depth + 1, False) for item in reversed(children))


def _serialize_parameter(
    value: Any,
    *,
    depth: int = 0,
    node_count: list[int] | None = None,
) -> Any:
    """
    Serialize tool proposal parameters into a restricted JSON-compatible form.

    Security properties:
    - Rejects arbitrary Python objects.
    - Rejects non-string dictionary keys.
    - Rejects NaN and Infinity.
    - Limits nesting depth.
    - Limits the number of items in each container.
    - Limits the total number of visited nodes.
    - Redacts values belonging to keys that look sensitive.
    - Handles recursive/cyclic structures through the depth/node limits.
    """
    if node_count is None:
        node_count = [0]

    node_count[0] += 1

    if node_count[0] > _MAX_PARAMETER_NODES:
        raise ValueError(
            "Proposal parameters exceed the supported size."
        )

    if value is None or type(value) in {str, bool, int}:
        return value

    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(
                "Proposal parameters cannot contain non-finite numbers."
            )
        return value

    if depth >= _MAX_PARAMETER_DEPTH:
        raise ValueError(
            "Proposal parameters exceed the supported nesting depth."
        )

    if type(value) is list:
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError(
                "Proposal parameter containers exceed the supported size."
            )

        return [
            _serialize_parameter(
                item,
                depth=depth + 1,
                node_count=node_count,
            )
            for item in value
        ]

    if type(value) is dict:
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise ValueError(
                "Proposal parameter containers exceed the supported size."
            )

        serialized: dict[str, Any] = {}

        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    "Proposal parameter keys must be strings."
                )

            if any(
                part in key.lower()
                for part in _SENSITIVE_PARAMETER_KEY_PARTS
            ):
                serialized[key] = "[REDACTED]"
            else:
                serialized[key] = _serialize_parameter(
                    item,
                    depth=depth + 1,
                    node_count=node_count,
                )

        return serialized

    raise TypeError(
        "Proposal parameters must contain only JSON-compatible values."
    )


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


@dataclass(slots=True)
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
    error: str | None = field(
        default=None,
        repr=False,
    )
    audit_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        proposal_data = None

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

            proposal_data = {
                "tool_name": proposal.tool_name,
                "parameters": _serialize_parameter(
                    proposal.parameters
                ),
                "permission": proposal.permission,
                "risk_level": proposal.risk_level,
                "requires_confirmation": proposal.requires_confirmation,
                "action_ref": proposal.action_ref,
            }

        return {
            "user_text": self.user_text,
            "intent": self.intent,
            "proposal": proposal_data,
            "authorized": self.authorized,
            "success": self.success,
            "tool_result": self.tool_result,
            "error": self.error,
            "audit_id": self.audit_id,
        }
