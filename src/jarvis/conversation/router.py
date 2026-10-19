from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import Any, Mapping

from jarvis.conversation.models import (
    MAX_USER_INPUT_LENGTH,
    RequestEnvelope,
    validate_tool_parameters,
)


class RouteCategory(str, Enum):
    DETERMINISTIC = "deterministic"
    SPECIALIZED = "specialized"
    GENERATIVE = "generative"
    EXTERNAL_SERVICE = "external_service"
    CLARIFY = "clarify"
    REJECT = "reject"


class RouteReason(str, Enum):
    DETERMINISTIC_REQUEST = "deterministic_request"
    ECHO_REQUEST = "echo_request"
    GENERATIVE_REQUEST = "generative_request"
    EXTERNAL_SERVICE_REQUEST = "external_service_request"
    MISSING_ARGUMENT = "missing_argument"
    UNKNOWN_COMMAND = "unknown_command"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"


class Capability(str, Enum):
    ECHO_MESSAGE = "echo_message"
    ANSWER_QUESTION = "answer_question"
    SUMMARIZE_TEXT = "summarize_text"
    WEB_SEARCH = "web_search"


class RequiredField(str, Enum):
    COMMAND = "command"
    MESSAGE = "message"
    TEXT = "text"
    QUERY = "query"


class InterpretationStatus(str, Enum):
    UNDERSTOOD = "understood"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"


class InterpretationSource(str, Enum):
    DETERMINISTIC_ROUTER = "deterministic_router"


class AmbiguityCode(str, Enum):
    INTENT_UNCLEAR = "intent_unclear"
    PARAMETER_UNCLEAR = "parameter_unclear"


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True, repr=False)
class RouteDecision:
    request: RequestEnvelope
    category: RouteCategory
    reason: RouteReason
    capability: Capability | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    missing_fields: tuple[RequiredField, ...] = ()
    ambiguities: tuple[AmbiguityCode, ...] = ()
    source: InterpretationSource = InterpretationSource.DETERMINISTIC_ROUTER

    def __post_init__(self) -> None:
        if type(self.request) is not RequestEnvelope:
            raise TypeError("Interpretation request is invalid.")
        if type(self.category) is not RouteCategory:
            raise TypeError("Route category is invalid.")
        if type(self.reason) is not RouteReason:
            raise TypeError("Route reason is invalid.")
        if self.capability is not None and type(self.capability) is not Capability:
            raise TypeError("Route capability is invalid.")
        if type(self.source) is not InterpretationSource:
            raise TypeError("Interpretation source is invalid.")
        if type(self.missing_fields) is not tuple or any(
            type(item) is not RequiredField for item in self.missing_fields
        ):
            raise TypeError("Route missing fields are invalid.")
        if len(self.missing_fields) != len(set(self.missing_fields)):
            raise ValueError("Route missing fields must be unique.")
        if type(self.ambiguities) is not tuple or any(
            type(item) is not AmbiguityCode for item in self.ambiguities
        ):
            raise TypeError("Interpretation ambiguities are invalid.")
        if len(self.ambiguities) != len(set(self.ambiguities)):
            raise ValueError("Interpretation ambiguities must be unique.")

        raw_parameters: Any = self.parameters
        if type(raw_parameters) is not dict:
            raise TypeError("Interpretation parameters must be a dictionary.")
        validate_tool_parameters(raw_parameters)
        copied_parameters = deepcopy(raw_parameters)
        validate_tool_parameters(copied_parameters)
        object.__setattr__(self, "parameters", _freeze_json(copied_parameters))

        if self.category is RouteCategory.CLARIFY:
            if not (self.missing_fields or self.ambiguities):
                raise ValueError("Clarification routes require missing fields or ambiguities.")
        elif self.missing_fields or self.ambiguities:
            raise ValueError("Only clarification routes may have missing fields or ambiguities.")

        expected_reasons = {
            RouteCategory.DETERMINISTIC: {RouteReason.DETERMINISTIC_REQUEST},
            RouteCategory.SPECIALIZED: {RouteReason.ECHO_REQUEST},
            RouteCategory.GENERATIVE: {RouteReason.GENERATIVE_REQUEST},
            RouteCategory.EXTERNAL_SERVICE: {
                RouteReason.EXTERNAL_SERVICE_REQUEST
            },
            RouteCategory.CLARIFY: {
                RouteReason.MISSING_ARGUMENT,
                RouteReason.UNKNOWN_COMMAND,
            },
            RouteCategory.REJECT: {RouteReason.UNSUPPORTED_CAPABILITY},
        }
        if self.reason not in expected_reasons[self.category]:
            raise ValueError("Route reason does not match its category.")

        if self.category is RouteCategory.DETERMINISTIC and (
            self.capability is not None or self.parameters
        ):
            raise ValueError("Deterministic routes cannot contain tool arguments.")
        if self.category is RouteCategory.SPECIALIZED and (
            self.capability is not Capability.ECHO_MESSAGE
            or "message" not in self.parameters
            or type(self.parameters["message"]) is not str
            or not self.parameters["message"].strip()
            or len(self.parameters["message"]) > MAX_USER_INPUT_LENGTH
        ):
            raise ValueError("Specialized routes require a supported capability.")
        if self.category is RouteCategory.GENERATIVE and (
            self.capability
            not in {Capability.ANSWER_QUESTION, Capability.SUMMARIZE_TEXT}
            or "text" not in self.parameters
            or type(self.parameters["text"]) is not str
            or not self.parameters["text"].strip()
            or len(self.parameters["text"]) > MAX_USER_INPUT_LENGTH
        ):
            raise ValueError("Generative routes require a generative capability.")
        if self.category is RouteCategory.EXTERNAL_SERVICE and (
            self.capability is not Capability.WEB_SEARCH
            or set(self.parameters) != {"query"}
            or type(self.parameters["query"]) is not str
            or not self.parameters["query"].strip()
            or len(self.parameters["query"]) > MAX_USER_INPUT_LENGTH
        ):
            raise ValueError("External service routes require a supported query.")
        if self.category is RouteCategory.REJECT and (
            self.capability is not None or self.parameters
        ):
            raise ValueError("Rejected routes cannot contain executable arguments.")
        if self.category is RouteCategory.CLARIFY:
            if self.parameters:
                raise ValueError("Clarification routes cannot contain parameters.")
            if self.reason is RouteReason.UNKNOWN_COMMAND and (
                self.capability is not None
                or (
                    self.missing_fields != (RequiredField.COMMAND,)
                    and not self.ambiguities
                )
            ):
                raise ValueError("Unknown commands must request command clarification.")
            if self.reason is RouteReason.MISSING_ARGUMENT and (
                self.capability is None
                or len(self.missing_fields) != 1
                or (
                    self.capability is Capability.ECHO_MESSAGE
                    and self.missing_fields != (RequiredField.MESSAGE,)
                )
                or (
                    self.capability
                    in {Capability.ANSWER_QUESTION, Capability.SUMMARIZE_TEXT}
                    and self.missing_fields != (RequiredField.TEXT,)
                )
                or (
                    self.capability is Capability.WEB_SEARCH
                    and self.missing_fields != (RequiredField.QUERY,)
                )
            ):
                raise ValueError("Missing arguments must identify one required field.")

    @property
    def status(self) -> InterpretationStatus:
        if self.category is RouteCategory.CLARIFY:
            return InterpretationStatus.NEEDS_CLARIFICATION
        if self.category is RouteCategory.REJECT:
            return InterpretationStatus.UNSUPPORTED
        return InterpretationStatus.UNDERSTOOD

    @property
    def intent(self) -> Capability | None:
        return self.capability

    @property
    def message(self) -> str | None:
        key = "message" if "message" in self.parameters else "text"
        value = self.parameters.get(key)
        return value if type(value) is str else None

    @property
    def needs_clarification(self) -> bool:
        return self.status is InterpretationStatus.NEEDS_CLARIFICATION

    def mutable_parameters(self) -> dict[str, Any]:
        return _thaw_json(self.parameters)

    def __repr__(self) -> str:
        return "RouteDecision(<validated, redacted>)"


_SUPPORTED_COMMANDS = frozenset({"echo", "repite"})
_GENERATIVE_COMMANDS = {
    "pregunta": Capability.ANSWER_QUESTION,
    "resumen": Capability.SUMMARIZE_TEXT,
}
_EXTERNAL_SERVICE_COMMANDS = {
    "web": Capability.WEB_SEARCH,
    "busca en internet": Capability.WEB_SEARCH,
}
_UNSUPPORTED_ACTION = re.compile(
    r"^(?:abre|abrir|open|ejecuta|ejecutar|inicia|iniciar|cierra|cerrar|"
    r"captura|capturar|crea|crear|elimina|eliminar|mueve|mover|borra|borrar|"
    r"busca|buscar|navega|navegar|envia|enviar|manda|mandar)\b",
    re.IGNORECASE,
)


class DeterministicRequestRouter:
    __slots__ = ()

    def route(self, request: RequestEnvelope) -> RouteDecision:
        if type(request) is not RequestEnvelope:
            raise TypeError("Router requires a validated request envelope.")

        text = request.text.strip()
        command, separator, argument = text.partition(":")
        normalized_command = command.strip().casefold()

        if normalized_command in _SUPPORTED_COMMANDS:
            if not separator or not argument.strip():
                return RouteDecision(
                    request=request,
                    category=RouteCategory.CLARIFY,
                    reason=RouteReason.MISSING_ARGUMENT,
                    capability=Capability.ECHO_MESSAGE,
                    missing_fields=(RequiredField.MESSAGE,),
                )
            return RouteDecision(
                request=request,
                category=RouteCategory.SPECIALIZED,
                reason=RouteReason.ECHO_REQUEST,
                capability=Capability.ECHO_MESSAGE,
                parameters={"message": argument.strip()},
            )

        generative_capability = _GENERATIVE_COMMANDS.get(normalized_command)
        if generative_capability is not None:
            if not separator or not argument.strip():
                return RouteDecision(
                    request=request,
                    category=RouteCategory.CLARIFY,
                    reason=RouteReason.MISSING_ARGUMENT,
                    capability=generative_capability,
                    missing_fields=(RequiredField.TEXT,),
                )
            return RouteDecision(
                request=request,
                category=RouteCategory.GENERATIVE,
                reason=RouteReason.GENERATIVE_REQUEST,
                capability=generative_capability,
                parameters={"text": argument.strip()},
            )

        external_capability = _EXTERNAL_SERVICE_COMMANDS.get(normalized_command)
        if external_capability is not None:
            if not separator or not argument.strip():
                return RouteDecision(
                    request=request,
                    category=RouteCategory.CLARIFY,
                    reason=RouteReason.MISSING_ARGUMENT,
                    capability=external_capability,
                    missing_fields=(RequiredField.QUERY,),
                )
            return RouteDecision(
                request=request,
                category=RouteCategory.EXTERNAL_SERVICE,
                reason=RouteReason.EXTERNAL_SERVICE_REQUEST,
                capability=external_capability,
                parameters={"query": argument.strip()},
            )

        if separator or _UNSUPPORTED_ACTION.match(text):
            return RouteDecision(
                request=request,
                category=RouteCategory.REJECT,
                reason=RouteReason.UNSUPPORTED_CAPABILITY,
            )

        return RouteDecision(
            request=request,
            category=RouteCategory.CLARIFY,
            reason=RouteReason.UNKNOWN_COMMAND,
            missing_fields=(RequiredField.COMMAND,),
        )
