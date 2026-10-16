from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

from jarvis.config.validation import validate_permission_name


_TOOL_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class SecurityPolicy:
    __slots__ = (
        "default_deny",
        "mode",
        "identity_required",
        "_allowed_permissions",
        "_allowed_tool_permissions",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {
            "default_deny",
            "mode",
            "identity_required",
            "_allowed_permissions",
            "_allowed_tool_permissions",
        } and hasattr(self, name):
            raise AttributeError(f"{name} is immutable after policy construction.")
        if name in {"allowed_permissions", "allowed_tool_permissions"}:
            raise AttributeError(f"{name} is read-only; define the policy at construction time.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in {
            "default_deny",
            "mode",
            "identity_required",
            "_allowed_permissions",
            "_allowed_tool_permissions",
            "allowed_permissions",
            "allowed_tool_permissions",
        }:
            raise AttributeError(f"{name} cannot be deleted from a security policy.")
        super().__delattr__(name)

    def __init__(
        self,
        *,
        default_deny: bool = True,
        mode: str = "simulated",
        identity_required: bool = False,
        allowed_permissions: Iterable[str] | None = None,
        allowed_tool_permissions: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        if not isinstance(mode, str):
            raise TypeError("mode must be a string.")
        if mode != "simulated":
            raise ValueError("Fase 1 only supports simulated mode.")
        if not isinstance(default_deny, bool):
            raise TypeError("default_deny must be a boolean.")
        if default_deny is not True:
            raise ValueError("Fase 1 requires deny-by-default policy semantics.")
        if not isinstance(identity_required, bool):
            raise TypeError("identity_required must be a boolean.")

        self.default_deny = True
        self.mode = mode
        self.identity_required = bool(identity_required)
        self._allowed_permissions = self._normalize_permissions(allowed_permissions)
        self._allowed_tool_permissions = self._normalize_tool_permissions(
            allowed_tool_permissions,
            self._allowed_permissions,
        )

    @property
    def allowed_permissions(self) -> frozenset[str]:
        return self._allowed_permissions

    @property
    def allowed_tool_permissions(self) -> frozenset[tuple[str, str]]:
        return self._allowed_tool_permissions

    @allowed_permissions.setter
    def allowed_permissions(self, value: Iterable[str]) -> None:
        raise AttributeError("allowed_permissions is read-only; define the policy at construction time.")

    @allowed_tool_permissions.setter
    def allowed_tool_permissions(self, value: Iterable[tuple[str, str]]) -> None:
        raise AttributeError("allowed_tool_permissions is read-only; define the policy at construction time.")

    @classmethod
    def for_application(cls) -> "SecurityPolicy":
        return cls(
            allowed_permissions={"tool.echo_message"},
            allowed_tool_permissions={("echo_message", "tool.echo_message")},
        )

    @staticmethod
    def _normalize_permissions(permissions: Iterable[str] | None) -> frozenset[str]:
        if permissions is None:
            return frozenset()
        if isinstance(permissions, str):
            raise TypeError("allowed_permissions must be an iterable of permission strings, not a single string.")
        if not isinstance(permissions, Iterable):
            raise TypeError("allowed_permissions must be an iterable of permission strings.")
        normalized: set[str] = set()
        for permission_name in permissions:
            normalized.add(validate_permission_name(permission_name))
        return frozenset(normalized)

    @staticmethod
    def _normalize_tool_permissions(
        allowed_tool_permissions: Iterable[tuple[str, str]] | None,
        allowed_permissions: frozenset[str],
    ) -> frozenset[tuple[str, str]]:
        if allowed_tool_permissions is None:
            return frozenset()
        if isinstance(allowed_tool_permissions, (str, bytes)) or not isinstance(
            allowed_tool_permissions,
            Iterable,
        ):
            raise TypeError("allowed_tool_permissions must be an iterable of tool-permission pairs.")

        normalized: set[tuple[str, str]] = set()
        for pair in allowed_tool_permissions:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("Each allowed tool permission must be a two-item tuple.")
            tool_name, permission_name = pair
            if type(tool_name) is not str or not _TOOL_NAME_PATTERN.fullmatch(tool_name):
                raise ValueError("Allowed tool name has an invalid format.")
            permission = validate_permission_name(permission_name)
            if permission not in allowed_permissions:
                raise ValueError("Tool permission pairs must also be allowed by permission policy.")
            normalized.add((tool_name, permission))
        return frozenset(normalized)

    def can_execute(self, permission_name: str) -> bool:
        if type(permission_name) is not str or not permission_name.strip():
            return False
        if self.default_deny:
            return permission_name in self._allowed_permissions
        return True

    def can_execute_tool(self, tool_name: str, permission_name: str) -> bool:
        if type(tool_name) is not str or type(permission_name) is not str:
            return False
        if self.default_deny and (tool_name, permission_name) not in self._allowed_tool_permissions:
            return False
        return self.can_execute(permission_name)

    def has_rule_for(self, permission_name: str) -> bool:
        return type(permission_name) is str and permission_name in self._allowed_permissions

    def validate_required_inputs(self, permission_name: str, context: object | None = None) -> bool:
        if type(permission_name) is not str or not permission_name.strip():
            return False
        if self.default_deny and not self.has_rule_for(permission_name):
            return False
        if self.identity_required:
            return False
        return True
