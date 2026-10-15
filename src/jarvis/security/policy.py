from __future__ import annotations

from collections.abc import Iterable


class SecurityPolicy:
    __slots__ = ("default_deny", "mode", "identity_required", "_allowed_permissions")

    def __init__(
        self,
        *,
        default_deny: bool = True,
        mode: str = "simulated",
        identity_required: bool = False,
        allowed_permissions: Iterable[str] | None = None,
    ) -> None:
        if mode != "simulated":
            raise ValueError("Fase 1 only supports simulated mode.")
        if default_deny is not True:
            raise ValueError("Fase 1 requires deny-by-default policy semantics.")
        self.default_deny = True
        self.mode = mode
        self.identity_required = bool(identity_required)
        self._allowed_permissions = self._normalize_permissions(allowed_permissions)

    @property
    def allowed_permissions(self) -> frozenset[str]:
        return self._allowed_permissions

    @allowed_permissions.setter
    def allowed_permissions(self, value: Iterable[str]) -> None:
        raise AttributeError("allowed_permissions is read-only; define the policy at construction time.")

    @classmethod
    def for_application(cls) -> "SecurityPolicy":
        return cls(allowed_permissions={"tool.echo_message"})

    @staticmethod
    def _normalize_permissions(permissions: Iterable[str] | None) -> frozenset[str]:
        if permissions is None:
            return frozenset()
        if not isinstance(permissions, Iterable):
            raise TypeError("allowed_permissions must be an iterable of permission strings.")
        normalized: set[str] = set()
        for permission_name in permissions:
            if not isinstance(permission_name, str):
                raise TypeError("Permission names must be strings.")
            if not permission_name or not permission_name.strip() or permission_name != permission_name.strip():
                raise ValueError(f"Invalid permission name: {permission_name!r}")
            normalized.add(permission_name)
        return frozenset(normalized)

    def can_execute(self, permission_name: str) -> bool:
        if not isinstance(permission_name, str) or not permission_name.strip():
            return False
        if self.default_deny:
            return permission_name in self._allowed_permissions
        return True

    def has_rule_for(self, permission_name: str) -> bool:
        return isinstance(permission_name, str) and permission_name in self._allowed_permissions

    def validate_required_inputs(self, permission_name: str, context: object | None = None) -> bool:
        if not isinstance(permission_name, str) or not permission_name.strip():
            return False
        if self.default_deny and not self.has_rule_for(permission_name):
            return False
        if self.identity_required and context is None:
            return False
        return True
