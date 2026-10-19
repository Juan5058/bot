class JARVISCoreError(Exception):
    """Base exception for JARVIS CORE."""


class ConfigError(JARVISCoreError):
    """Configuration is invalid."""


class AuthorizationError(JARVISCoreError):
    """Authorization failed."""


class ToolNotRegisteredError(JARVISCoreError):
    """Attempted to use a tool that is not registered."""


class SimulatedModeError(JARVISCoreError):
    """Real execution is blocked in simulated mode."""


class ToolExecutionError(JARVISCoreError):
    """A registered simulated tool failed during execution."""


class AuditLoggingError(JARVISCoreError):
    """A required audit event could not be recorded."""


class PreExecutionAuditError(AuditLoggingError):
    """Dispatch stopped because the authorization event was not recorded."""

    def __init__(self) -> None:
        super().__init__(
            "Tool execution was stopped because authorization could not be audited."
        )


class PostExecutionAuditError(AuditLoggingError):
    """A tool may have executed, but its outcome could not be recorded."""

    def __init__(self) -> None:
        super().__init__(
            "The tool may have executed, but its outcome could not be audited."
        )


class InvalidInputError(JARVISCoreError):
    """User input cannot be processed safely."""
