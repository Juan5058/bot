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


class InvalidInputError(JARVISCoreError):
    """User input cannot be processed safely."""
