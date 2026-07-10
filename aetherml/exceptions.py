"""AetherML exception hierarchy.

All custom exceptions inherit from AetherMLError so callers can catch
the entire SDK surface with a single clause while still being able to
target specific failure modes.
"""


class AetherMLError(Exception):
    """Base exception for all AetherML errors."""


class ConfigurationError(AetherMLError):
    """Raised when required configuration is missing or invalid."""


class DataError(AetherMLError):
    """Raised for data loading, validation, or transformation failures."""


class DataLoadError(DataError):
    """Raised when data cannot be loaded from the specified source."""


class DataTransformError(DataError):
    """Raised when a data transformation fails."""


class DataValidationError(DataError):
    """Raised when data fails schema or quality validation checks."""


class EngineError(AetherMLError):
    """Raised when a computation engine operation fails."""


class EngineSelectionError(EngineError):
    """Raised when no suitable engine can be selected for the given data."""


class WorkflowError(AetherMLError):
    """Raised when the LangGraph workflow encounters a failure."""


class AgentError(AetherMLError):
    """Raised when an agent execution fails."""


class AgentNotImplementedError(AgentError):
    """Raised by agent stubs that have not been implemented yet."""
