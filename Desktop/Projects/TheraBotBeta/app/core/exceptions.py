class TheraBotError(Exception):
    """Base exception for all application errors."""


class LLMProviderError(TheraBotError):
    """Raised when an LLM provider call fails."""


class BudgetExceededError(TheraBotError):
    """Raised when the monthly token budget hard limit is reached."""
