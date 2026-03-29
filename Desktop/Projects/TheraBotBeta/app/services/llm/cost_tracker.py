import threading

from app.core.config import get_settings
from app.core.exceptions import BudgetExceededError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Pricing per 1M tokens (USD).
# Direct-key model IDs and OpenRouter provider-prefixed slugs are listed separately
# because OpenRouter charges its own rates (not necessarily identical to direct).
# Last updated: 2026-03-29. Source: provider pricing pages + openrouter.ai/models.
_PRICE_PER_1M: dict[str, dict[str, float]] = {
    # --- OpenAI (direct) ---
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini": {"input": 0.15,  "output":  0.60},

    # --- Anthropic (direct) ---
    "claude-opus-4-6":           {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input":  0.80, "output":  4.00},

    # --- DeepSeek (direct) ---
    "deepseek-chat": {"input": 0.27, "output": 1.10},

    # --- Kimi / Moonshot (direct) ---
    "moonshot-v1-8k": {"input": 0.60, "output": 2.40},

    # --- OpenRouter slugs ---
    "openai/gpt-4o":               {"input": 2.50,  "output": 10.00},
    "anthropic/claude-sonnet-4-5": {"input": 3.00,  "output": 15.00},
    "deepseek/deepseek-chat":      {"input": 0.32,  "output":  0.89},
    "moonshotai/kimi-k2.5":        {"input": 0.42,  "output":  2.20},
}

# Applied when the model string isn't in the table.
# Set conservatively high so unknown models never appear cheaper than they are.
_FALLBACK_PRICE = {"input": 10.00, "output": 30.00}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _PRICE_PER_1M.get(model, _FALLBACK_PRICE)
    if model not in _PRICE_PER_1M:
        logger.warning("cost_model_unknown_using_fallback", model=model)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


class CostTracker:
    """Thread-safe in-process cost accumulator.

    Tracks total USD spend for the lifetime of the process and enforces
    the monthly budget threshold defined in settings.monthly_budget_usd.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record usage for one LLM call. Returns the cost for that call (USD).

        Raises BudgetExceededError if the hard dollar limit has been reached.
        Logs a warning when the soft threshold is crossed.
        """
        cost = calculate_cost(model, input_tokens, output_tokens)
        settings = get_settings()
        budget_usd = settings.monthly_budget_usd
        warn_threshold = settings.cost_alert_threshold

        with self._lock:
            self._total_cost += cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            total = self._total_cost

        usage_ratio = total / budget_usd if budget_usd > 0 else 0.0

        if usage_ratio >= 1.0:
            raise BudgetExceededError(
                f"Monthly budget exceeded (${total:.4f} / ${budget_usd:.2f})"
            )

        if usage_ratio >= warn_threshold:
            logger.warning(
                "cost_budget_warning",
                total_cost_usd=round(total, 4),
                budget_usd=round(budget_usd, 2),
                usage_pct=round(usage_ratio * 100, 1),
            )

        return cost

    @property
    def total_cost(self) -> float:
        with self._lock:
            return self._total_cost

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total_cost_usd": round(self._total_cost, 6),
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
            }
