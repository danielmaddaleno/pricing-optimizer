"""Business rules and constraint configuration."""

from pydantic import BaseModel


class PricingRules(BaseModel):
    """Configurable business constraints for optimization."""

    min_margin_pct: float = 0.15
    max_price_change_pct: float = 0.10
    min_price: float = 1.0
    max_price: float = 10000.0
    round_to: int = 2  # decimal places
