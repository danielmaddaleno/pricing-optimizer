"""Business rules and constraint configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class PricingRules(BaseModel):
    """Configurable business constraints for optimization.

    The field bounds reject constraints that cannot describe a real pricing
    policy: a margin outside ``[0, 1)``, a negative price-change limit, a
    non-positive price floor, or a negative rounding precision. The
    cross-field check then rejects a price band whose ceiling is not above
    its floor, so a caller sees the mistake here instead of getting an empty
    feasible range deep inside the optimizer.
    """

    min_margin_pct: float = Field(default=0.15, ge=0.0, lt=1.0)
    max_price_change_pct: float = Field(default=0.10, ge=0.0)
    min_price: float = Field(default=1.0, gt=0.0)
    max_price: float = Field(default=10000.0, gt=0.0)
    round_to: int = Field(default=2, ge=0)  # decimal places

    @model_validator(mode="after")
    def _check_price_band(self) -> "PricingRules":
        if self.max_price <= self.min_price:
            raise ValueError(f"max_price ({self.max_price}) must be greater than min_price ({self.min_price})")
        return self
