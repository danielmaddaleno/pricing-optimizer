"""Revenue-maximizing price optimizer with business constraints."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.elasticity import ElasticityModel

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    product: str
    segment: str
    current_price: float
    optimal_price: float
    expected_demand: float
    expected_revenue: float
    revenue_lift_pct: float


class PriceOptimizer:
    """Find revenue-maximizing prices subject to business rules.

    Parameters
    ----------
    model : ElasticityModel
        Fitted elasticity model.
    min_margin : float
        Minimum gross margin (e.g. 0.15 = 15 %).
    max_change : float
        Maximum allowed price change from current (e.g. 0.10 = ±10 %).
    cost_col : str
        Column in data with unit cost for margin constraint.
    """

    def __init__(
        self,
        model: ElasticityModel,
        min_margin: float = 0.15,
        max_change: float = 0.10,
        cost_col: str = "unit_cost",
    ):
        self.model = model
        self.min_margin = min_margin
        self.max_change = max_change
        self.cost_col = cost_col

    def optimize(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Optimize price for every row in *products_df*.

        Expects columns: product, segment, current_price, unit_cost.
        """
        self._warn_insignificant_segments(products_df["segment"].unique())

        results: list[OptimizationResult] = []

        for _, row in products_df.iterrows():
            res = self._optimize_product(
                product=row["product"],
                segment=row["segment"],
                current_price=row["current_price"],
                unit_cost=row[self.cost_col],
            )
            results.append(res)

        return pd.DataFrame([r.__dict__ for r in results])

    def _warn_insignificant_segments(self, segments) -> None:
        """Log a warning for any segment whose elasticity is not significant.

        Optimizing a price off an elasticity we cannot distinguish from zero
        is guesswork, so we surface it once per segment instead of silently
        trusting the number.
        """
        for segment in segments:
            result = self.model.result_for(segment)
            if result is not None and not result.significant:
                logger.warning(
                    "segment %r elasticity is not statistically significant (p=%.3f); price is unreliable",
                    segment,
                    result.p_value,
                )

    def _optimize_product(
        self,
        product: str,
        segment: str,
        current_price: float,
        unit_cost: float,
    ) -> OptimizationResult:
        # Price bounds
        min_price = max(
            current_price * (1 - self.max_change),
            unit_cost / (1 - self.min_margin),  # margin floor
        )
        max_price = current_price * (1 + self.max_change)

        if min_price >= max_price:
            # Infeasible, return the current price
            demand = self.model.predict_demand(segment, np.array([current_price]))[0]
            return OptimizationResult(
                product=product,
                segment=segment,
                current_price=current_price,
                optimal_price=current_price,
                expected_demand=demand,
                expected_revenue=current_price * demand,
                revenue_lift_pct=0.0,
            )

        def neg_revenue(p: float) -> float:
            d = self.model.predict_demand(segment, np.array([p]))[0]
            return -(p * d)

        result = minimize_scalar(
            neg_revenue,
            bounds=(min_price, max_price),
            method="bounded",
        )

        opt_price = result.x
        opt_demand = self.model.predict_demand(segment, np.array([opt_price]))[0]
        cur_demand = self.model.predict_demand(segment, np.array([current_price]))[0]

        cur_rev = current_price * cur_demand
        opt_rev = opt_price * opt_demand
        lift = ((opt_rev - cur_rev) / cur_rev * 100) if cur_rev > 0 else 0.0

        logger.info(
            "%-20s  $%.2f -> $%.2f  (lift %.1f%%)",
            product,
            current_price,
            opt_price,
            lift,
        )

        return OptimizationResult(
            product=product,
            segment=segment,
            current_price=round(current_price, 2),
            optimal_price=round(opt_price, 2),
            expected_demand=round(opt_demand, 1),
            expected_revenue=round(opt_rev, 2),
            revenue_lift_pct=round(lift, 2),
        )
