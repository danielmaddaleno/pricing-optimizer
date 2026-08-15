"""Demand simulation at candidate price points."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.elasticity import ElasticityModel


def simulate_demand_curve(
    model: ElasticityModel,
    segment: str,
    price_range: tuple[float, float],
    n_points: int = 50,
) -> pd.DataFrame:
    """Generate a demand curve for a segment across a price range.

    Returns DataFrame with columns: price, predicted_demand, revenue.

    Raises
    ------
    ValueError
        If ``n_points`` is below 2, or ``price_range`` is not strictly
        increasing. A single point is not a curve, and a reversed range would
        quietly produce a descending price grid that breaks the usual
        "demand falls as price rises" reading of the output.
    """
    low, high = price_range
    if n_points < 2:
        raise ValueError(f"a demand curve needs at least 2 price points, got n_points={n_points}")
    if not low < high:
        raise ValueError(f"price_range must be strictly increasing (low < high), got {price_range}")

    prices = np.linspace(low, high, n_points)
    demands = model.predict_demand(segment, prices)
    revenues = prices * demands

    return pd.DataFrame(
        {
            "price": prices,
            "predicted_demand": demands,
            "revenue": revenues,
        }
    )


def what_if_analysis(
    model: ElasticityModel,
    segment: str,
    scenarios: dict[str, float],
) -> pd.DataFrame:
    """Compare multiple pricing scenarios for a segment.

    Parameters
    ----------
    scenarios : dict[str, float]
        Mapping of scenario name -> price.

    Returns
    -------
    DataFrame with scenario, price, demand, revenue columns.

    Raises
    ------
    ValueError
        If ``scenarios`` is empty. With no scenarios the result would be a
        DataFrame carrying none of the documented columns, so a downstream
        access like ``result["revenue"]`` would raise a KeyError far from the
        real cause. Failing here says exactly what is missing.
    """
    if not scenarios:
        raise ValueError("scenarios is empty; pass at least one name -> price to compare")

    rows = []
    for name, price in scenarios.items():
        demand = model.predict_demand(segment, np.array([price]))[0]
        rows.append(
            {
                "scenario": name,
                "price": price,
                "predicted_demand": round(demand, 1),
                "revenue": round(price * demand, 2),
            }
        )
    return pd.DataFrame(rows)
