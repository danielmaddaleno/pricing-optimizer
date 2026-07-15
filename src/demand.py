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
    """
    prices = np.linspace(price_range[0], price_range[1], n_points)
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
    """
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
