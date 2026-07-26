"""Tests for demand-curve simulation and what-if analysis."""

import numpy as np
import pandas as pd
import pytest

from src.demand import simulate_demand_curve, what_if_analysis
from src.elasticity import ElasticityModel


@pytest.fixture
def fitted_model():
    np.random.seed(42)
    n = 300
    prices = np.random.uniform(10, 50, n)
    quantity = 5000 * (prices**-1.2) * np.exp(np.random.normal(0, 0.05, n))
    df = pd.DataFrame({"price": prices, "quantity": quantity, "segment": "A"})
    model = ElasticityModel()
    model.fit(df)
    return model


class TestSimulateDemandCurve:
    def test_returns_expected_columns_and_length(self, fitted_model):
        curve = simulate_demand_curve(fitted_model, "A", (10.0, 50.0), n_points=25)
        assert list(curve.columns) == ["price", "predicted_demand", "revenue"]
        assert len(curve) == 25

    def test_revenue_equals_price_times_demand(self, fitted_model):
        curve = simulate_demand_curve(fitted_model, "A", (10.0, 50.0), n_points=25)
        expected = curve["price"] * curve["predicted_demand"]
        np.testing.assert_allclose(curve["revenue"].values, expected.values)

    def test_demand_falls_as_price_rises(self, fitted_model):
        # Elasticity is negative, so predicted demand should be monotonically
        # decreasing across an ascending price grid.
        curve = simulate_demand_curve(fitted_model, "A", (10.0, 50.0), n_points=25)
        diffs = np.diff(curve["predicted_demand"].values)
        assert np.all(diffs < 0)


class TestWhatIfAnalysis:
    def test_one_row_per_scenario(self, fitted_model):
        scenarios = {"cut": 15.0, "hold": 25.0, "raise": 35.0}
        result = what_if_analysis(fitted_model, "A", scenarios)
        assert len(result) == 3
        assert set(result["scenario"]) == set(scenarios)

    def test_cheaper_scenario_sells_more(self, fitted_model):
        result = what_if_analysis(fitted_model, "A", {"cheap": 15.0, "pricey": 40.0})
        by_name = result.set_index("scenario")
        assert by_name.loc["cheap", "predicted_demand"] > by_name.loc["pricey", "predicted_demand"]
