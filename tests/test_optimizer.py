"""Tests for the price optimizer."""

import logging

import numpy as np
import pandas as pd
import pytest

from src.elasticity import ElasticityModel
from src.optimizer import PriceOptimizer


@pytest.fixture
def fitted_model():
    np.random.seed(42)
    n = 300
    prices = np.random.uniform(10, 50, n)
    quantity = 5000 * (prices**-1.3) * np.exp(np.random.normal(0, 0.05, n))
    df = pd.DataFrame({"price": prices, "quantity": quantity, "segment": "A"})
    model = ElasticityModel()
    model.fit(df)
    return model


class TestPriceOptimizer:
    def test_optimize_returns_dataframe(self, fitted_model):
        products = pd.DataFrame(
            {
                "product": ["SKU-001"],
                "segment": ["A"],
                "current_price": [25.0],
                "unit_cost": [10.0],
            }
        )
        opt = PriceOptimizer(fitted_model, min_margin=0.15, max_change=0.10)
        result = opt.optimize(products)
        assert "optimal_price" in result.columns
        assert len(result) == 1

    def test_optimal_within_bounds(self, fitted_model):
        products = pd.DataFrame(
            {
                "product": ["SKU-002"],
                "segment": ["A"],
                "current_price": [30.0],
                "unit_cost": [12.0],
            }
        )
        opt = PriceOptimizer(fitted_model, min_margin=0.15, max_change=0.10)
        result = opt.optimize(products)
        row = result.iloc[0]
        assert row["optimal_price"] >= 30.0 * 0.90 - 0.01
        assert row["optimal_price"] <= 30.0 * 1.10 + 0.01

    def test_revenue_lift_non_negative(self, fitted_model):
        products = pd.DataFrame(
            {
                "product": ["SKU-003"],
                "segment": ["A"],
                "current_price": [20.0],
                "unit_cost": [8.0],
            }
        )
        opt = PriceOptimizer(fitted_model, min_margin=0.15, max_change=0.20)
        result = opt.optimize(products)
        assert result.iloc[0]["revenue_lift_pct"] >= -0.01

    def test_warns_when_segment_elasticity_not_significant(self, caplog):
        # A segment whose quantity does not move with price has an elasticity
        # indistinguishable from zero, so the optimizer should flag it.
        flat = pd.DataFrame(
            {
                "price": np.linspace(10, 50, 40),
                "quantity": [100.0] * 40,
                "segment": "flat",
            }
        )
        model = ElasticityModel()
        model.fit(flat)
        assert not model.result_for("flat").significant

        products = pd.DataFrame(
            {
                "product": ["SKU-flat"],
                "segment": ["flat"],
                "current_price": [25.0],
                "unit_cost": [10.0],
            }
        )
        opt = PriceOptimizer(model)
        with caplog.at_level(logging.WARNING):
            opt.optimize(products)
        assert "not statistically significant" in caplog.text

    @pytest.mark.parametrize("bad_margin", [1.0, 1.5, -0.1])
    def test_rejects_out_of_range_min_margin(self, fitted_model, bad_margin):
        with pytest.raises(ValueError, match="min_margin"):
            PriceOptimizer(fitted_model, min_margin=bad_margin)

    def test_rejects_negative_max_change(self, fitted_model):
        with pytest.raises(ValueError, match="max_change"):
            PriceOptimizer(fitted_model, max_change=-0.05)

    def test_no_warning_for_significant_segment(self, fitted_model, caplog):
        products = pd.DataFrame(
            {
                "product": ["SKU-004"],
                "segment": ["A"],
                "current_price": [25.0],
                "unit_cost": [10.0],
            }
        )
        opt = PriceOptimizer(fitted_model)
        with caplog.at_level(logging.WARNING):
            opt.optimize(products)
        assert "not statistically significant" not in caplog.text

    @pytest.mark.parametrize("bad_cost", [float("nan"), float("inf"), -1.0])
    def test_rejects_unusable_unit_cost(self, fitted_model, bad_cost):
        # max(price_floor, nan) returns the price floor, so a NaN cost used to
        # drop the margin constraint without a word and let the optimizer
        # recommend a price below cost.
        products = pd.DataFrame(
            {
                "product": ["SKU-005"],
                "segment": ["A"],
                "current_price": [25.0],
                "unit_cost": [bad_cost],
            }
        )
        opt = PriceOptimizer(fitted_model, min_margin=0.15, max_change=0.10)
        with pytest.raises(ValueError, match="unit_cost"):
            opt.optimize(products)

    def test_margin_floor_binds_when_cost_is_high(self, fitted_model):
        # Cost 24 with a 15% margin floor puts the cheapest legal price at
        # 28.24, above the -10% change bound, so the floor is what decides.
        products = pd.DataFrame(
            {
                "product": ["SKU-006"],
                "segment": ["A"],
                "current_price": [30.0],
                "unit_cost": [24.0],
            }
        )
        opt = PriceOptimizer(fitted_model, min_margin=0.15, max_change=0.10)
        row = opt.optimize(products).iloc[0]
        assert row["optimal_price"] >= 24.0 / 0.85 - 0.01
        assert row["optimal_price"] == pytest.approx(24.0 / 0.85, abs=0.01)
