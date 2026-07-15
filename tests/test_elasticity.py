"""Tests for elasticity estimation."""

import numpy as np
import pandas as pd
import pytest

from src.elasticity import ElasticityModel


@pytest.fixture
def synthetic_data():
    """Generate data with known elasticity ≈ -1.5."""
    np.random.seed(42)
    n = 500
    prices = np.random.uniform(5, 50, n)
    # Q = A * P^beta  with beta = -1.5
    A = 10000
    beta = -1.5
    noise = np.exp(np.random.normal(0, 0.1, n))
    quantity = A * (prices**beta) * noise

    return pd.DataFrame(
        {
            "price": prices,
            "quantity": quantity,
            "segment": "electronics",
        }
    )


class TestElasticityModel:
    def test_fit_returns_results(self, synthetic_data):
        model = ElasticityModel()
        results = model.fit(synthetic_data)
        assert len(results) == 1
        assert results[0].segment == "electronics"

    def test_elasticity_close_to_true(self, synthetic_data):
        model = ElasticityModel()
        results = model.fit(synthetic_data)
        # True elasticity is -1.5
        assert -2.0 < results[0].elasticity < -1.0

    def test_r_squared_reasonable(self, synthetic_data):
        model = ElasticityModel()
        results = model.fit(synthetic_data)
        assert results[0].r_squared > 0.85

    def test_predict_demand_decreases_with_price(self, synthetic_data):
        model = ElasticityModel()
        model.fit(synthetic_data)
        d_low = model.predict_demand("electronics", np.array([10.0]))[0]
        d_high = model.predict_demand("electronics", np.array([40.0]))[0]
        assert d_low > d_high

    def test_fit_skips_nonpositive_and_nan_rows(self, synthetic_data):
        # log-log OLS can't take price/quantity <= 0 or NaN, so the fit drops
        # those rows. n_observations should count only the usable ones.
        n_clean = len(synthetic_data)
        dirty = pd.concat(
            [
                synthetic_data,
                pd.DataFrame(
                    {
                        "price": [0.0, -5.0, 20.0, np.nan],
                        "quantity": [100.0, 100.0, np.nan, 100.0],
                        "segment": ["electronics"] * 4,
                    }
                ),
            ],
            ignore_index=True,
        )
        results = ElasticityModel().fit(dirty)
        assert results[0].n_observations == n_clean
