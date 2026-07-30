"""Tests for K-Means product segmentation."""

import numpy as np
import pandas as pd
import pytest

from src.segmentation import segment_products


@pytest.fixture
def products():
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "avg_price": np.random.uniform(5, 50, n),
            "avg_volume": np.random.uniform(1, 100, n),
            "margin": np.random.uniform(0.1, 0.6, n),
        }
    )


FEATURES = ["avg_price", "avg_volume", "margin"]


class TestSegmentProducts:
    def test_adds_segment_column_without_dropping_rows(self, products):
        out = segment_products(products, FEATURES, n_clusters=4)
        assert "segment" in out.columns
        assert len(out) == len(products)

    def test_produces_requested_number_of_segments(self, products):
        out = segment_products(products, FEATURES, n_clusters=4)
        assert out["segment"].nunique() == 4

    def test_does_not_mutate_input_frame(self, products):
        before = products.copy()
        segment_products(products, FEATURES, n_clusters=4)
        assert "segment" not in products.columns
        assert products.equals(before)

    def test_same_random_state_gives_stable_labels(self, products):
        first = segment_products(products, FEATURES, n_clusters=4, random_state=7)
        second = segment_products(products, FEATURES, n_clusters=4, random_state=7)
        assert (first["segment"].values == second["segment"].values).all()

    def test_labels_follow_segment_prefix(self, products):
        out = segment_products(products, FEATURES, n_clusters=3)
        assert all(label.startswith("segment_") for label in out["segment"])
