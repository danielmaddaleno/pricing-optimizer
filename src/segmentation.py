"""Product segmentation via K-Means clustering for group pricing."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def segment_products(
    df: pd.DataFrame,
    features: list[str],
    n_clusters: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign each product to a pricing segment via K-Means.

    Parameters
    ----------
    df : pd.DataFrame
        Product-level data.
    features : list[str]
        Columns to use for clustering (e.g. avg_price, avg_volume, margin).
    n_clusters : int
        Number of segments.

    Returns
    -------
    Original DataFrame with an added ``segment`` column.
    """
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)

    df = df.copy()
    df["segment"] = [f"segment_{label}" for label in labels]

    for seg in sorted(df["segment"].unique()):
        subset = df[df["segment"] == seg]
        logger.info(
            "Segment %-12s  n=%4d  avg_price=$%.2f",
            seg,
            len(subset),
            subset[features[0]].mean(),
        )

    return df
