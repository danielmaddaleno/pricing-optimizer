"""Price-elasticity estimation via log-log OLS regression.

The standard model: ln(Q) = alpha + beta * ln(P) + gamma * X + epsilon
where beta is the price elasticity of demand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


@dataclass
class ElasticityResult:
    """Container for elasticity estimation output."""

    segment: str
    elasticity: float
    r_squared: float
    n_observations: int


class ElasticityModel:
    """Estimate price elasticity per product segment using log-log OLS.

    Parameters
    ----------
    price_col : str
        Column with unit price.
    demand_col : str
        Column with quantity sold.
    segment_col : str | None
        Column to group by for segment-level estimation.
    controls : list[str] | None
        Extra regressors (e.g. competitor price index, season dummy).
    """

    def __init__(
        self,
        price_col: str = "price",
        demand_col: str = "quantity",
        segment_col: str | None = "segment",
        controls: list[str] | None = None,
    ):
        self.price_col = price_col
        self.demand_col = demand_col
        self.segment_col = segment_col
        self.controls = controls or []
        self._models: dict[str, LinearRegression] = {}
        self._results: list[ElasticityResult] = []

    def fit(self, df: pd.DataFrame) -> list[ElasticityResult]:
        """Fit one log-log model per segment.

        Returns list of :class:`ElasticityResult`.
        """
        self._results = []
        segments = df[self.segment_col].unique() if self.segment_col else ["_all"]

        for seg in segments:
            subset = df[df[self.segment_col] == seg] if self.segment_col else df
            result = self._fit_segment(seg, subset)
            self._results.append(result)
            logger.info(
                "Segment %-20s  elasticity=%.3f  R²=%.3f  n=%d",
                seg,
                result.elasticity,
                result.r_squared,
                result.n_observations,
            )

        return self._results

    def _fit_segment(self, segment: str, df: pd.DataFrame) -> ElasticityResult:
        df = df.dropna(subset=[self.price_col, self.demand_col])
        df = df[(df[self.price_col] > 0) & (df[self.demand_col] > 0)]

        ln_p = np.log(df[self.price_col].values).reshape(-1, 1)
        ln_q = np.log(df[self.demand_col].values)

        # Add control variables in log-space if numeric
        X = ln_p
        if self.controls:
            ctrl = df[self.controls].values
            ctrl = np.where(ctrl > 0, np.log(ctrl), ctrl)
            X = np.hstack([ln_p, ctrl])

        model = LinearRegression()
        model.fit(X, ln_q)
        self._models[segment] = model

        return ElasticityResult(
            segment=segment,
            elasticity=float(model.coef_[0]),
            r_squared=float(model.score(X, ln_q)),
            n_observations=len(df),
        )

    def predict_demand(self, segment: str, prices: np.ndarray, controls: np.ndarray | None = None) -> np.ndarray:
        """Predict demand for given *prices* using the fitted segment model."""
        model = self._models[segment]
        ln_p = np.log(prices).reshape(-1, 1)
        X = ln_p
        if controls is not None:
            X = np.hstack([ln_p, np.log(np.where(controls > 0, controls, 1))])
        ln_q = model.predict(X)
        return np.exp(ln_q)

    @property
    def results(self) -> list[ElasticityResult]:
        return self._results
