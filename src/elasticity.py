"""Price-elasticity estimation via log-log OLS regression.

The standard model: ln(Q) = alpha + beta * ln(P) + gamma * X + epsilon
where beta is the price elasticity of demand.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


@dataclass
class ElasticityResult:
    """Container for elasticity estimation output.

    Beyond the point estimate this carries the OLS standard error, a 95%
    confidence interval and the two-sided p-value for the price coefficient,
    so a caller can tell a real elasticity from noise before pricing off it.
    Inference fields default to NaN and stay NaN when the segment has too few
    rows to estimate them.
    """

    segment: str
    elasticity: float
    r_squared: float
    n_observations: int
    std_error: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    p_value: float = float("nan")

    @property
    def significant(self) -> bool:
        """True if the elasticity differs from zero at the 5% level.

        A statistically flat elasticity means the data cannot say how demand
        responds to price, so optimizing a price off it would be guesswork.
        """
        return math.isfinite(self.p_value) and self.p_value < 0.05


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
                "Segment %-20s  elasticity=%.3f  95%% CI [%.3f, %.3f]  R²=%.3f  n=%d%s",
                seg,
                result.elasticity,
                result.ci_low,
                result.ci_high,
                result.r_squared,
                result.n_observations,
                "" if result.significant else "  (not significant)",
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

        std_error, ci_low, ci_high, p_value = self._slope_inference(X, ln_q, model)

        return ElasticityResult(
            segment=segment,
            elasticity=float(model.coef_[0]),
            r_squared=float(model.score(X, ln_q)),
            n_observations=len(df),
            std_error=std_error,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
        )

    @staticmethod
    def _slope_inference(X: np.ndarray, y: np.ndarray, model: LinearRegression) -> tuple[float, float, float, float]:
        """Standard error, 95% CI and two-sided p-value for the price slope.

        ``sklearn`` fits the coefficients but reports no uncertainty, so we
        recover the classical OLS inference by hand: rebuild the design matrix
        with an intercept column (the price slope is then column 1), estimate
        the residual variance, and read the slope variance off the diagonal of
        ``sigma^2 (X'X)^-1``. Returns NaNs when there are no residual degrees
        of freedom or the design is singular (for example a constant column).
        """
        nan = float("nan")
        n = X.shape[0]
        design = np.column_stack([np.ones(n), X])
        dof = n - design.shape[1]
        if dof <= 0:
            return nan, nan, nan, nan

        residuals = y - model.predict(X)
        sigma2 = float(residuals @ residuals) / dof
        try:
            xtx_inv = np.linalg.inv(design.T @ design)
        except np.linalg.LinAlgError:
            return nan, nan, nan, nan

        var_slope = sigma2 * float(xtx_inv[1, 1])
        if not var_slope > 0:
            return nan, nan, nan, nan

        slope = float(model.coef_[0])
        se = math.sqrt(var_slope)
        t_crit = float(stats.t.ppf(0.975, dof))
        p_value = float(2 * stats.t.sf(abs(slope / se), dof))
        return se, slope - t_crit * se, slope + t_crit * se, p_value

    def predict_demand(self, segment: str, prices: np.ndarray, controls: np.ndarray | None = None) -> np.ndarray:
        """Predict demand for given *prices* using the fitted segment model."""
        try:
            model = self._models[segment]
        except KeyError:
            available = ", ".join(map(str, sorted(self._models)))
            hint = f"available segments: {available}" if available else "call fit() first"
            raise KeyError(f"no fitted model for segment {segment!r} ({hint})") from None
        prices = np.asarray(prices, dtype=float)
        if prices.size and prices.min() <= 0:
            raise ValueError(
                f"prices must be strictly positive for a log-log demand model, got a minimum of {prices.min()}"
            )
        ln_p = np.log(prices).reshape(-1, 1)
        X = ln_p
        if controls is not None:
            X = np.hstack([ln_p, np.log(np.where(controls > 0, controls, 1))])
        ln_q = np.asarray(model.predict(X), dtype=float)
        demand: np.ndarray = np.exp(ln_q)
        return demand

    @property
    def results(self) -> list[ElasticityResult]:
        return self._results

    def result_for(self, segment: str) -> ElasticityResult | None:
        """Return the fitted result for *segment*, or None if it was not fit."""
        for result in self._results:
            if result.segment == segment:
                return result
        return None
