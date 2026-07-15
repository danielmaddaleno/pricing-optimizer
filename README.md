![Tests](https://github.com/danielmaddaleno/pricing-optimizer/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# pricing-optimizer

Price elasticity modeling and constrained revenue optimization. Estimates demand curves from historical price/quantity data and finds the revenue-maximizing price for each product segment.

## Approach

1. **Elasticity estimation**: log-log OLS regression to estimate price elasticity of demand per product segment
2. **Demand simulation**: predict volume at candidate price points from the fitted model
3. **Revenue optimization**: maximize `price * predicted_demand` subject to business constraints (min margin, max price change)
4. **What-if analysis**: compare demand and revenue across a set of pricing scenarios

## Features

- Elasticity models per product segment (scikit-learn linear regression on log-transformed price/quantity)
- Bounded revenue maximization with `scipy.optimize.minimize_scalar`
- Optional control variables (e.g. competitor price index) as extra log-space regressors
- Configurable business rules: min margin %, max price change %
- K-Means product segmentation for group pricing

## Project structure

```
src/
├── elasticity.py       # Price-elasticity estimation (log-log OLS)
├── optimizer.py        # Revenue maximization with constraints
├── demand.py           # Demand simulation at candidate prices
├── segmentation.py     # Product clustering for group pricing
├── config.py           # Business rules & constraints
tests/
├── test_elasticity.py
├── test_optimizer.py
README.md
requirements.txt
```

## Installation

```bash
git clone https://github.com/danielmaddaleno/pricing-optimizer.git
cd pricing-optimizer
pip install -r requirements.txt
```

## Quick start

```python
from src.elasticity import ElasticityModel
from src.optimizer import PriceOptimizer

model = ElasticityModel()
model.fit(historical_df)

optimizer = PriceOptimizer(model, min_margin=0.15, max_change=0.10)
results = optimizer.optimize(current_prices_df)
print(results[["product", "current_price", "optimal_price", "revenue_lift_pct"]])
```

`historical_df` needs `price`, `quantity`, and `segment` columns. `current_prices_df` needs `product`, `segment`, `current_price`, and `unit_cost`.

## Development

```bash
make install  # Install deps
make test     # Run tests
make lint     # Linters
```

## Limitations

Elasticity is estimated per segment with plain OLS on log-transformed data, no regularization or cross-validation. Works well on the synthetic data in the test suite; on real data you'd want to check residuals and consider a more robust estimator before trusting the coefficients.

## Roadmap

- [ ] Improve test coverage
- [ ] Add benchmarks
- [ ] Docker support

## License

MIT
