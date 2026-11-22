# Fractional Differentiation for Financial Machine Learning

Research whether fractional differencing of log prices preserves more statistical memory than first differencing while improving stationarity and next-day return prediction.

Fractional filter: (1-B)^d X_t = sum_k (-1)^k binom(d,k) X_(t-k), for 0<=d<=1. Coefficients are calculated by w_0=1; w_k=-w_(k-1)*(d-k+1)/k, with causal fixed-width truncation.

## Install and run

Python 3.10+:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
fracdiff-research --days 1200 --threshold .005 --output results
```

Use your own CSV with ascending, unique dates and positive point-in-time adjusted closing prices:

```csv
date,close
2023-01-03,100.02
2023-01-04,101.13
```

```bash
fracdiff-research --csv adjusted_prices.csv --output market-results
```

Outputs: `order_selection.csv`, `predictions.csv`, `metrics.json`, `research.png`.

## Methodology

1. Chronological 60% train, 20% validation, 20% test.
2. On **training prices only**, run ADF unit-root tests on orders d=0,0.1,...,1.0. Select the smallest d with p < 0.05, otherwise report fallback to the largest tested d.
3. Report train-only correlation of filtered signal to original log-price as a simple information retention proxy (not an information-theoretic measure).
4. Compare Ridge next-return forecasts using raw log price, first differencing, selected fractional difference, and return-only features. Shared causal features are last daily return, five-day momentum and ten-day rolling volatility.
5. Fit normalization and Ridge only on train. Report MSE, MAE, directional accuracy, and illustrative cost-adjusted return. All models use identical eligible dates.

## Synthetic experiment

On an 800-trading-day synthetic series (seed 42, threshold .005), selected d=0.4. Test MSE:

| Model | Test MSE |
| --- | ---: |
| Raw log price | 0.000430764 |
| Integer diff | 0.000430048 |
| Fractional diff | 0.000431661 |
| Returns only | 0.000430038 |

Fractional differentiation **did not improve test MSE** relative to the baselines. Synthetic results cannot establish tradability or efficacy in actual financial markets.

## Limitations

The finite-width filter approximates infinite fractional differencing. ADF has low power under breaks; searching multiple d values increases selection bias. Evaluate multiple markets and periods with nested walk-forward methods before drawing substantive conclusions. Daily close-t features predict close-t+1 returns; trading at the same close after observing it is not executable. Trading metrics use simplistic proportional turnover costs, and do not account for impact, financing or real fills.

14 tests cover filter boundary cases, causality, train-only order selection, data integrity, reproducibility, chronology and CLI outputs. GitHub Actions targets Python 3.10 and 3.12.

MIT license.
