# HDB Resale Price Regression

What predicts HDB resale prices in Singapore? A regression analysis of ~50,000 recent transactions (May 2024 – April 2026), with historical comparisons going back to 1990.

## Notebooks

1. **Download and prepare data** — load all 5 data.gov.sg datasets (1990–present), derive variables
2. **Explore distributions** — EDA on price, location, size, floor, lease, lucky numbers
3. **Download supplementary data** — geocode blocks via OneMap, compute distances to MRT/hawker/schools/columbaria/temples, build superstition variables
4. **Merge** — combine HDB transactions with supplementary data
5. **Single-variable regressions** — which predictors explain the most variance individually?
6. **Multivariable regression** — build up a 10-model progression, final model explains 90.2% of price variance
7. **Period comparison** — run the same model on 2022–2024 vs 2024–2026 to test stability
8. **Lease decay over time** — track how the market's pricing of remaining lease changed from 1990 to 2025

## Data

All data is from public sources:
- [data.gov.sg](https://data.gov.sg) — HDB resale flat prices
- [OneMap API](https://www.onemap.gov.sg) — geocoding
- [data.gov.sg](https://data.gov.sg) — school, hawker, hospital, park locations

CSV files are gitignored. Run notebooks 1–4 to regenerate them.

## Setup

```
pip install -r requirements.txt
```

R packages needed: `tidyverse`, `scales`, `sandwich`, `lmtest`, `broom`, `stargazer`
