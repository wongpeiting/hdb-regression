# HDB Resale Price Regression

What predicts HDB resale prices in Singapore? A regression analysis of 51,748 recent transactions (May 2024 – April 2026) and a 35-year historical trend analysis of 976,261 transactions (1990–2026). Built for data journalism — the coefficients are the story, the residuals are the leads.

The final model (log-price with terrace interaction and cubic lease term, 88 variables) explains 94% of resale price variance. Key finding: number-linked superstitions have a measurable effect on HDB prices, and the effect has grown roughly sevenfold since the early 1990s.

## Notebooks

| # | Purpose |
|---|---------|
| 1 | Download from data.gov.sg, derive variables |
| 2 | EDA: distributions, histograms, boxplots |
| 3 | Geocode via OneMap, compute distances to amenities |
| 4 | Merge HDB + supplementary into master table |
| 5 | Single-variable regressions (each X vs Y alone) |
| 6 | Multivariable regression (10-model step-up) |
| 7 | Period comparison (2022–24 vs 2024–26) |
| 8 | Lease decay over time (1990–2025) |
| 9 | Interaction terms (storey×town, lease×town) |
| 10 | Matched pairs for storytelling |
| 11 | Log-price model, terrace interaction |
| 12 | Model diagnostics (Q-Q, Cook's D, VIF, AIC/BIC) |
| 13 | Robustness checks (median regression, hold-out validation) |
| 14 | Revised final model, editorial decision on model selection |
| 14a | Final model diagnostics (residuals, Q-Q, Cook's D, heteroskedasticity) |
| 14b | Final model robustness (train/test split, raw-price spec, Cook's D removal) |
| 15 | 168 vs 8 analysis by price segment, PropertyGuru listing analysis |
| 16 | Superstition over time (35 years, 18 two-year windows) |
| 17 | High-value superstition (>$800K segment) |
| 18 | Block number digit significance (town vs street-level FE) |
| 19 | Story charts and Datawrapper data exports |

## Key findings

- **Block-4 penalty:** Flats in blocks containing the digit 4 sell for about $8,500 less (1.3% at the median price). The penalty has grown from $600 in 1990–91 to $13,200 in 2024–25.
- **Trailing-8 premium:** Each trailing 8 in the price adds about $1,600 (0.25%). Survives every robustness check including outlier removal and raw-price specification.
- **888 prevalence:** 0.02% of transactions ended in 888 in 1990–91. By 2024–25, 9.4% did.
- **168 anchor:** 24 flats priced at exactly $1,168,000–$1,168,888 all sold above predicted value, by $105,000 on average (10%).

## Data

All data is from public sources:
- [data.gov.sg](https://data.gov.sg) — HDB resale flat prices (976,261 transactions, 1990–2026)
- [OneMap API](https://www.onemap.gov.sg) — geocoding
- [data.gov.sg](https://data.gov.sg) — school, hawker, hospital, park, temple, columbarium locations
- [PropertyGuru](https://www.propertyguru.com.sg) — 13,338 HDB listings as of March 2026

## Setup

```
pip install -r requirements.txt
```

R packages needed: `tidyverse`, `scales`, `sandwich`, `lmtest`, `broom`, `stargazer`, `car`
