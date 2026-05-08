# HDB Resale Price Regression

This project analyses what drives HDB resale prices in Singapore, using a regression model (R²=0.9403) trained on 52,000 transactions from May 2024 to April 2026 and a historical analysis of almost a million transactions spanning 35 years (1990-2026).

The exploratory data analysis phase led me down several interesting paths, including the effect of policy announcements on how fast lease decay kicks into homebuyer consciousness in Singapore, but the first story that I wanted to tell was one I feel would have been difficult to show convincingly without the backing of regression analysis: the effect of number-linked superstitions on resale prices. 

I came away finding that both trailing 8s in sale prices and the digit 4 in block numbers have statistically significant effects on HDB resale values, with these effects strengthening over time, even after controlling for location, flat type, size, floor, remaining lease, proximity to amenities, month of sale, and structural differences in terrace houses and lease depreciation.

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

## Methodology

The model uses a log-price specification with 88 variables, controlling for town, flat type, floor area, storey range, remaining lease (with quadratic and cubic terms to capture accelerating depreciation), flat model, proximity to MRT stations, hawker centres, popular schools, hospitals, parks, temples, columbaria and the central business district, month of sale, and a floor area adjustment for terrace houses. The typical prediction is off by about $28,000, or roughly 4% of a median flat's price.

**Overfitting check:** The model was trained on two-thirds of the data and tested on the remaining third. R² dropped from 0.9407 to 0.9394 — a 0.1% decline. The results were also tested with a raw-price specification; both produced consistent findings.

**Robustness:** The trailing-8 premium and block-4 penalty are both significant at the 99.9% confidence level in every specification tested. The trailing-8 premium strengthened slightly from about $1,600 to $1,700 when the most influential transactions were removed (Cook's D). The block-4 penalty fell from about $8,500 to $6,700 but remained significant. The block-4 historical figures ($600 in 1990–91, $13,200 in 2024–25) come from running the same model on each two-year period separately.

**Dollar equivalents:** The log model produces percentage coefficients. This project converts them to dollar equivalents at the median flat price ($620,000) for readability: a premium of $1,600 per trailing 8 is more intuitive than 0.25%.

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
