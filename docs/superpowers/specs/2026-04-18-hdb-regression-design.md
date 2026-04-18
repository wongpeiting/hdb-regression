# What Predicts HDB Resale Prices?

**Date:** 2026-04-18
**Status:** Approved
**Author:** Wong Pei Ting

## Question

What drives HDB resale prices? And which transactions are outliers that can't be explained by location, size, age, floor, or flat model — feeding into the WTF flat tracker with a statistically rigorous methodology?

## Why this matters

The existing WTF flat tracker at `/Users/wongpeiting/Desktop/CU/python-work/wtf_flat_alert_system/` uses a crude methodology: group medians and z-scores within town+flat_type bins. This doesn't control for multiple factors simultaneously — a high-floor flat in a group that doesn't account for floor level will score as an outlier when it isn't one. A proper multivariable regression controls for all factors at once, so residuals represent genuinely unexplained price deviations.

## Data Sources

### Phase 1 (this project): data.gov.sg HDB resale transactions
- Dataset: `d_8b84c4ee58e3cfc0ece0d773c8ca6abc` (HDB Resale Flat Prices)
- API: `https://data.gov.sg/api/action/datastore_search`
- Scope: Last 2 years of transactions (~50K+ rows)
- Fields: month, town, flat_type, block, street_name, storey_range, floor_area_sqm, flat_model, lease_commence_date, remaining_lease, resale_price

### Phase 2 (future): PropertyGuru enrichment
- Join PropertyGuru scraped listings (`/Users/wongpeiting/propertyguru/data/propertyguru_hdb_listings.csv`) by block + street to add:
  - MRT proximity (nearest_mrt, distance)
  - Furnished status
  - Green score (BCA Green Mark)
  - Agent data
- Test whether these variables improve R² beyond Phase 1 model

## Y Variable

`log_resale_price` — log10 of transaction price.

**Why log?** Price distributions are right-skewed. Log transformation makes the distribution approximately normal and lets regression coefficients be interpreted as percentage changes. `floor_area_sqm` is included as an X variable so its coefficient captures the size elasticity — same reasoning as `log_median_bid` in the GeBIZ analysis.

## X Variables (Phase 1)

| Variable | Type | Derived from | Hypothesis |
|---|---|---|---|
| `town` | Categorical (26 towns) | `town` | Location is the biggest driver — Bishan ≠ Woodlands |
| `flat_type` | Categorical | `flat_type` | 3-room vs 4-room vs 5-room vs Executive |
| `floor_area_sqm` | Continuous | `floor_area_sqm` | Bigger = more expensive |
| `storey_mid` | Continuous | Midpoint of `storey_range` (e.g., "07 TO 09" → 8) | Higher floor = premium |
| `remaining_lease_years` | Continuous | 99 - (transaction_year - `lease_commence_date`) | Lease decay — at what point does price drop sharply? |
| `flat_model` | Categorical | `flat_model` | DBSS, Premium Apartment, Maisonette command premiums |
| `ends_in_8` | Binary | Last digit of `resale_price` == 8 | Singapore lucky number superstition — do "auspicious" prices sell higher? |

## Model-Building Progression

Following the GeBIZ 4-model step-up approach with ANOVA between each:

| Model | Variables | What it tests |
|---|---|---|
| **Model 1** | `town` + `flat_type` | Location + type = baseline. How much does this alone explain? |
| **Model 2** | + `floor_area_sqm` + `storey_mid` | Physical attributes — does size and floor premium add to location? |
| **Model 3** | + `remaining_lease_years` | Lease decay — does remaining lease matter after controlling for location, size, floor? |
| **Model 4** | + `flat_model` | Do DBSS/Premium/Maisonette command premiums beyond their size and location? |
| **Model 5** | + `ends_in_8` | The lucky 8 test — does pricing ending in 8 predict higher prices after everything else? |

ANOVA at each step to test whether added variables significantly improve the model. Stargazer side-by-side comparison table.

## Residual Analysis → WTF Flats

The key output: residuals from the full model replace the current crude z-score methodology in the WTF flat tracker.

- **Large positive residuals** = sold for much more than the model predicts → WTF flats (overpriced)
- **Large negative residuals** = sold for much less than predicted → bargain alerts (underpriced)
- Residuals are standardised and ranked to produce a WTF score

This is statistically grounded: a flat only scores as "WTF" if its price can't be explained by town, flat type, size, floor, lease, model, or lucky-8 pricing. The current z-score approach flags flats that are merely above their group median — which could just mean they're high-floor or large.

## 6-Notebook Structure

Following Dhrumil Mehta's EDA-with-regression pipeline:

### Notebook 1: Download and prepare data
- Pull HDB resale data from data.gov.sg API (same API the WTF tracker uses)
- Filter to last 2 years of transactions
- Derive variables: `storey_mid`, `remaining_lease_years`, `log_resale_price`, `ends_in_8`
- Clean and document data quality issues
- Save cleaned CSV

### Notebook 2: Explore distributions
- R via rpy2, ggplot2
- Histogram of resale_price (raw and log)
- Boxplots: price by town, flat_type, flat_model
- Scatter: floor area vs price, storey vs price, remaining lease vs price
- Distribution of ends_in_8: how common are "lucky" prices?
- Summary statistics tables

### Notebook 3: Download supplementary data
- Placeholder for Phase 2 (PropertyGuru join)
- For now: any additional derived variables or external data that's quick to add

### Notebook 4: Merge
- Build master regression table
- Null check — confirm data is clean
- Save `data/hdb_analysis.csv`

### Notebook 5: Single-variable regressions
- R: `lm(log_resale_price ~ X)` for each variable individually
- Table: coefficient, p-value, R² for each
- ggplot chart for each significant predictor
- Identify which variables pass p < 0.2 screen

### Notebook 6: Multivariable regression
- Build up Models 1–5 with ANOVA between each
- Stargazer side-by-side comparison table
- R² progression table
- Residual analysis: top 20 most overpriced transactions (WTF flats)
- Residual analysis: top 20 most underpriced transactions (bargains)
- Bonus: compare residual-based WTF list to the current z-score-based list — how much overlap?
- Final interpretation and story leads

## Technical Setup

- Jupyter notebooks with Python + R via rpy2
- R packages: tidyverse, broom, scales, stargazer
- Python: pandas, numpy, matplotlib, requests
- Editorial annotations on methodological choices (same style as GeBIZ/Spotify notebooks)

## What This Answers

1. **What drives HDB prices?** R² progression shows which factors matter most — is it location, size, floor, lease, or model?
2. **How much is a high floor worth?** Coefficient on `storey_mid` after controlling for everything else
3. **At what remaining lease does price drop sharply?** Non-linear lease decay effect
4. **Do DBSS/Premium flats command genuine premiums?** After controlling for size and location
5. **Does the lucky 8 work?** After controlling for everything, do 8-ending prices transact higher?
6. **Which flats are genuinely overpriced?** Residual-based WTF flat list, replacing the crude z-score method

## Limitations

- **Asking price ≠ transaction price** — data.gov.sg has actual sales, but we don't know the original asking price or how long the flat was listed. Phase 2 (PropertyGuru) can add listing duration.
- **No MRT distance in Phase 1** — one of the strongest predictors, but not in data.gov.sg. Added in Phase 2 via PropertyGuru join.
- **`flat_model` has many levels** — some models (e.g., "Type S2") have very few transactions. May need to group rare models.
- **2-year window** — captures current market but misses long-term trends. Could extend to 5 years with a year fixed effect.
- **`ends_in_8` is a fun variable but has a methodological subtlety**: sellers SET the price, so it reflects seller psychology, not buyer willingness to pay more. A significant coefficient means "sellers who price at 8 endings tend to get higher prices" — which could mean the 8 works, or that savvy sellers (who use psychological pricing) are also better negotiators.

## Reference Notebooks

- GeBIZ multivariable regression: `/Users/wongpeiting/final-project/eda-with-regression-wongpeiting/6-multivariable-regression.ipynb`
- GeBIZ data download: `/Users/wongpeiting/final-project/eda-with-regression-wongpeiting/1-download-your-data.ipynb`
- Existing WTF flat tracker: `/Users/wongpeiting/Desktop/CU/python-work/wtf_flat_alert_system/auto_updating_site.ipynb`
