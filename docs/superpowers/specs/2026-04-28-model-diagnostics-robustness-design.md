# Design Spec: Notebooks 12-13 (Model Diagnostics & Robustness Checks)

Date: 2026-04-28

## Context

Expert feedback from Mohan Sandrasegeran (PropertyGuru), Prof Sing Tien Foo (NUS), Nicholas Mak (ERA), and Mark Hensen identified diagnostic and robustness gaps in the HDB resale price regression. The core model (raw-price Model 10, R-squared 0.9023) and log model (R-squared 0.9373) are built in Notebooks 6 and 11. This spec covers two new notebooks to validate them.

## Notebook 12: Model Diagnostics

**Purpose:** "Is the model well-behaved?" — validate Model 10 against standard regression assumptions and check model size.

**Runs on:** Raw-price Model 10 (the reporting model), with log model comparisons where relevant.

### Section 1: Residual normality (Q-Q plots)

- Two side-by-side Q-Q plots: raw Model 10 residuals vs log Model 10 residuals
- Histograms of residuals for both models (echoing the price histogram Hensen reviewed)
- Interpretation: severity of non-normality, whether it matters with n=50K and robust SEs, and how the log transform improves tail behavior
- This directly addresses Hensen's point about "heavy tails" and the log transform fixing them

### Section 2: Influential points (Cook's distance)

- Compute Cook's distance for all 50,718 transactions in Model 10
- Show top 20 most influential transactions — compare against the outlier lists from NB6/NB11 (are influential points the same as large residuals, or different?)
- Cook's distance plot (vs fitted values)
- Flag any transaction with Cook's D > 4/n threshold
- Key test: rerun Model 10 **without** the top influential points and compare coefficients. Do superstition variables change? Does the lucky-8 coefficient move? This is Hensen's concern about "one super expensive place with eights" driving the result
- Also check: is the 367-sqm Jalan Ma'mor terrace house influential?

### Section 3: Multicollinearity (VIF)

- `car::vif()` on Model 10
- Table of VIF values for all variables, flagging >5 (moderate) and >10 (serious)
- Expected findings:
  - `remaining_lease_years` / `remaining_lease_sq`: very high VIF (by design — quadratic term, r=0.996)
  - Town dummies vs `dist_cbd_km`: Mohan's concern about double-counting location
- Interpretation: high VIF inflates standard errors on individual coefficients but doesn't bias them or affect overall model fit. For the superstition variables (which are the story), VIF should be low because they're not correlated with location.

### Section 4: AIC/BIC model selection

- Compute AIC and BIC for the Model 1-10 progression from NB6
- Chart: x-axis = number of parameters, y-axis = AIC and BIC (dual lines). Mark where each model sits. Show where the penalty curve bottoms out.
- This is Hensen's "one curve comes down, one curve goes up, find the minimum" visualization
- Then run `step()` (both directions) starting from Model 10's variable set, using AIC as criterion
- Report: does stepwise agree with Model 10? What does it add or drop?
- If `leaps` package is available, run best-subset for models of each size (Hensen's "leaps and bounds" suggestion). Otherwise, stepwise is sufficient.

### Section 5: Coefficient stability

- Table showing how key coefficients move across model specifications (Models 5 through 10, plus log model from NB11)
- Variables tracked: `num_eights_tail`, `price_has_168`, `block_has_4`, `cny_month`, `dist_cbd_km`, `mrt_dist_m`, `columbarium_dist_m`
- Show point estimates with 95% confidence intervals (robust SEs) across specifications
- Coefficient plot (forest plot style): one row per variable, one column per model specification, with CI bars
- Addresses Mohan's concern about "unstable coefficients across model specifications" and "unexpected signs on key variables"

### Interpretation section

Summarize: which diagnostics pass cleanly, which flag concerns, and what the concerns mean for the published findings.

---

## Notebook 13: Robustness Checks

**Purpose:** "Do the results hold up under different methods?" — three stress tests on the key findings.

### Section 1: Hold-out validation

- Random 2/3-1/3 split, `set.seed()` for reproducibility
- Train raw Model 10 on training set (n ~ 33,800), predict on test set (n ~ 16,900)
- Report:
  - In-sample vs out-of-sample R-squared
  - MAE and median absolute error on both sets
  - Scatter plot: predicted vs actual for test set, with 45-degree reference line
  - Residual distribution comparison (training vs test)
- Repeat for log model
- Key question: does R-squared drop substantially out of sample? If yes, fixed effects are overfitting (Mohan's concern). If no, the model generalises.
- Also report: out-of-sample performance by price quartile (does the model predict expensive flats as well as cheap ones?)

### Section 2: L1 / LAD regression

- `quantreg::rq(tau = 0.5)` with the same Model 10 formula (median regression)
- Side-by-side coefficient table: OLS vs LAD for all key variables
- Focus comparison on superstition variables:
  - `num_eights_tail`: OLS = +$1,070. LAD = ?
  - `block_has_4`: OLS = -$10,160. LAD = ?
  - `price_has_168`: OLS = +$32,795. LAD = ?
  - `cny_month`: OLS = +$59,310. LAD = ?
- Also compare distance variables (CBD, MRT, columbarium, temple)
- Interpretation tied to Hensen: LAD uses median not mean, so outliers can't pull the coefficient. If results are similar, the findings are robust. If they differ, flag which coefficients are outlier-sensitive.
- Optional: run LAD on log model too, for completeness

### Section 3: Lucky-8 interaction with price level

- Create price quartiles based on Model 10's **predicted** price (not actual — avoids endogeneity since trailing 8s are part of the actual price)
- Interaction model: `num_eights_tail × price_quartile`
- Show the 8-premium in each quartile:
  - Q1 (cheapest ~25%): 8-premium = ?
  - Q2: ?
  - Q3: ?
  - Q4 (most expensive ~25%): ?
- Is the premium constant in dollar terms, proportional to price, or concentrated in one segment?
- Also check: what variables cluster with trailing 8s? (correlation table of `num_eights_tail` with other predictors)
- Addresses Hensen's question: "Is it that these shenanigans work for lower price more often than upper price?"

### Section 4: Summary table

A single publishable table consolidating all robustness findings:

| Check | Result | Implication |
|---|---|---|
| Out-of-sample R-squared (raw) | ? | Generalises / overfits |
| Out-of-sample R-squared (log) | ? | Generalises / overfits |
| LAD vs OLS: num_eights_tail | $? vs $1,070 | Outlier-robust / not |
| LAD vs OLS: block_has_4 | $? vs -$10,160 | Stable / not |
| LAD vs OLS: price_has_168 | $? vs $32,795 | Stable / not |
| 8-premium by price quartile | Uniform / varies | Effect is real everywhere / concentrated |

---

## Technical notes

- **Language:** R via rpy2 (consistent with NB6-11)
- **Packages needed:** `car` (VIF), `quantreg` (LAD), `sandwich`/`lmtest` (robust SEs), `tidyverse`, `ggplot2`. All installed. `leaps` not installed — will attempt `install.packages('leaps')` in NB12 if needed, otherwise stepwise via `step()` is sufficient.
- **Style:** Editorial annotations on methodological choices (consistent with the clean-notebook skill). Each section has a markdown interpretation block aimed at a reader who knows what regression is but wouldn't know what Cook's distance is without explanation.
- **Model formula:** Identical to Model 10 from NB6 (raw price) and the log model from NB11. No new variables introduced.
