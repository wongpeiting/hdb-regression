# Notebooks 12-13: Model Diagnostics & Robustness Checks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate HDB resale price Model 10 against standard regression diagnostics and stress-test key findings with alternative methods.

**Architecture:** Two Jupyter notebooks (R via rpy2), building on the existing Model 10 (NB6) and log model (NB11). NB12 checks model assumptions. NB13 checks whether findings survive different data splits and estimation methods. Both notebooks load `data/hdb_analysis.csv` and reconstruct Model 10 from scratch (self-contained).

**Tech Stack:** R (tidyverse, sandwich, lmtest, car, quantreg, ggplot2), Python (rpy2, jupyter)

**Spec:** `docs/superpowers/specs/2026-04-28-model-diagnostics-robustness-design.md`

---

## Task 1: Create Notebook 12 — Setup and Model Reconstruction

**Files:**
- Create: `12-model-diagnostics.ipynb`

This task creates the notebook file and writes the opening markdown + setup cells. Every subsequent task in NB12 appends cells to this notebook.

- [ ] **Step 1: Create notebook with title and setup cells**

Cell 0 (markdown):
```markdown
# HDB Resale Price Regression — Notebook 12: Model Diagnostics

Experts flagged several diagnostic checks for Model 10 (R-squared 0.90, 50,718 transactions). This notebook validates the model against standard regression assumptions:

1. **Residual normality** — Q-Q plots and histograms (are the residuals well-behaved?)
2. **Influential points** — Cook's distance (is any single transaction pulling the fit?)
3. **Multicollinearity** — VIF (are town dummies and distance variables double-counting?)
4. **Model selection** — AIC/BIC (is Model 10 the right size, or is it over/under-parameterised?)
5. **Coefficient stability** — do the key findings survive across specifications?

Raw-price Model 10 is the primary subject (the reporting model). Log model comparisons are shown where relevant.
```

Cell 1 (code):
```python
%load_ext rpy2.ipython
import warnings
warnings.filterwarnings('ignore')
```

Cell 2 (code — reconstruct Model 10 and log model):
```r
%%R
library(tidyverse)
library(sandwich)
library(lmtest)
library(car)
library(ggplot2)

df <- read_csv('data/hdb_analysis.csv', show_col_types = FALSE)
df$remaining_lease_sq <- df$remaining_lease_years^2
df$month_factor <- factor(format(df$month, '%Y-%m'))
df$ln_price <- log(df$resale_price)

# Raw-price Model 10 (from Notebook 6)
model10 <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
              remaining_lease_years + remaining_lease_sq +
              flat_model_grouped +
              dist_cbd_km + mrt_dist_m + hawker_dist_m +
              popular_school_dist_m +
              park_dist_m + hospital_dist_m +
              columbarium_dist_m + temple_dist_m +
              coast_dist_m +
              num_eights_tail +
              price_has_168 +
              block_has_4 +
              cny_month +
              month_factor,
            data = df)

# Log model (from Notebook 11)
model_log <- lm(ln_price ~ town + flat_type + floor_area_sqm + storey_mid +
              remaining_lease_years + remaining_lease_sq +
              flat_model_grouped +
              dist_cbd_km + mrt_dist_m + hawker_dist_m +
              popular_school_dist_m +
              park_dist_m + hospital_dist_m +
              columbarium_dist_m + temple_dist_m +
              coast_dist_m +
              num_eights_tail +
              price_has_168 +
              block_has_4 +
              cny_month +
              month_factor,
            data = df)

cat(sprintf('Raw Model 10:  R² = %.4f, %d parameters\n',
    summary(model10)$r.squared, length(coef(model10))))
cat(sprintf('Log Model 10:  R² = %.4f, %d parameters\n',
    summary(model_log)$r.squared, length(coef(model_log))))
cat(sprintf('Observations:  %s\n', format(nrow(df), big.mark = ',')))
```

- [ ] **Step 2: Run the notebook to verify setup loads correctly**

Run: `jupyter nbconvert --to notebook --execute 12-model-diagnostics.ipynb --output 12-model-diagnostics.ipynb`

Expected: Both models reconstruct with R² = 0.9023 (raw) and 0.9373 (log).

- [ ] **Step 3: Commit**

```bash
git add 12-model-diagnostics.ipynb
git commit -m "Add notebook 12 setup: model diagnostics scaffold"
```

---

## Task 2: NB12 Section 1 — Residual Normality (Q-Q Plots)

**Files:**
- Modify: `12-model-diagnostics.ipynb`

Append cells for Q-Q plots and residual histograms. This addresses Mark Hensen's point about heavy-tailed residuals and the log transform fixing them.

- [ ] **Step 1: Add markdown header**

Cell (markdown):
```markdown
## 1. Residual normality

OLS assumes residuals are normally distributed. With n = 50,718 and robust standard errors, mild non-normality won't bias the coefficients — but heavy tails mean outliers are pulling the fit harder than they should. The log transform should improve this.

**How to read Q-Q plots:** Points follow the diagonal line = normal. Points curving away at the ends = heavy tails (outliers more extreme than a normal distribution would produce).
```

- [ ] **Step 2: Add Q-Q plots cell (side by side, raw vs log)**

Cell (code):
```r
%%R -w 900 -h 450
# Helper functions (moments package not installed)
skew <- function(x) { m <- mean(x); s <- sd(x); mean(((x - m) / s)^3) }
kurt <- function(x) { m <- mean(x); s <- sd(x); mean(((x - m) / s)^4) }

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# Raw model Q-Q
raw_resid <- rstandard(model10)
qqnorm(raw_resid, main = 'Raw Model 10 residuals',
       pch = '.', col = 'grey40', cex = 0.8)
qqline(raw_resid, col = 'red', lwd = 2)
mtext(sprintf('Skewness: %.2f  |  Kurtosis: %.2f',
    skew(resid(model10)), kurt(resid(model10))),
    side = 3, line = 0, cex = 0.8)

# Log model Q-Q
log_resid <- rstandard(model_log)
qqnorm(log_resid, main = 'Log Model 10 residuals',
       pch = '.', col = 'grey40', cex = 0.8)
qqline(log_resid, col = 'red', lwd = 2)
mtext(sprintf('Skewness: %.2f  |  Kurtosis: %.2f',
    skew(resid(model_log)), kurt(resid(model_log))),
    side = 3, line = 0, cex = 0.8)
```

- [ ] **Step 3: Add residual histogram cell (side by side)**

Cell (code):
```r
%%R -w 900 -h 400
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# Raw model residuals
hist(resid(model10), breaks = 100, col = 'grey80', border = 'white',
     main = 'Raw Model 10 residuals', xlab = 'Residual ($)',
     xlim = c(-500000, 500000))
abline(v = 0, col = 'red', lwd = 2)
abline(v = median(resid(model10)), col = 'blue', lwd = 2, lty = 2)
legend('topright', c('Zero', sprintf('Median ($%s)',
    format(round(median(resid(model10))), big.mark = ','))),
    col = c('red', 'blue'), lwd = 2, lty = c(1, 2), cex = 0.8)

# Log model residuals
hist(resid(model_log), breaks = 100, col = 'grey80', border = 'white',
     main = 'Log Model 10 residuals', xlab = 'Residual (log $)')
abline(v = 0, col = 'red', lwd = 2)
abline(v = median(resid(model_log)), col = 'blue', lwd = 2, lty = 2)
legend('topright', c('Zero', sprintf('Median (%.4f)',
    median(resid(model_log)))),
    col = c('red', 'blue'), lwd = 2, lty = c(1, 2), cex = 0.8)
```

- [ ] **Step 4: Add interpretation markdown**

Cell (markdown):
```markdown
### Interpretation

**Raw model:** [To be filled after running — describe the Q-Q shape. Expected: heavy right tail from expensive flats, left tail from terrace/3Gen outliers. Median residual is negative, meaning the model slightly overpredicts typical flats and underpredicts expensive ones.]

**Log model:** [Expected: much closer to the diagonal. The log transform compresses the right tail. Kurtosis should be lower.]

**What this means for inference:** With 50,718 observations and HC1 robust standard errors, non-normality in the residuals does not bias the coefficients or invalidate the p-values. The robust SEs already account for heteroskedasticity. However, the heavy tails in the raw model mean that the OLS estimates are being pulled toward outliers — a concern we address with L1 (median) regression in Notebook 13.
```

- [ ] **Step 5: Run notebook and verify Q-Q plots render**

Run: `jupyter nbconvert --to notebook --execute 12-model-diagnostics.ipynb --output 12-model-diagnostics.ipynb`

- [ ] **Step 6: Commit**

```bash
git add 12-model-diagnostics.ipynb
git commit -m "NB12: add Q-Q plots and residual histograms"
```

---

## Task 3: NB12 Section 2 — Influential Points (Cook's Distance)

**Files:**
- Modify: `12-model-diagnostics.ipynb`

- [ ] **Step 1: Add markdown header**

Cell (markdown):
```markdown
## 2. Influential points (Cook's distance)

A residual tells you how far off the prediction was. Cook's distance tells you how much the **entire model** would change if you removed that one transaction. A flat with a large residual isn't necessarily influential — it might be an outlier that the model shrugs off. A flat with high Cook's distance is actively pulling the regression line toward it.

**Threshold:** Cook's D > 4/n (= 4/50,718 ≈ 0.00008) is a common rule of thumb. Points above this are worth inspecting.

**Key question:** Is the lucky-8 coefficient being driven by a handful of expensive eights-laden transactions? If we remove the most influential points, does the superstition story survive?
```

- [ ] **Step 2: Add Cook's distance computation and top-20 table**

Cell (code):
```r
%%R
# Cook's distance for every observation
cooks_d <- cooks.distance(model10)
threshold <- 4 / nrow(df)

cat(sprintf('Cook\'s distance threshold (4/n): %.6f\n', threshold))
cat(sprintf('Observations above threshold: %d (%.1f%%)\n',
    sum(cooks_d > threshold), sum(cooks_d > threshold) / nrow(df) * 100))
cat(sprintf('Max Cook\'s D: %.4f (%.0fx threshold)\n\n',
    max(cooks_d), max(cooks_d) / threshold))

# Top 20 most influential transactions
df$cooks_d <- cooks_d
top20 <- df[order(-cooks_d), ][1:20, ]

cat(sprintf('%-5s %-30s %-10s %10s %10s %12s\n',
    'Rank', 'Address', 'Type', 'Price', 'Residual', 'Cook\'s D'))
cat(paste(rep('-', 82), collapse = ''), '\n')

for (i in 1:20) {
    r <- top20[i, ]
    label <- sprintf('Blk %s %s', r$block, substr(r$street_name, 1, 20))
    cat(sprintf('%-5d %-30s %-10s $%9s $%+9s %11.5f\n',
        i, label, r$flat_type,
        format(round(r$resale_price), big.mark = ','),
        format(round(r$resale_price - predict(model10, r)), big.mark = ','),
        r$cooks_d))
}
```

- [ ] **Step 3: Add Cook's distance plot**

Cell (code):
```r
%%R -w 900 -h 400
fitted_vals <- fitted(model10)

plot(fitted_vals, cooks_d,
     pch = '.', col = ifelse(cooks_d > threshold, 'red', 'grey60'),
     main = "Cook's distance vs fitted values (Model 10)",
     xlab = 'Fitted value ($)', ylab = "Cook's distance",
     cex = ifelse(cooks_d > threshold, 1.5, 0.5))
abline(h = threshold, col = 'blue', lty = 2, lwd = 1.5)
text(x = fitted_vals[which.max(cooks_d)], y = max(cooks_d),
     labels = sprintf('Blk %s %s', df$block[which.max(cooks_d)],
     substr(df$street_name[which.max(cooks_d)], 1, 15)),
     pos = 2, cex = 0.7, col = 'red')
legend('topright', c(sprintf('Above threshold (%d pts)', sum(cooks_d > threshold)),
    'Below threshold'),
    col = c('red', 'grey60'), pch = 16, cex = 0.8)
```

- [ ] **Step 4: Add sensitivity test — rerun without influential points**

Cell (code):
```r
%%R
# Remove observations above Cook's D threshold and refit
df_clean <- df[cooks_d <= threshold, ]
cat(sprintf('Removed %d influential observations (%.1f%% of data)\n',
    nrow(df) - nrow(df_clean),
    (nrow(df) - nrow(df_clean)) / nrow(df) * 100))

model10_clean <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
              remaining_lease_years + remaining_lease_sq +
              flat_model_grouped +
              dist_cbd_km + mrt_dist_m + hawker_dist_m +
              popular_school_dist_m +
              park_dist_m + hospital_dist_m +
              columbarium_dist_m + temple_dist_m +
              coast_dist_m +
              num_eights_tail +
              price_has_168 +
              block_has_4 +
              cny_month +
              month_factor,
            data = df_clean)

# Compare key coefficients
key_vars <- c('floor_area_sqm', 'storey_mid', 'remaining_lease_years',
              'dist_cbd_km', 'mrt_dist_m', 'columbarium_dist_m', 'temple_dist_m',
              'num_eights_tail', 'price_has_168', 'block_has_4', 'cny_month')

robust_full <- coeftest(model10, vcov = vcovHC(model10, type = 'HC1'))
robust_clean <- coeftest(model10_clean, vcov = vcovHC(model10_clean, type = 'HC1'))

cat(sprintf('\n%-25s %12s %12s %10s\n', 'Variable', 'Full model', 'No outliers', 'Change'))
cat(paste(rep('-', 62), collapse = ''), '\n')

for (v in key_vars) {
    c_full <- coef(model10)[v]
    c_clean <- coef(model10_clean)[v]
    pct_change <- (c_clean - c_full) / abs(c_full) * 100
    cat(sprintf('%-25s $%+10.0f $%+10.0f %+9.1f%%\n', v, c_full, c_clean, pct_change))
}

cat(sprintf('\n%-25s %12.4f %12.4f\n', 'R-squared',
    summary(model10)$r.squared, summary(model10_clean)$r.squared))
```

- [ ] **Step 5: Add interpretation markdown**

Cell (markdown):
```markdown
### Interpretation

**Most influential transactions:** [To be filled — expected: Jalan Ma'mor terrace houses, 3Gen bargain flats, and a few extreme alamak flats. These are the same outliers from NB6/NB11, confirming that large residuals = large influence here.]

**Sensitivity test:** [Expected: coefficients barely move. The superstition variables should be stable because they operate across many transactions, not driven by a few. If num_eights_tail changes by more than ~20%, that's a concern.]

**Key finding:** [Expected: "Removing the X most influential transactions changes the lucky-8 coefficient from $1,070 to $Y — a Z% change. The model's conclusions are / are not sensitive to individual transactions."]
```

- [ ] **Step 6: Run notebook and commit**

```bash
jupyter nbconvert --to notebook --execute 12-model-diagnostics.ipynb --output 12-model-diagnostics.ipynb
git add 12-model-diagnostics.ipynb
git commit -m "NB12: add Cook's distance analysis and sensitivity test"
```

---

## Task 4: NB12 Section 3 — Multicollinearity (VIF)

**Files:**
- Modify: `12-model-diagnostics.ipynb`

- [ ] **Step 1: Add markdown header**

Cell (markdown):
```markdown
## 3. Multicollinearity (VIF)

Variance Inflation Factor measures how much each coefficient's standard error is inflated by correlation with other predictors. VIF = 1 means no correlation. VIF > 5 is moderate concern. VIF > 10 is serious.

**Mohan's concern:** Town fixed effects AND continuous distance variables (dist_cbd_km, mrt_dist_m, etc.) might be double-counting location. If they're highly correlated, individual coefficients become unreliable — though the overall model fit and predictions are unaffected.

**Note:** VIF for factor variables (town, flat_type, flat_model) uses Generalized VIF (GVIF), adjusted for degrees of freedom. The comparable threshold is GVIF^(1/(2*df)) > √5 ≈ 2.24.
```

- [ ] **Step 2: Add VIF computation cell**

Cell (code):
```r
%%R
# VIF for Model 10
# car::vif handles factors via GVIF
vif_results <- vif(model10)

# For factors, vif() returns a matrix with GVIF, Df, GVIF^(1/(2*Df))
# For continuous, it returns a named vector
cat('=== VIF Results ===\n\n')

if (is.matrix(vif_results)) {
    # All results in matrix form (when there are factors)
    cat(sprintf('%-30s %8s %4s %12s %8s\n',
        'Variable', 'GVIF', 'Df', 'GVIF^(1/2Df)', 'Flag'))
    cat(paste(rep('-', 66), collapse = ''), '\n')

    for (i in 1:nrow(vif_results)) {
        gvif <- vif_results[i, 'GVIF']
        df_val <- vif_results[i, 'Df']
        adj_vif <- vif_results[i, 'GVIF^(1/(2*Df))']
        flag <- ifelse(adj_vif > sqrt(10), '*** HIGH',
                ifelse(adj_vif > sqrt(5), '** MOD', ''))
        cat(sprintf('%-30s %8.2f %4.0f %12.2f %8s\n',
            rownames(vif_results)[i], gvif, df_val, adj_vif, flag))
    }
} else {
    # Simple numeric vector (no factors)
    for (v in names(vif_results)) {
        flag <- ifelse(vif_results[v] > 10, '*** HIGH',
                ifelse(vif_results[v] > 5, '** MOD', ''))
        cat(sprintf('%-30s %8.2f %8s\n', v, vif_results[v], flag))
    }
}
```

- [ ] **Step 3: Add interpretation markdown**

Cell (markdown):
```markdown
### Interpretation

**Expected high VIF:**
- `remaining_lease_years` / `remaining_lease_sq`: These are a designed quadratic pair (r = 0.996). High VIF is expected and harmless — the pair works together; we don't interpret them individually.

**Mohan's concern — town vs distance:**
- [To be filled — expected: town has moderate GVIF because it's correlated with dist_cbd_km and other distances. dist_cbd_km may show VIF ~ 3-6. This means the individual coefficient on dist_cbd_km (-$16,116/km) is less precise than it looks, but the overall model fit is fine.]

**Superstition variables:**
- [Expected: low VIF for num_eights_tail, price_has_168, block_has_4. These are constructed from price digits and block numbers, which are essentially random with respect to location and physical characteristics. If VIF < 2 for all superstition variables, multicollinearity is not a concern for the superstition findings.]

**Bottom line:** [Expected: "Multicollinearity is present between location variables (as Mohan noted), but it does not affect the superstition findings. The town dummies and distance variables absorb overlapping location information, which inflates their individual SEs but doesn't bias the model. For the published story, interpret location variables with caution — the $16,116/km CBD coefficient is a rough guide, not a precise estimate — but the superstition coefficients are clean."]
```

- [ ] **Step 4: Run notebook and commit**

```bash
jupyter nbconvert --to notebook --execute 12-model-diagnostics.ipynb --output 12-model-diagnostics.ipynb
git add 12-model-diagnostics.ipynb
git commit -m "NB12: add VIF multicollinearity analysis"
```

---

## Task 5: NB12 Section 4 — AIC/BIC Model Selection

**Files:**
- Modify: `12-model-diagnostics.ipynb`

- [ ] **Step 1: Add markdown header**

Cell (markdown):
```markdown
## 4. AIC/BIC model selection

Mark Hensen: "As you add more variables, the RSS goes down, but the counterweight — the number of variables — goes up. You want to find the place where it's minimum."

AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) both penalise model complexity. BIC penalises harder — it prefers simpler models. If both agree that Model 10 is near the minimum, the model is the right size. If AIC says "keep adding" but BIC says "stop earlier," there's a tension between fit and parsimony.

**Two tests:**
1. Score the existing Models 1-10 progression from Notebook 6
2. Run automated stepwise selection (forward + backward) to see if the algorithm agrees with our manual choices
```

- [ ] **Step 2: Add AIC/BIC across Models 1-10**

Cell (code):
```r
%%R
# Reconstruct Models 1-9 to compute AIC/BIC
# (Model 10 and model_log already exist)

m1 <- lm(resale_price ~ town + flat_type, data = df)
m2 <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid, data = df)
m3 <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
         remaining_lease_years, data = df)
m4 <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
         remaining_lease_years + flat_model_grouped, data = df)
m5 <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
         remaining_lease_years + flat_model_grouped + ends_in_8, data = df)
m6 <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
         remaining_lease_years + flat_model_grouped + ends_in_8 +
         dist_cbd_km + mrt_dist_m + school_dist_m + hawker_dist_m +
         supermarket_dist_m + park_dist_m + hospital_dist_m, data = df)
m7 <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
         remaining_lease_years + flat_model_grouped + ends_in_8 +
         dist_cbd_km + mrt_dist_m + school_dist_m + hawker_dist_m +
         supermarket_dist_m + park_dist_m + hospital_dist_m +
         columbarium_dist_m + funeral_dist_m + temple_dist_m +
         reservoir_dist_m + coast_dist_m, data = df)
m8 <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
         remaining_lease_years + flat_model_grouped + ends_in_8 +
         dist_cbd_km + mrt_dist_m + school_dist_m + hawker_dist_m +
         supermarket_dist_m + park_dist_m + hospital_dist_m +
         columbarium_dist_m + funeral_dist_m + temple_dist_m +
         reservoir_dist_m + coast_dist_m +
         num_eights_in_price + num_fours_in_price +
         price_has_888 + price_has_168 +
         has_floor_4 + has_floor_13 + has_floor_14 +
         block_has_4 + block_has_8 +
         hungry_ghost + cny_month, data = df)
m9 <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
         remaining_lease_years + remaining_lease_sq +
         flat_model_grouped + ends_in_8 +
         dist_cbd_km + mrt_dist_m + school_dist_m + hawker_dist_m +
         supermarket_dist_m + park_dist_m + hospital_dist_m +
         columbarium_dist_m + funeral_dist_m + temple_dist_m +
         reservoir_dist_m + coast_dist_m +
         num_eights_in_price + num_fours_in_price +
         price_has_888 + price_has_168 +
         has_floor_4 + has_floor_13 + has_floor_14 +
         block_has_4 + block_has_8 +
         hungry_ghost + cny_month +
         month_factor + floor_area_sqm:flat_type, data = df)

models <- list(m1, m2, m3, m4, m5, m6, m7, m8, m9, model10)
labels <- c('M1: Town+Type', 'M2: +Size/Floor', 'M3: +Lease',
            'M4: +FlatModel', 'M5: +Lucky8', 'M6: +Geography',
            'M7: +FengShui', 'M8: +Superstition', 'M9: +Lease²+MonthFE',
            'M10: Parsimonious')

results <- data.frame(
    model = labels,
    n_params = sapply(models, function(m) length(coef(m))),
    r_squared = sapply(models, function(m) summary(m)$r.squared),
    aic = sapply(models, AIC),
    bic = sapply(models, BIC),
    stringsAsFactors = FALSE
)

cat(sprintf('%-25s %8s %10s %14s %14s\n', 'Model', 'Params', 'R²', 'AIC', 'BIC'))
cat(paste(rep('-', 75), collapse = ''), '\n')
for (i in 1:nrow(results)) {
    aic_flag <- ifelse(results$aic[i] == min(results$aic), ' <-- min', '')
    bic_flag <- ifelse(results$bic[i] == min(results$bic), ' <-- min', '')
    cat(sprintf('%-25s %8d %10.4f %14s%s %14s%s\n',
        results$model[i], results$n_params[i], results$r_squared[i],
        format(round(results$aic[i]), big.mark = ','), aic_flag,
        format(round(results$bic[i]), big.mark = ','), bic_flag))
}
```

- [ ] **Step 3: Add AIC/BIC chart**

Cell (code):
```r
%%R -w 800 -h 450
# Normalize AIC and BIC to make them plottable on same scale
aic_vals <- sapply(models, AIC)
bic_vals <- sapply(models, BIC)
n_params <- sapply(models, function(m) length(coef(m)))

par(mar = c(5, 5, 3, 5))
plot(n_params, aic_vals, type = 'b', pch = 16, col = 'steelblue', lwd = 2,
     xlab = 'Number of parameters', ylab = 'AIC',
     main = 'AIC and BIC across model progression')
par(new = TRUE)
plot(n_params, bic_vals, type = 'b', pch = 17, col = 'coral', lwd = 2,
     xlab = '', ylab = '', axes = FALSE)
axis(4, col = 'coral', col.axis = 'coral')
mtext('BIC', side = 4, line = 3, col = 'coral')

# Mark the minima
aic_min <- which.min(aic_vals)
bic_min <- which.min(bic_vals)
points(n_params[aic_min], aic_vals[aic_min], pch = 1, cex = 3, col = 'steelblue', lwd = 2)
points(n_params[bic_min], bic_vals[bic_min], pch = 1, cex = 3, col = 'coral', lwd = 2)

# Label models
text(n_params, aic_vals, labels = paste0('M', 1:10), pos = 3, cex = 0.7, col = 'steelblue')

legend('topright',
    c(sprintf('AIC (min at M%d)', aic_min), sprintf('BIC (min at M%d)', bic_min)),
    col = c('steelblue', 'coral'), pch = c(16, 17), lwd = 2, cex = 0.8)
```

- [ ] **Step 4: Add stepwise selection**

Cell (code):
```r
%%R
# Stepwise from Model 10 (both directions, AIC criterion)
# Suppress verbose output, just show result
cat('Running stepwise selection (both directions) from Model 10...\n\n')

step_model <- step(model10, direction = 'both', trace = 0)

# Compare: what did stepwise keep vs drop?
kept_vars <- names(coef(step_model))
orig_vars <- names(coef(model10))

dropped <- setdiff(orig_vars, kept_vars)
added <- setdiff(kept_vars, orig_vars)

cat(sprintf('Model 10 parameters: %d\n', length(orig_vars)))
cat(sprintf('Stepwise parameters: %d\n', length(kept_vars)))

if (length(dropped) > 0) {
    cat(sprintf('\nDropped by stepwise: %s\n', paste(dropped, collapse = ', ')))
} else {
    cat('\nStepwise kept all Model 10 variables.\n')
}

if (length(added) > 0) {
    cat(sprintf('Added by stepwise: %s\n', paste(added, collapse = ', ')))
}

cat(sprintf('\nAIC: Model 10 = %s, Stepwise = %s\n',
    format(round(AIC(model10)), big.mark = ','),
    format(round(AIC(step_model)), big.mark = ',')))
```

- [ ] **Step 5: Add interpretation markdown**

Cell (markdown):
```markdown
### Interpretation

**AIC/BIC across models:** [To be filled — expected: both AIC and BIC decrease steadily from M1 to M10, with the biggest drops at M3 (+lease) and M6 (+geography). If M10 is at or near the minimum for both, the model is the right size. If BIC prefers a smaller model (e.g., M9), that suggests the month fixed effects are adding complexity without enough predictive payoff — but given they control for a 5-8% market rise, they're substantively important regardless.]

**Stepwise result:** [Expected: stepwise keeps all or nearly all Model 10 variables. If it drops anything, it'll be one of the smaller-effect variables like park_dist_m or hospital_dist_m. If it keeps everything, that validates the manual variable selection.]

**Hensen's insight:** "As you add more variables, the RSS goes down, but the counterweight goes up." Our chart shows exactly this curve. Model 10 sits at [position] — [at the minimum / slightly past it / before it]. [Interpret what this means.]
```

- [ ] **Step 6: Run notebook and commit**

```bash
jupyter nbconvert --to notebook --execute 12-model-diagnostics.ipynb --output 12-model-diagnostics.ipynb
git add 12-model-diagnostics.ipynb
git commit -m "NB12: add AIC/BIC model selection and stepwise"
```

---

## Task 6: NB12 Section 5 — Coefficient Stability

**Files:**
- Modify: `12-model-diagnostics.ipynb`

- [ ] **Step 1: Add markdown header**

Cell (markdown):
```markdown
## 5. Coefficient stability across specifications

If a coefficient changes sign or magnitude dramatically when you add or remove other variables, it's fragile — probably confounded or absorbed by something else. Stable coefficients are trustworthy.

We track seven key variables across six model specifications: Models 5 through 10 (raw price), plus the log model (coefficients back-transformed to approximate dollar equivalents for comparability).
```

- [ ] **Step 2: Add coefficient stability table**

Cell (code):
```r
%%R
# Track key coefficients across specifications
track_vars <- c('num_eights_tail', 'price_has_168', 'block_has_4', 'cny_month',
                'dist_cbd_km', 'mrt_dist_m', 'columbarium_dist_m')

# Models to compare (m5 doesn't have all vars, so start from m6 where geo is added)
# For superstition vars: they first appear in m8
# Use models where the variable exists

# Collect coefficients and CIs for each model where the variable is present
spec_models <- list(m6, m7, m8, m9, model10)
spec_labels <- c('M6', 'M7', 'M8', 'M9', 'M10')

cat(sprintf('%-22s', 'Variable'))
for (lab in spec_labels) cat(sprintf('%14s', lab))
cat('\n')
cat(paste(rep('-', 22 + 14 * length(spec_labels)), collapse = ''), '\n')

for (v in track_vars) {
    cat(sprintf('%-22s', v))
    for (i in seq_along(spec_models)) {
        m <- spec_models[[i]]
        if (v %in% names(coef(m))) {
            robust <- coeftest(m, vcov = vcovHC(m, type = 'HC1'))
            coef_val <- robust[v, 'Estimate']
            cat(sprintf(' $%+12.0f', coef_val))
        } else {
            cat(sprintf('%14s', '--'))
        }
    }
    cat('\n')
}

# Add log model (back-transformed to dollar scale using median price)
median_price <- median(df$resale_price)
cat(sprintf('\n%-22s', ''))
cat(sprintf('%14s', 'Log (approx $)'))
cat('\n')

robust_log <- coeftest(model_log, vcov = vcovHC(model_log, type = 'HC1'))
for (v in track_vars) {
    if (v %in% rownames(robust_log)) {
        # Approximate dollar effect: coefficient * median_price
        approx_dollar <- robust_log[v, 'Estimate'] * median_price
        cat(sprintf('%-22s ~$%+11.0f\n', v, approx_dollar))
    }
}
```

- [ ] **Step 3: Add coefficient plot (forest plot style)**

Cell (code):
```r
%%R -w 900 -h 600
# Forest plot: point estimates + 95% CI for superstition variables across specs
super_vars <- c('num_eights_tail', 'price_has_168', 'block_has_4', 'cny_month')

# Only use models where all superstition vars exist (M8 onwards)
plot_models <- list(m8, m9, model10)
plot_labels <- c('M8 (full superstition)', 'M9 (+lease²+monthFE)', 'M10 (parsimonious)')
n_models <- length(plot_models)

par(mfrow = c(2, 2), mar = c(4, 2, 3, 1))

for (v in super_vars) {
    coefs <- numeric(n_models)
    ci_lo <- numeric(n_models)
    ci_hi <- numeric(n_models)

    for (i in seq_along(plot_models)) {
        m <- plot_models[[i]]
        robust <- coeftest(m, vcov = vcovHC(m, type = 'HC1'))
        # Note: m8 uses different variable names for some vars
        # num_eights_tail may not exist in m8 (it uses num_eights_in_price)
        if (v %in% rownames(robust)) {
            coefs[i] <- robust[v, 'Estimate']
            se <- robust[v, 'Std. Error']
            ci_lo[i] <- coefs[i] - 1.96 * se
            ci_hi[i] <- coefs[i] + 1.96 * se
        } else {
            coefs[i] <- NA
            ci_lo[i] <- NA
            ci_hi[i] <- NA
        }
    }

    valid <- !is.na(coefs)
    if (sum(valid) > 0) {
        xlim <- range(c(ci_lo[valid], ci_hi[valid]))
        xlim <- xlim + c(-0.1, 0.1) * diff(xlim)

        plot(coefs[valid], seq_along(plot_models)[valid], xlim = xlim,
             yaxt = 'n', pch = 16, cex = 1.5,
             xlab = 'Coefficient ($)', ylab = '',
             main = v)
        axis(2, at = seq_along(plot_models)[valid],
             labels = plot_labels[valid], las = 1, cex.axis = 0.7)
        segments(ci_lo[valid], seq_along(plot_models)[valid],
                 ci_hi[valid], seq_along(plot_models)[valid], lwd = 2)
        abline(v = 0, col = 'red', lty = 2)
    }
}
```

Note: `num_eights_tail` doesn't exist in M8 (which uses `num_eights_in_price`). The plot may only show M9 and M10 for that variable. Adjust the code if needed — could use `ends_in_8` from M5-M7 as an earlier proxy, or skip M8 for that variable.

- [ ] **Step 4: Add interpretation markdown**

Cell (markdown):
```markdown
### Interpretation

**Superstition variables:** [To be filled — expected: stable across M8-M10. block_has_4 should be the most stable (it's a physical characteristic of the block, not derived from price). num_eights_tail and price_has_168 might shift slightly as other variables are added/removed, but should keep the same sign and order of magnitude.]

**Distance variables:** [Expected: dist_cbd_km and mrt_dist_m are stable because they capture distinct spatial variation. columbarium_dist_m might shift more because it's partially correlated with specific towns.]

**Overall:** [Expected: "No coefficient changes sign across specifications. The largest movements are in [variable], which shifts by [X]% between M6 and M10. The superstition findings are robust to specification choice."]
```

- [ ] **Step 5: Add closing summary for NB12**

Cell (markdown):
```markdown
## Summary: Diagnostic Scorecard

| Diagnostic | Result | Concern level |
|---|---|---|
| Residual normality | [Q-Q result] | [Low/Medium/High] |
| Influential points | [Cook's D result] | [Low/Medium/High] |
| Multicollinearity (VIF) | [VIF result] | [Low/Medium/High] |
| Model size (AIC/BIC) | [Selection result] | [Low/Medium/High] |
| Coefficient stability | [Stability result] | [Low/Medium/High] |

**Bottom line:** [1-2 sentences summarising: which diagnostics the model passes cleanly, which flag concerns, and whether the concerns affect the published findings.]
```

- [ ] **Step 6: Run notebook and commit**

```bash
jupyter nbconvert --to notebook --execute 12-model-diagnostics.ipynb --output 12-model-diagnostics.ipynb
git add 12-model-diagnostics.ipynb
git commit -m "NB12: add coefficient stability and diagnostic summary"
```

---

## Task 7: Create Notebook 13 — Setup and Hold-Out Validation

**Files:**
- Create: `13-robustness-checks.ipynb`

- [ ] **Step 1: Create notebook with title, setup, and model reconstruction**

Cell 0 (markdown):
```markdown
# HDB Resale Price Regression — Notebook 13: Robustness Checks

Three stress tests on Model 10's findings:

1. **Hold-out validation** — train on 2/3 of data, predict the rest. Does the model generalise, or are the fixed effects overfitting?
2. **L1 (LAD) regression** — re-estimate using median regression. Are the results driven by outliers?
3. **Lucky-8 × price level** — does the 8-premium differ for cheap vs expensive flats?

If the findings survive all three, they're robust. If they don't, we'll know exactly where they break.
```

Cell 1 (code):
```python
%load_ext rpy2.ipython
import warnings
warnings.filterwarnings('ignore')
```

Cell 2 (code):
```r
%%R
library(tidyverse)
library(sandwich)
library(lmtest)
library(quantreg)

df <- read_csv('data/hdb_analysis.csv', show_col_types = FALSE)
df$remaining_lease_sq <- df$remaining_lease_years^2
df$month_factor <- factor(format(df$month, '%Y-%m'))
df$ln_price <- log(df$resale_price)

cat(sprintf('Loaded %s rows\n', format(nrow(df), big.mark = ',')))
```

- [ ] **Step 2: Add hold-out validation section header**

Cell (markdown):
```markdown
## 1. Hold-out validation

Mark Hensen: "Take a third of your data and hold it out. You have 50,000 points — hold 15,000 out, not a problem. Then check how well each model does at predicting."

If Model 10 explains 90% of price variation in-sample, how much does it explain on transactions it has never seen? A big drop means the model is memorising patterns specific to the training data (overfitting). A small drop means the model has learned genuine pricing relationships.
```

- [ ] **Step 3: Add train/test split and validation cell**

Cell (code):
```r
%%R
set.seed(2026)  # reproducible split
n <- nrow(df)
train_idx <- sample(1:n, size = round(2/3 * n))
test_idx <- setdiff(1:n, train_idx)

df_train <- df[train_idx, ]
df_test <- df[test_idx, ]

cat(sprintf('Training set: %s rows (%.0f%%)\n',
    format(nrow(df_train), big.mark = ','), nrow(df_train) / n * 100))
cat(sprintf('Test set:     %s rows (%.0f%%)\n',
    format(nrow(df_test), big.mark = ','), nrow(df_test) / n * 100))

# Fit Model 10 on training data only
model10_train <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
              remaining_lease_years + remaining_lease_sq +
              flat_model_grouped +
              dist_cbd_km + mrt_dist_m + hawker_dist_m +
              popular_school_dist_m +
              park_dist_m + hospital_dist_m +
              columbarium_dist_m + temple_dist_m +
              coast_dist_m +
              num_eights_tail +
              price_has_168 +
              block_has_4 +
              cny_month +
              month_factor,
            data = df_train)

# Predict on test data
test_pred <- predict(model10_train, newdata = df_test)
test_actual <- df_test$resale_price

# In-sample metrics
train_pred <- predict(model10_train, df_train)
train_actual <- df_train$resale_price
train_ss_res <- sum((train_actual - train_pred)^2)
train_ss_tot <- sum((train_actual - mean(train_actual))^2)
train_r2 <- 1 - train_ss_res / train_ss_tot

# Out-of-sample metrics
test_ss_res <- sum((test_actual - test_pred)^2)
test_ss_tot <- sum((test_actual - mean(test_actual))^2)
test_r2 <- 1 - test_ss_res / test_ss_tot

cat(sprintf('\n%-30s %12s %12s\n', '', 'Training', 'Test'))
cat(paste(rep('-', 55), collapse = ''), '\n')
cat(sprintf('%-30s %12.4f %12.4f\n', 'R-squared', train_r2, test_r2))
cat(sprintf('%-30s $%10s $%10s\n', 'Mean absolute error',
    format(round(mean(abs(train_actual - train_pred))), big.mark = ','),
    format(round(mean(abs(test_actual - test_pred))), big.mark = ',')))
cat(sprintf('%-30s $%10s $%10s\n', 'Median absolute error',
    format(round(median(abs(train_actual - train_pred))), big.mark = ','),
    format(round(median(abs(test_actual - test_pred))), big.mark = ',')))
cat(sprintf('%-30s $%10s $%10s\n', 'RMSE',
    format(round(sqrt(mean((train_actual - train_pred)^2))), big.mark = ','),
    format(round(sqrt(mean((test_actual - test_pred)^2))), big.mark = ',')))

cat(sprintf('\nR² drop: %.4f (%.1f%% relative drop)\n',
    train_r2 - test_r2, (train_r2 - test_r2) / train_r2 * 100))
```

- [ ] **Step 4: Add predicted vs actual scatter plot**

Cell (code):
```r
%%R -w 800 -h 500
plot(test_pred / 1000, test_actual / 1000,
     pch = '.', col = rgb(0, 0, 0, 0.1),
     xlab = 'Predicted price ($K)', ylab = 'Actual price ($K)',
     main = 'Hold-out validation: Predicted vs Actual (test set)',
     xlim = c(200, 1800), ylim = c(200, 1800))
abline(0, 1, col = 'red', lwd = 2)
abline(lm(test_actual ~ test_pred), col = 'blue', lty = 2, lwd = 1.5)

legend('topleft',
    c('Perfect prediction (45°)',
      sprintf('Best fit (R² = %.3f)', test_r2)),
    col = c('red', 'blue'), lty = c(1, 2), lwd = 2, cex = 0.8)
```

- [ ] **Step 5: Add hold-out by price quartile**

Cell (code):
```r
%%R
# Out-of-sample performance by price quartile
df_test$pred <- test_pred
df_test$quartile <- cut(df_test$resale_price,
    breaks = quantile(df_test$resale_price, probs = c(0, 0.25, 0.5, 0.75, 1)),
    labels = c('Q1 (cheapest)', 'Q2', 'Q3', 'Q4 (most expensive)'),
    include.lowest = TRUE)

cat(sprintf('%-20s %8s %12s %12s %12s\n',
    'Quartile', 'N', 'MAE', 'Median AE', 'MAPE'))
cat(paste(rep('-', 68), collapse = ''), '\n')

for (q in levels(df_test$quartile)) {
    subset <- df_test[df_test$quartile == q, ]
    ae <- abs(subset$resale_price - subset$pred)
    mape <- mean(ae / subset$resale_price) * 100
    cat(sprintf('%-20s %8d $%10s $%10s %10.1f%%\n',
        q, nrow(subset),
        format(round(mean(ae)), big.mark = ','),
        format(round(median(ae)), big.mark = ','),
        mape))
}
```

- [ ] **Step 6: Repeat for log model**

Cell (code):
```r
%%R
# Log model hold-out
model_log_train <- lm(ln_price ~ town + flat_type + floor_area_sqm + storey_mid +
              remaining_lease_years + remaining_lease_sq +
              flat_model_grouped +
              dist_cbd_km + mrt_dist_m + hawker_dist_m +
              popular_school_dist_m +
              park_dist_m + hospital_dist_m +
              columbarium_dist_m + temple_dist_m +
              coast_dist_m +
              num_eights_tail +
              price_has_168 +
              block_has_4 +
              cny_month +
              month_factor,
            data = df_train)

# Back-transform predictions to dollar scale
test_pred_log <- exp(predict(model_log_train, newdata = df_test))

test_r2_log_ss_res <- sum((test_actual - test_pred_log)^2)
test_r2_log <- 1 - test_r2_log_ss_res / test_ss_tot

cat(sprintf('%-30s %12s %12s\n', '', 'Raw model', 'Log model'))
cat(paste(rep('-', 55), collapse = ''), '\n')
cat(sprintf('%-30s %12.4f %12.4f\n', 'Test R² (dollar scale)', test_r2, test_r2_log))
cat(sprintf('%-30s $%10s $%10s\n', 'Test MAE',
    format(round(mean(abs(test_actual - test_pred))), big.mark = ','),
    format(round(mean(abs(test_actual - test_pred_log))), big.mark = ',')))
cat(sprintf('%-30s $%10s $%10s\n', 'Test median AE',
    format(round(median(abs(test_actual - test_pred))), big.mark = ','),
    format(round(median(abs(test_actual - test_pred_log))), big.mark = ',')))
```

- [ ] **Step 7: Add interpretation markdown**

Cell (markdown):
```markdown
### Interpretation

**Does the model generalise?** [To be filled — expected: test R² is within ~0.5-1% of training R². If R² drops from 0.90 to 0.89, the 86 parameters are not overfitting. If it drops to 0.85, the fixed effects are memorising training-set patterns.]

**By price quartile:** [Expected: MAE is roughly proportional to price level. The model should predict Q1 (cheap flats) within ~$30K and Q4 (expensive flats) within ~$60K. If Q4 is dramatically worse, it confirms that the model struggles with the high end (where views, renovation, and scarcity matter most).]

**Raw vs log:** [Expected: log model has lower MAE out-of-sample too, consistent with NB11 findings.]
```

- [ ] **Step 8: Run notebook and commit**

```bash
jupyter nbconvert --to notebook --execute 13-robustness-checks.ipynb --output 13-robustness-checks.ipynb
git add 13-robustness-checks.ipynb
git commit -m "Add notebook 13: hold-out validation"
```

---

## Task 8: NB13 Section 2 — L1 / LAD Regression

**Files:**
- Modify: `13-robustness-checks.ipynb`

- [ ] **Step 1: Add markdown header**

Cell (markdown):
```markdown
## 2. L1 (LAD) regression — robust to outliers

Mark Hensen: "Instead of drawing a line to minimise the squared error terms, you could draw a line to minimise the absolute value. The median. It will behave a lot better. Because the last thing you want is your answers driven by one super expensive place that's got eights in the value."

OLS minimises squared errors — like taking a mean. LAD (Least Absolute Deviations) minimises absolute errors — like taking a median. The median is resistant to outliers. If the lucky-8 coefficient survives LAD, it's not an artifact of a few extreme transactions.

**Package:** `quantreg::rq()` with `tau = 0.5` (median regression).
```

- [ ] **Step 2: Add LAD regression and comparison**

Cell (code):
```r
%%R
# LAD (median) regression with the same Model 10 formula
# quantreg::rq can be slow with many factor levels — this may take a minute
cat('Fitting LAD regression (median, tau = 0.5)...\n')

model10_lad <- rq(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
              remaining_lease_years + remaining_lease_sq +
              flat_model_grouped +
              dist_cbd_km + mrt_dist_m + hawker_dist_m +
              popular_school_dist_m +
              park_dist_m + hospital_dist_m +
              columbarium_dist_m + temple_dist_m +
              coast_dist_m +
              num_eights_tail +
              price_has_168 +
              block_has_4 +
              cny_month +
              month_factor,
            tau = 0.5,
            data = df)

cat('Done.\n\n')

# Compare key coefficients: OLS vs LAD
key_vars <- c('floor_area_sqm', 'storey_mid', 'remaining_lease_years',
              'dist_cbd_km', 'mrt_dist_m', 'hawker_dist_m',
              'columbarium_dist_m', 'temple_dist_m', 'coast_dist_m',
              'num_eights_tail', 'price_has_168', 'block_has_4', 'cny_month')

ols_coefs <- coef(model10)
lad_coefs <- coef(model10_lad)

# LAD summary for p-values (can be slow — use bootstrap or rank test)
cat('Computing LAD standard errors...\n')
lad_summary <- summary(model10_lad, se = 'rank')
cat('Done.\n\n')

cat(sprintf('%-25s %12s %12s %10s %8s\n',
    'Variable', 'OLS ($)', 'LAD ($)', 'Change', 'LAD p'))
cat(paste(rep('-', 70), collapse = ''), '\n')

for (v in key_vars) {
    if (v %in% names(ols_coefs) & v %in% names(lad_coefs)) {
        c_ols <- ols_coefs[v]
        c_lad <- lad_coefs[v]
        pct_change <- (c_lad - c_ols) / abs(c_ols) * 100

        # Get p-value from LAD summary
        lad_p <- tryCatch({
            lad_summary$coefficients[v, 'Pr(>|t|)']
        }, error = function(e) NA)

        p_str <- ifelse(is.na(lad_p), '  --',
                 ifelse(lad_p < 0.001, '<0.001',
                 sprintf('%.3f', lad_p)))
        sig <- ifelse(!is.na(lad_p) & lad_p < 0.05, '*', '')

        cat(sprintf('%-25s $%+10.0f $%+10.0f %+9.1f%% %7s%s\n',
            v, c_ols, c_lad, pct_change, p_str, sig))
    }
}
```

- [ ] **Step 3: Add interpretation markdown**

Cell (markdown):
```markdown
### Interpretation

**Superstition variables — OLS vs LAD:**

| Variable | OLS | LAD | Verdict |
|---|---|---|---|
| num_eights_tail | $1,070 | $? | [Survives / weakens / disappears] |
| price_has_168 | $32,795 | $? | [Survives / weakens / disappears] |
| block_has_4 | -$10,160 | $? | [Survives / weakens / disappears] |
| cny_month | $59,310 | $? | [Survives / weakens / disappears] |

[To be filled — expected: all superstition variables survive LAD with similar magnitudes. The lucky-8 premium might shrink slightly (from ~$1,070 to ~$800-900) because LAD down-weights the expensive eights-heavy transactions. But if it stays significant and positive, Hensen's concern is addressed: "The fact that the 8-premium survives LAD means it's not an artifact of outliers."]

**Distance variables:** [Expected: mostly stable. dist_cbd_km might shift because it's most affected by expensive central-area outliers.]

**Key finding:** [1-2 sentences on whether the core story holds up.]
```

- [ ] **Step 4: Run notebook and commit**

```bash
jupyter nbconvert --to notebook --execute 13-robustness-checks.ipynb --output 13-robustness-checks.ipynb
git add 13-robustness-checks.ipynb
git commit -m "NB13: add L1/LAD regression comparison"
```

---

## Task 9: NB13 Section 3 — Lucky-8 Interaction with Price Level

**Files:**
- Modify: `13-robustness-checks.ipynb`

- [ ] **Step 1: Add markdown header**

Cell (markdown):
```markdown
## 3. Does the lucky-8 premium differ by price level?

Mark Hensen: "Is it that these shenanigans work for lower price more often than upper price?"

Model 10 assumes the 8-premium is constant at ~$1,070 per trailing 8. But maybe it's bigger for cheaper flats (where $1K matters more) or bigger for expensive flats (where buyers with more money are more superstitious, or where sellers have more pricing power). Testing with an interaction term.

**Method:** Create price quartiles based on **predicted** price (from Model 10 without the superstition variables), not actual price. This avoids endogeneity — trailing 8s are part of the actual price, so splitting on actual price would be circular.
```

- [ ] **Step 2: Add price-quartile interaction model**

Cell (code):
```r
%%R
# First, get predicted price WITHOUT superstition variables
# This avoids endogeneity in the quartile assignment
model_no_super <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
              remaining_lease_years + remaining_lease_sq +
              flat_model_grouped +
              dist_cbd_km + mrt_dist_m + hawker_dist_m +
              popular_school_dist_m +
              park_dist_m + hospital_dist_m +
              columbarium_dist_m + temple_dist_m +
              coast_dist_m +
              month_factor,
            data = df)

df$pred_no_super <- predict(model_no_super, df)
df$price_quartile <- cut(df$pred_no_super,
    breaks = quantile(df$pred_no_super, probs = c(0, 0.25, 0.5, 0.75, 1)),
    labels = c('Q1', 'Q2', 'Q3', 'Q4'),
    include.lowest = TRUE)

cat('Price quartile ranges (based on predicted price ex-superstition):\n')
for (q in levels(df$price_quartile)) {
    subset <- df[df$price_quartile == q, ]
    cat(sprintf('  %s: $%s - $%s (n = %s)\n', q,
        format(round(min(subset$pred_no_super)), big.mark = ','),
        format(round(max(subset$pred_no_super)), big.mark = ','),
        format(nrow(subset), big.mark = ',')))
}

# Interaction model
model_interact <- lm(resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
              remaining_lease_years + remaining_lease_sq +
              flat_model_grouped +
              dist_cbd_km + mrt_dist_m + hawker_dist_m +
              popular_school_dist_m +
              park_dist_m + hospital_dist_m +
              columbarium_dist_m + temple_dist_m +
              coast_dist_m +
              num_eights_tail * price_quartile +
              price_has_168 +
              block_has_4 +
              cny_month +
              month_factor,
            data = df)

robust_int <- coeftest(model_interact, vcov = vcovHC(model_interact, type = 'HC1'))

# Extract the 8-premium for each quartile
base_8 <- robust_int['num_eights_tail', 'Estimate']  # Q1 (reference)
cat(sprintf('\nLucky-8 premium by price quartile:\n\n'))
cat(sprintf('  %-25s %10s %10s\n', 'Quartile', '8-premium', 'p-value'))
cat(paste(rep('-', 48), collapse = ''), '\n')

cat(sprintf('  %-25s $%+8.0f %10.4f\n', 'Q1 (cheapest 25%)',
    base_8, robust_int['num_eights_tail', 'Pr(>|t|)']))

for (q in c('Q2', 'Q3', 'Q4')) {
    int_name <- sprintf('num_eights_tail:price_quartile%s', q)
    if (int_name %in% rownames(robust_int)) {
        total <- base_8 + robust_int[int_name, 'Estimate']
        p_int <- robust_int[int_name, 'Pr(>|t|)']
        cat(sprintf('  %-25s $%+8.0f %10.4f (interaction p)\n',
            sprintf('%s', q), total, p_int))
    }
}
```

- [ ] **Step 3: Add correlation check — what clusters with trailing 8s?**

Cell (code):
```r
%%R
# What's correlated with num_eights_tail?
check_vars <- c('resale_price', 'floor_area_sqm', 'storey_mid',
                'remaining_lease_years', 'dist_cbd_km', 'mrt_dist_m',
                'columbarium_dist_m', 'num_fours_tail',
                'block_has_4', 'pred_no_super')

cat('Correlations with num_eights_tail:\n\n')
for (v in check_vars) {
    r <- cor(df$num_eights_tail, df[[v]], use = 'complete.obs')
    flag <- ifelse(abs(r) > 0.1, ' <--', '')
    cat(sprintf('  %-25s r = %+.4f%s\n', v, r, flag))
}
```

- [ ] **Step 4: Add interpretation markdown**

Cell (markdown):
```markdown
### Interpretation

**8-premium by price level:** [To be filled — expected: the 8-premium is roughly constant across quartiles in dollar terms (~$1,000 everywhere). This would mean it's a bigger deal for cheap flats (where $1K is a larger share of the price) and negligible for expensive ones. Alternatively, it might grow with price, suggesting wealthier buyers pay more for lucky numbers.]

**Clustering:** [Expected: num_eights_tail has near-zero correlation with everything except resale_price itself (which is mechanical — more expensive flats have more digits, more chances for 8s). If the correlation with pred_no_super (predicted price before superstition) is also near zero, the variable is genuinely exogenous to the pricing fundamentals.]

**Hensen's question answered:** [1-2 sentences.]
```

- [ ] **Step 5: Run notebook and commit**

```bash
jupyter nbconvert --to notebook --execute 13-robustness-checks.ipynb --output 13-robustness-checks.ipynb
git add 13-robustness-checks.ipynb
git commit -m "NB13: add lucky-8 interaction with price level"
```

---

## Task 10: NB13 Summary Table and Final Commit

**Files:**
- Modify: `13-robustness-checks.ipynb`

- [ ] **Step 1: Add summary table**

Cell (markdown):
```markdown
## Summary: Robustness Scorecard

| Check | Result | Implication |
|---|---|---|
| Out-of-sample R² (raw) | [?] vs 0.9023 in-sample | [Generalises / overfits] |
| Out-of-sample R² (log) | [?] vs 0.9373 in-sample | [Generalises / overfits] |
| Out-of-sample MAE | $[?] vs $46,402 in-sample | [Consistent / degrades] |
| LAD: num_eights_tail | $[?] vs $1,070 (OLS) | [Survives / weakens] |
| LAD: block_has_4 | $[?] vs -$10,160 (OLS) | [Survives / weakens] |
| LAD: price_has_168 | $[?] vs $32,795 (OLS) | [Survives / weakens] |
| LAD: cny_month | $[?] vs $59,310 (OLS) | [Survives / weakens] |
| 8-premium Q1 vs Q4 | $[?] vs $[?] | [Uniform / varies by price] |

**Bottom line:** [2-3 sentences. Expected: "Model 10's findings are robust. The 90% R-squared holds out of sample, ruling out overfitting. The superstition coefficients survive median regression, ruling out outlier-driven artifacts. The lucky-8 premium is [constant / varies] across price levels."]
```

- [ ] **Step 2: Run notebook fully and commit**

```bash
jupyter nbconvert --to notebook --execute 13-robustness-checks.ipynb --output 13-robustness-checks.ipynb
git add 13-robustness-checks.ipynb
git commit -m "NB13: add summary table, complete robustness checks"
```

- [ ] **Step 3: Final commit for both notebooks**

```bash
git add 12-model-diagnostics.ipynb 13-robustness-checks.ipynb
git commit -m "Complete notebooks 12-13: model diagnostics and robustness checks"
```
