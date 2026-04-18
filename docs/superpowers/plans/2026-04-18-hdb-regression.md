# HDB Resale Price Regression — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 6-notebook EDA-with-regression pipeline that answers "What predicts HDB resale prices?" and produces a statistically rigorous WTF flat outlier list, replacing the current z-score methodology.

**Architecture:** 6 Jupyter notebooks mirroring the GeBIZ EDA-with-regression pipeline at `/Users/wongpeiting/final-project/eda-with-regression-wongpeiting/`. Each notebook uses Python for data wrangling and R (via rpy2) for regression and ggplot visualisation. Data flows through CSVs between notebooks: API → cleaned → merged → regression-ready. The final notebook produces residuals that identify genuinely overpriced/underpriced flats.

**Tech Stack:** Python (pandas, numpy, matplotlib, requests), R via rpy2 (tidyverse, broom, scales, stargazer), Jupyter notebooks.

**Reference notebooks (for style/structure):**
- GeBIZ notebook 1: `/Users/wongpeiting/final-project/eda-with-regression-wongpeiting/1-download-your-data.ipynb`
- GeBIZ notebook 2: `/Users/wongpeiting/final-project/eda-with-regression-wongpeiting/2-explore-distributions.ipynb`
- GeBIZ notebook 6: `/Users/wongpeiting/final-project/eda-with-regression-wongpeiting/6-multivariable-regression.ipynb`
- WTF flat tracker (current methodology): `/Users/wongpeiting/Desktop/CU/python-work/wtf_flat_alert_system/auto_updating_site.ipynb`

**Spec:** `docs/superpowers/specs/2026-04-18-hdb-regression-design.md`

---

## File Structure

```
hdb-regression/
├── 1-download-your-data.ipynb
├── 2-explore-distributions.ipynb
├── 3-download-supplementary-data.ipynb
├── 4-merge.ipynb
├── 5-single-variable-regressions.ipynb
├── 6-multivariable-regression.ipynb
├── data/
│   ├── hdb_resale_raw.csv            # Raw API download
│   ├── hdb_resale_cleaned.csv        # Output of notebook 1
│   └── hdb_analysis.csv              # Output of notebook 4 (regression-ready)
└── requirements.txt
```

---

### Task 1: Project setup and data download (Notebook 1)

**Files:**
- Create: `requirements.txt`
- Create: `data/hdb_resale_raw.csv`
- Create: `data/hdb_resale_cleaned.csv`
- Create: `1-download-your-data.ipynb`

- [ ] **Step 1: Create requirements.txt**

```
pandas
numpy
matplotlib
seaborn
jupyter
rpy2
requests
```

- [ ] **Step 2: Create notebook 1**

Create `1-download-your-data.ipynb` with these cells in order:

**Cell 1 (markdown):**
```markdown
# Download and Prepare HDB Resale Data

**Question**: What drives HDB resale prices? And which transactions are outliers that can't be explained by location, size, age, or floor?

**Y variable**: `log_resale_price` (log10 of resale price) — continuous, for linear regression

**Data source**:
- data.gov.sg HDB Resale Flat Prices API
- Dataset ID: `d_8b84c4ee58e3cfc0ece0d773c8ca6abc`
- Scope: Last 2 years of transactions
```

**Cell 2 (markdown):**
```markdown
## Setup Python and R environment
```

**Cell 3 (code):**
```python
%load_ext rpy2.ipython
%load_ext autoreload
%autoreload 2

%matplotlib inline
from matplotlib import rcParams
rcParams['figure.figsize'] = (16, 100)

import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
```

**Cell 4 (code — R):**
```
%%R

require('tidyverse')
```

**Cell 5 (markdown):**
```markdown
## Download from data.gov.sg API

Using the same API the WTF flat tracker uses. Paginated — fetches 1,000 rows at a time.
```

**Cell 6 (code):**
```python
import requests
import time

dataset_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
base_url = "https://data.gov.sg/api/action/datastore_search"

all_records = []
offset = 0
limit = 1000

while True:
    params = {
        "resource_id": dataset_id,
        "limit": limit,
        "offset": offset,
    }
    
    print(f"Fetching offset {offset}...")
    resp = requests.get(base_url, params=params, timeout=60)
    
    if resp.status_code == 429:
        print("Rate limited — waiting 10s...")
        time.sleep(10)
        continue
    
    data = resp.json()
    
    if not data.get("success", False):
        print(f"API error: {data}")
        break
    
    records = data["result"]["records"]
    if not records:
        break
    
    all_records.extend(records)
    offset += limit
    time.sleep(1)  # be polite to the API

print(f"Total rows downloaded: {len(all_records):,}")

df = pd.DataFrame(all_records)
df.to_csv('data/hdb_resale_raw.csv', index=False)
print(f"Saved to data/hdb_resale_raw.csv")
```

**Cell 7 (markdown):**
```markdown
## Inspect raw data
```

**Cell 8 (code):**
```python
print(f"=== Raw dataset: {len(df):,} rows ===")
print(f"Columns: {list(df.columns)}")
print(f"\n=== Column types and nulls ===")
for col in df.columns:
    non_null = df[col].notna().sum()
    dtype = df[col].dtype
    print(f"  {col:25s}  {str(dtype):10s}  {non_null}/{len(df)} valid")

print(f"\n=== Date range ===")
df['month'] = pd.to_datetime(df['month'], errors='coerce')
print(f"  {df['month'].min().date()} to {df['month'].max().date()}")
print(f"  {df['month'].dt.to_period('M').nunique()} months of data")
```

**Cell 9 (markdown):**
```markdown
## Filter to last 2 years

We restrict to the last 2 years of transactions for two reasons:
1. Recent data reflects current market conditions (post-cooling measures)
2. ~50K+ rows is more than enough for stable regression coefficients

### EDITORIAL CHOICES

> **[SCOPE]** 2-year window from the most recent transaction date. This captures current market dynamics but misses long-term trends. A 5-year window with year fixed effects would be more comprehensive but adds complexity.

> **[MISSING DATA]** Rows with missing resale_price, floor_area_sqm, or lease_commence_date are dropped. These are rare and likely data entry errors.
```

**Cell 10 (code):**
```python
# Parse numeric columns
for col in ['resale_price', 'floor_area_sqm', 'lease_commence_date']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['month'] = pd.to_datetime(df['month'], errors='coerce')

# Drop rows with missing key fields
before = len(df)
df = df.dropna(subset=['resale_price', 'floor_area_sqm', 'lease_commence_date', 'month'])
print(f"Dropped {before - len(df)} rows with missing key fields")

# Filter to last 2 years
latest_month = df['month'].max()
cutoff = latest_month - pd.DateOffset(years=2)
df_recent = df[df['month'] >= cutoff].copy()

print(f"\n=== After filtering to last 2 years ===")
print(f"Date range: {df_recent['month'].min().date()} to {df_recent['month'].max().date()}")
print(f"Rows: {len(df_recent):,} (dropped {len(df) - len(df_recent):,} older transactions)")
print(f"Towns: {df_recent['town'].nunique()}")
print(f"Flat types: {df_recent['flat_type'].unique().tolist()}")
```

**Cell 11 (markdown):**
```markdown
## Derive variables

New variables needed for regression:

1. **`storey_mid`** — midpoint of storey range (e.g., "07 TO 09" → 8)
2. **`remaining_lease_years`** — approximate years of lease left at time of sale
3. **`log_resale_price`** — log10 of resale price (Y variable)
4. **`ends_in_8`** — does the price end in digit 8? (Singapore lucky number superstition)

### EDITORIAL CHOICES

> **[STOREY]** We extract the midpoint of the storey range as a continuous variable. "07 TO 09" → (7+9)/2 = 8. This is an approximation — the actual floor within the range is unknown.

> **[LEASE]** Remaining lease is calculated as 99 - (transaction_year - lease_commence_date). This assumes all HDB flats have 99-year leases, which is true for resale flats. The `remaining_lease` text field in the data (e.g., "71 years 10 months") is more precise but requires parsing; we use the simpler calculation and cross-check.

> **[LUCKY 8]** We test whether the last digit of the resale price is 8. This captures prices like $518,888 or $450,008. The hypothesis is that sellers who set "auspicious" prices may transact at higher prices — but the causal interpretation is ambiguous (see spec limitations).
```

**Cell 12 (code):**
```python
# Derive storey_mid from storey_range
# Format: "07 TO 09" → midpoint = (7 + 9) / 2 = 8
df_recent['storey_low'] = df_recent['storey_range'].str.extract(r'(\d+)\s+TO').astype(float)
df_recent['storey_high'] = df_recent['storey_range'].str.extract(r'TO\s+(\d+)').astype(float)
df_recent['storey_mid'] = (df_recent['storey_low'] + df_recent['storey_high']) / 2

print("=== storey_mid ===")
print(df_recent['storey_mid'].describe().to_string())
print(f"\nSample mapping:")
for sr in df_recent['storey_range'].unique()[:8]:
    mid = df_recent[df_recent['storey_range'] == sr]['storey_mid'].iloc[0]
    print(f"  '{sr}' → {mid}")
```

**Cell 13 (code):**
```python
# Derive remaining_lease_years
df_recent['transaction_year'] = df_recent['month'].dt.year
df_recent['remaining_lease_years'] = 99 - (df_recent['transaction_year'] - df_recent['lease_commence_date'])

print("=== remaining_lease_years ===")
print(df_recent['remaining_lease_years'].describe().to_string())

# Sanity check: any negative or >99?
bad_lease = (df_recent['remaining_lease_years'] < 0) | (df_recent['remaining_lease_years'] > 99)
print(f"\nBad values (negative or >99): {bad_lease.sum()}")
```

**Cell 14 (code):**
```python
# Derive log_resale_price (Y variable)
df_recent['log_resale_price'] = np.log10(df_recent['resale_price'])

print("=== log_resale_price ===")
print(f"Range: {df_recent['log_resale_price'].min():.3f} to {df_recent['log_resale_price'].max():.3f}")
print(f"  (${10**df_recent['log_resale_price'].min():,.0f} to ${10**df_recent['log_resale_price'].max():,.0f})")
```

**Cell 15 (code):**
```python
# Derive ends_in_8 (lucky number variable)
df_recent['last_digit'] = (df_recent['resale_price'] % 10).astype(int)
df_recent['ends_in_8'] = (df_recent['last_digit'] == 8).astype(int)

print("=== Lucky 8 distribution ===")
print(f"Prices ending in 8: {df_recent['ends_in_8'].sum():,} ({df_recent['ends_in_8'].mean()*100:.1f}%)")
print(f"Prices NOT ending in 8: {(1-df_recent['ends_in_8']).sum():,.0f} ({(1-df_recent['ends_in_8']).mean()*100:.1f}%)")

print(f"\n=== Last digit distribution ===")
digit_dist = df_recent['last_digit'].value_counts().sort_index()
for digit, count in digit_dist.items():
    bar = '#' * int(count / digit_dist.max() * 40)
    print(f"  {digit}: {count:6,} ({count/len(df_recent)*100:5.1f}%)  {bar}")

print(f"\nIf digits were random, each would be ~10%. Large deviations suggest seller psychology.")
```

**Cell 16 (code):**
```python
# Look at some examples
print("=== 10 most expensive transactions ===")
for _, row in df_recent.nlargest(10, 'resale_price').iterrows():
    e8 = " [ENDS IN 8]" if row['ends_in_8'] else ""
    print(f"  ${row['resale_price']:>10,.0f}  {row['town']:15s}  {row['flat_type']:12s}  "
          f"{row['storey_range']}  {row['floor_area_sqm']:.0f}sqm  "
          f"lease={row['remaining_lease_years']:.0f}yr{e8}")

print(f"\n=== 10 cheapest transactions ===")
for _, row in df_recent.nsmallest(10, 'resale_price').iterrows():
    e8 = " [ENDS IN 8]" if row['ends_in_8'] else ""
    print(f"  ${row['resale_price']:>10,.0f}  {row['town']:15s}  {row['flat_type']:12s}  "
          f"{row['storey_range']}  {row['floor_area_sqm']:.0f}sqm  "
          f"lease={row['remaining_lease_years']:.0f}yr{e8}")
```

**Cell 17 (code):**
```python
# Save cleaned data
keep_cols = [
    # Identifiers
    'month', 'town', 'block', 'street_name',
    # Y variable
    'resale_price', 'log_resale_price',
    # X variables
    'flat_type', 'flat_model', 'floor_area_sqm',
    'storey_range', 'storey_mid',
    'lease_commence_date', 'remaining_lease_years',
    'ends_in_8', 'last_digit',
    # Reference
    'remaining_lease',
]
keep_cols = [c for c in keep_cols if c in df_recent.columns]
df_out = df_recent[keep_cols].copy()

df_out.to_csv('data/hdb_resale_cleaned.csv', index=False)

print(f"Saved {len(df_out):,} rows to data/hdb_resale_cleaned.csv")
print(f"Columns: {list(df_out.columns)}")

print(f"\n=== Null check ===")
has_nulls = False
for col in df_out.columns:
    nulls = df_out[col].isna().sum()
    if nulls > 0:
        print(f"  {col}: {nulls} nulls ({nulls/len(df_out)*100:.1f}%)")
        has_nulls = True
if not has_nulls:
    print("  Zero nulls — ready for analysis!")
```

**Cell 18 (markdown):**
```markdown
## Known limitations

- **2-year window** — captures current market (post-cooling measures) but misses long-term trends. Could extend to 5 years with year fixed effects.
- **Storey is approximate** — "07 TO 09" → midpoint 8, but actual floor unknown within that range.
- **Remaining lease is approximate** — calculated from lease_commence_date, not from the precise `remaining_lease` text field (which has months-level precision).
- **No MRT distance** — one of the strongest price predictors, but not in this dataset. Added in Phase 2 via PropertyGuru.
- **No renovation/condition data** — a freshly renovated flat vs a run-down flat in the same block will have very different prices. Not captured.
- **`flat_model` has many levels** — some models have very few transactions. May need to group rare models in the regression.
```

- [ ] **Step 3: Run notebook 1**

Run the notebook end-to-end. Expected: `data/hdb_resale_cleaned.csv` created with ~50K+ rows. The API download may take 5-10 minutes due to pagination and rate limiting.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt 1-download-your-data.ipynb data/hdb_resale_raw.csv data/hdb_resale_cleaned.csv
git commit -m "feat: notebook 1 — download and prepare HDB resale data from data.gov.sg"
```

---

### Task 2: Explore distributions (Notebook 2)

**Files:**
- Create: `2-explore-distributions.ipynb`

- [ ] **Step 1: Create notebook 2**

Create `2-explore-distributions.ipynb` with these cells:

**Cell 1 (markdown):**
```markdown
# 1-D Exploratory Data Analysis: HDB Resale Prices

Exploring the distributions of our key variables before running regressions. What does the price landscape look like across towns, flat types, storeys, and lease durations?
```

**Cell 2 (code — Python setup):**
```python
%load_ext rpy2.ipython
%load_ext autoreload
%autoreload 2

%matplotlib inline
from matplotlib import rcParams
rcParams['figure.figsize'] = (16, 100)

import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
```

**Cell 3 (code — R setup):**
```
%%R

require('tidyverse')
require('scales')
```

**Cell 4 (code — R):**
```
%%R

df <- read_csv('data/hdb_resale_cleaned.csv', show_col_types = FALSE)
cat("Loaded", nrow(df), "HDB resale transactions\n")
```

**Cell 5 (markdown):**
```markdown
## Summary statistics
```

**Cell 6 (code — R):**
```
%%R

cat("=== Y variable: resale_price ===\n")
df |>
  summarise(
    count = n(),
    mean = mean(resale_price),
    median = median(resale_price),
    sd = sd(resale_price),
    min = min(resale_price),
    max = max(resale_price),
    pct_over_1M = mean(resale_price >= 1000000) * 100
  ) |> print()

cat("\n=== X variable: floor_area_sqm ===\n")
df |> summarise(count = n(), mean = mean(floor_area_sqm), median = median(floor_area_sqm),
                min = min(floor_area_sqm), max = max(floor_area_sqm)) |> print()

cat("\n=== X variable: storey_mid ===\n")
df |> summarise(count = n(), mean = mean(storey_mid), median = median(storey_mid),
                min = min(storey_mid), max = max(storey_mid)) |> print()

cat("\n=== X variable: remaining_lease_years ===\n")
df |> summarise(count = n(), mean = mean(remaining_lease_years), median = median(remaining_lease_years),
                min = min(remaining_lease_years), max = max(remaining_lease_years)) |> print()

cat("\n=== Categorical: town ===\n")
df |> count(town, sort = TRUE) |> print(n = 30)

cat("\n=== Categorical: flat_type ===\n")
df |> count(flat_type, sort = TRUE) |> print()

cat("\n=== Categorical: flat_model ===\n")
df |> count(flat_model, sort = TRUE) |> print(n = 25)

cat("\n=== Binary: ends_in_8 ===\n")
df |> count(ends_in_8) |> mutate(pct = n / sum(n) * 100) |> print()
```

**Cell 7 (markdown):**
```markdown
## 1-D visualisations

### Continuous variables
```

**Cell 8 (code — R):**
```
%%R -w 800 -h 500

ggplot(df) +
  aes(x = resale_price) +
  geom_histogram(bins = 50, fill = "#2c7fb8", color = "white", linewidth = 0.2) +
  geom_vline(aes(xintercept = median(resale_price)), linetype = "dashed", color = "red", linewidth = 0.8) +
  scale_x_continuous(labels = comma) +
  labs(title = "HDB resale prices are right-skewed — most under $600K",
       subtitle = "Red line = median. Long tail from million-dollar flats.",
       x = "Resale price (S$)", y = "Count") +
  theme_minimal()
```

**Cell 9 (code — R):**
```
%%R -w 800 -h 500

ggplot(df) +
  aes(x = log_resale_price) +
  geom_histogram(bins = 50, fill = "#2c7fb8", color = "white", linewidth = 0.2) +
  geom_vline(aes(xintercept = median(log_resale_price)), linetype = "dashed", color = "red", linewidth = 0.8) +
  labs(title = "Log-transformed prices are much more symmetric — good for regression",
       subtitle = "Red line = median.",
       x = "log10(resale price)", y = "Count") +
  theme_minimal()
```

**Cell 10 (code — R):**
```
%%R -w 800 -h 800

ggplot(df) +
  aes(x = reorder(town, resale_price, FUN = median), y = resale_price, fill = town) +
  geom_boxplot(show.legend = FALSE, outlier.alpha = 0.1, outlier.size = 0.5) +
  scale_y_continuous(labels = comma) +
  coord_flip() +
  labs(title = "Which towns have the most expensive HDB flats?",
       subtitle = "Resale price by town. Compare medians and spread.",
       x = NULL, y = "Resale price (S$)") +
  theme_minimal()
```

**Cell 11 (code — R):**
```
%%R -w 800 -h 500

ggplot(df) +
  aes(x = flat_type, y = resale_price, fill = flat_type) +
  geom_boxplot(show.legend = FALSE) +
  scale_y_continuous(labels = comma) +
  labs(title = "Bigger flat types cost more — but overlap is substantial",
       x = "Flat type", y = "Resale price (S$)") +
  theme_minimal()
```

**Cell 12 (code — R):**
```
%%R -w 800 -h 500

ggplot(df) +
  aes(x = storey_mid, y = log_resale_price) +
  geom_point(alpha = 0.05, size = 0.5, color = "grey40") +
  geom_smooth(method = "lm", color = "#e41a1c", linewidth = 1) +
  labs(title = "Higher floors → higher prices",
       subtitle = "Each dot = one transaction. Red line = linear fit.",
       x = "Storey (midpoint of range)", y = "log10(resale price)") +
  theme_minimal()
```

**Cell 13 (code — R):**
```
%%R -w 800 -h 500

ggplot(df) +
  aes(x = remaining_lease_years, y = log_resale_price) +
  geom_point(alpha = 0.05, size = 0.5, color = "grey40") +
  geom_smooth(method = "loess", color = "#e41a1c", linewidth = 1) +
  labs(title = "The lease decay curve — at what point does price drop sharply?",
       subtitle = "LOESS smoother (red) shows the non-linear relationship.",
       x = "Remaining lease (years)", y = "log10(resale price)") +
  theme_minimal()
```

**Cell 14 (code — R):**
```
%%R -w 800 -h 500

ggplot(df) +
  aes(x = floor_area_sqm, y = log_resale_price) +
  geom_point(alpha = 0.05, size = 0.5, color = "grey40") +
  geom_smooth(method = "lm", color = "#e41a1c", linewidth = 1) +
  labs(title = "Bigger flats cost more — strong linear relationship",
       x = "Floor area (sqm)", y = "log10(resale price)") +
  theme_minimal()
```

**Cell 15 (markdown):**
```markdown
### The lucky 8
```

**Cell 16 (code — R):**
```
%%R -w 800 -h 500

# Last digit distribution — are sellers drawn to certain digits?
ggplot(df) +
  aes(x = factor(last_digit)) +
  geom_bar(fill = "#2c7fb8", alpha = 0.8) +
  geom_hline(yintercept = nrow(df) / 10, linetype = "dashed", color = "red", linewidth = 0.5) +
  labs(title = "Which digits do HDB resale prices end in?",
       subtitle = "Red line = expected if digits were random (10% each). 0 and 8 stand out.",
       x = "Last digit of resale price", y = "Count") +
  theme_minimal()
```

**Cell 17 (code — R):**
```
%%R -w 800 -h 400

# Price comparison: ends in 8 vs doesn't
ggplot(df) +
  aes(x = factor(ends_in_8, labels = c("Other digit", "Ends in 8")),
      y = resale_price, fill = factor(ends_in_8)) +
  geom_boxplot(show.legend = FALSE, width = 0.5) +
  scale_y_continuous(labels = comma) +
  labs(title = "Do prices ending in 8 tend to be higher?",
       subtitle = "Raw comparison (not controlling for flat type, town, etc.)",
       x = NULL, y = "Resale price (S$)") +
  theme_minimal()
```

**Cell 18 (markdown):**
```markdown
### Faceted: storey vs price by town
```

**Cell 19 (code — R):**
```
%%R -w 1200 -h 800

# Top 9 towns by volume — storey premium within each
top_towns <- df |> count(town, sort = TRUE) |> slice_head(n = 9) |> pull(town)

ggplot(df |> filter(town %in% top_towns)) +
  aes(x = storey_mid, y = log_resale_price) +
  geom_point(alpha = 0.05, size = 0.3, color = "grey40") +
  geom_smooth(method = "lm", color = "#e41a1c", linewidth = 1) +
  facet_wrap(~town) +
  labs(title = "Does the high-floor premium differ by town?",
       subtitle = "Compare slopes — steeper = stronger floor premium.",
       x = "Storey (midpoint)", y = "log10(resale price)") +
  theme_minimal()
```

- [ ] **Step 2: Run notebook 2**

Expected: all plots render, summary stats print cleanly.

- [ ] **Step 3: Commit**

```bash
git add 2-explore-distributions.ipynb
git commit -m "feat: notebook 2 — explore HDB resale price distributions"
```

---

### Task 3: Supplementary data placeholder (Notebook 3)

**Files:**
- Create: `3-download-supplementary-data.ipynb`

- [ ] **Step 1: Create notebook 3**

Create `3-download-supplementary-data.ipynb` with these cells:

**Cell 1 (markdown):**
```markdown
# Download Supplementary Data

## Phase 2 (future): PropertyGuru enrichment

This notebook is a placeholder for Phase 2, where we'll join PropertyGuru scraped listings to add:
- MRT proximity (nearest_mrt, distance)
- Furnished status
- Green score (BCA Green Mark)
- Agent data

For Phase 1, no supplementary data is needed — all variables come from data.gov.sg.

**PropertyGuru data location:** `/Users/wongpeiting/propertyguru/data/propertyguru_hdb_listings.csv`

**Join key:** block + street_name (after normalising case and whitespace)
```

**Cell 2 (code):**
```python
# Phase 2 placeholder — uncomment when ready to join PropertyGuru data

# import pandas as pd
# pg = pd.read_csv('/Users/wongpeiting/propertyguru/data/propertyguru_hdb_listings.csv')
# print(f"PropertyGuru HDB listings: {len(pg):,}")

print("Phase 1: no supplementary data needed. Proceed to notebook 4.")
```

- [ ] **Step 2: Commit**

```bash
git add 3-download-supplementary-data.ipynb
git commit -m "feat: notebook 3 — supplementary data placeholder for Phase 2"
```

---

### Task 4: Merge (Notebook 4)

**Files:**
- Create: `4-merge.ipynb`
- Create: `data/hdb_analysis.csv`

- [ ] **Step 1: Create notebook 4**

Create `4-merge.ipynb` with these cells:

**Cell 1 (markdown):**
```markdown
# Merge into Master Regression Table

Build the final analysis-ready dataset. In Phase 1, this is straightforward — just the cleaned data.gov.sg data with derived variables. Phase 2 will add PropertyGuru enrichment here.
```

**Cell 2 (code):**
```python
%load_ext rpy2.ipython
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from IPython.display import display, HTML
```

**Cell 3 (code):**
```python
df = pd.read_csv('data/hdb_resale_cleaned.csv')
print(f"Loaded {len(df):,} HDB resale transactions")
print(f"Columns: {list(df.columns)}")
```

**Cell 4 (code):**
```python
# Ensure correct types
df['month'] = pd.to_datetime(df['month'])
df['ends_in_8'] = df['ends_in_8'].astype(int)

# Check flat_model levels — group rare models
print("=== flat_model counts ===")
print("(Models with < 50 transactions may give unstable coefficients)\n")
model_counts = df['flat_model'].value_counts()
for model, count in model_counts.items():
    flag = " <-- RARE" if count < 50 else ""
    print(f"  {model:25s}: {count:6,}{flag}")

# Group rare models into "Other"
rare_models = model_counts[model_counts < 50].index.tolist()
if rare_models:
    df['flat_model_grouped'] = df['flat_model'].replace(rare_models, 'Other')
    print(f"\nGrouped {len(rare_models)} rare models into 'Other': {rare_models}")
else:
    df['flat_model_grouped'] = df['flat_model']
    print("\nNo rare models — all have 50+ transactions")
```

**Cell 5 (code):**
```python
# Final null check
print("=== Null check ===")
has_nulls = False
for col in df.columns:
    nulls = df[col].isna().sum()
    if nulls > 0:
        print(f"  {col}: {nulls} nulls ({nulls/len(df)*100:.1f}%)")
        has_nulls = True
if not has_nulls:
    print("  Zero nulls — ready for regression!")

print(f"\n=== Variable ranges ===")
for col in ['resale_price', 'log_resale_price', 'floor_area_sqm', 'storey_mid',
            'remaining_lease_years', 'ends_in_8']:
    valid = df[col].dropna()
    print(f"  {col:25s}: {valid.min():.2f} to {valid.max():.2f} (n={len(valid)})")
```

**Cell 6 (code):**
```python
df.to_csv('data/hdb_analysis.csv', index=False)
print(f"\nSaved {len(df):,} rows to data/hdb_analysis.csv")
print(f"This is the input for notebooks 5 and 6.")
```

- [ ] **Step 2: Run notebook 4**

Expected: `data/hdb_analysis.csv` created.

- [ ] **Step 3: Commit**

```bash
git add 4-merge.ipynb data/hdb_analysis.csv
git commit -m "feat: notebook 4 — merge into master regression table"
```

---

### Task 5: Single-variable regressions (Notebook 5)

**Files:**
- Create: `5-single-variable-regressions.ipynb`

- [ ] **Step 1: Create notebook 5**

Create `5-single-variable-regressions.ipynb` with these cells:

**Cell 1 (markdown):**
```markdown
# Single-Variable Regressions: What Predicts HDB Resale Prices?

**Y = `log_resale_price`** (log10 of resale price)

Before building a multivariable model, test each predictor on its own. This tells us which variables have ANY relationship with price — and how strong each one is individually.
```

**Cell 2 (code — Python setup):**
```python
%load_ext rpy2.ipython
%load_ext autoreload
%autoreload 2

%matplotlib inline
from matplotlib import rcParams
rcParams['figure.figsize'] = (16, 100)

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from IPython.display import display, HTML
```

**Cell 3 (code — R setup):**
```
%%R

require('tidyverse')
require('broom')
require('scales')
```

**Cell 4 (code — R):**
```
%%R

df <- read_csv('data/hdb_analysis.csv', show_col_types = FALSE)
cat("Loaded", nrow(df), "transactions\n")
```

**Cell 5 (markdown):**
```markdown
## 1. Town (location)
```

**Cell 6 (code — R):**
```
%%R -w 800 -h 700

ggplot(df) +
  aes(x = reorder(town, log_resale_price, FUN = median), y = log_resale_price, fill = town) +
  geom_boxplot(show.legend = FALSE, outlier.alpha = 0.05, outlier.size = 0.3) +
  coord_flip() +
  labs(title = "Location matters enormously for HDB prices",
       x = NULL, y = "log10(resale price)") +
  theme_minimal()
```

**Cell 7 (code — R):**
```
%%R

m_town <- lm(log_resale_price ~ town, data = df)
cat("=== Town only ===\n")
cat("R-squared:", round(summary(m_town)$r.squared, 4), "\n")
cat("Adj R-squared:", round(summary(m_town)$adj.r.squared, 4), "\n")
f <- summary(m_town)$fstatistic
p <- pf(f[1], f[2], f[3], lower.tail = FALSE)
cat("F-test p-value:", format(p, scientific = TRUE), "\n")
```

**Cell 8 (markdown):**
```markdown
## 2. Flat type
```

**Cell 9 (code — R):**
```
%%R

m_flat_type <- lm(log_resale_price ~ flat_type, data = df)
cat("=== Flat type only ===\n")
cat("R-squared:", round(summary(m_flat_type)$r.squared, 4), "\n")
f <- summary(m_flat_type)$fstatistic
p <- pf(f[1], f[2], f[3], lower.tail = FALSE)
cat("F-test p-value:", format(p, scientific = TRUE), "\n")
summary(m_flat_type)
```

**Cell 10 (markdown):**
```markdown
## 3. Floor area
```

**Cell 11 (code — R):**
```
%%R -w 800 -h 500

ggplot(df) +
  aes(x = floor_area_sqm, y = log_resale_price) +
  geom_point(alpha = 0.03, size = 0.5, color = "grey40") +
  geom_smooth(method = "lm", color = "#e41a1c", linewidth = 1) +
  labs(title = "Bigger flats cost more", x = "Floor area (sqm)", y = "log10(resale price)") +
  theme_minimal()
```

**Cell 12 (code — R):**
```
%%R

m_area <- lm(log_resale_price ~ floor_area_sqm, data = df)
summary(m_area)
```

**Cell 13 (markdown):**
```markdown
## 4. Storey (floor level)
```

**Cell 14 (code — R):**
```
%%R

m_storey <- lm(log_resale_price ~ storey_mid, data = df)
summary(m_storey)

cat("\nInterpretation: Each additional floor adds approximately",
    round((10^(coef(m_storey)["storey_mid"]) - 1) * 100, 2),
    "% to the resale price.\n")
```

**Cell 15 (markdown):**
```markdown
## 5. Remaining lease
```

**Cell 16 (code — R):**
```
%%R

m_lease <- lm(log_resale_price ~ remaining_lease_years, data = df)
summary(m_lease)
```

**Cell 17 (markdown):**
```markdown
## 6. Flat model
```

**Cell 18 (code — R):**
```
%%R

m_model <- lm(log_resale_price ~ flat_model_grouped, data = df)
cat("=== Flat model only ===\n")
cat("R-squared:", round(summary(m_model)$r.squared, 4), "\n")
summary(m_model)
```

**Cell 19 (markdown):**
```markdown
## 7. Ends in 8 (lucky number)
```

**Cell 20 (code — R):**
```
%%R

m_lucky8 <- lm(log_resale_price ~ ends_in_8, data = df)
summary(m_lucky8)

cat("\nRaw effect: prices ending in 8 are on average",
    round((10^(coef(m_lucky8)["ends_in_8"]) - 1) * 100, 2),
    "% different from other prices.\n")
cat("But this doesn't control for anything — the effect may disappear in the multivariable model.\n")
```

**Cell 21 (markdown):**
```markdown
## Summary: which variables matter?
```

**Cell 22 (code — R):**
```
%%R

models <- list(
  list(name = "town", model = m_town),
  list(name = "flat_type", model = m_flat_type),
  list(name = "floor_area_sqm", model = m_area),
  list(name = "storey_mid", model = m_storey),
  list(name = "remaining_lease_years", model = m_lease),
  list(name = "flat_model_grouped", model = m_model),
  list(name = "ends_in_8", model = m_lucky8)
)

cat("=== Single-variable regression summary ===\n")
cat(sprintf("%-25s %10s %10s %s\n", "Variable", "R²", "p-value", "Significant?"))
cat(paste(rep("-", 60), collapse = ""), "\n")

for (m in models) {
  s <- summary(m$model)
  f <- s$fstatistic
  p <- pf(f[1], f[2], f[3], lower.tail = FALSE)
  sig <- ifelse(p < 0.01, "***", ifelse(p < 0.05, "**", ifelse(p < 0.1, "*", "")))
  cat(sprintf("%-25s %10.4f %10.6f %s\n", m$name, s$r.squared, p, sig))
}

cat("\n*** p < 0.01  ** p < 0.05  * p < 0.1\n")
```

- [ ] **Step 2: Run notebook 5**

Expected: summary table ranks variables by R² and p-value.

- [ ] **Step 3: Commit**

```bash
git add 5-single-variable-regressions.ipynb
git commit -m "feat: notebook 5 — single-variable regressions for HDB prices"
```

---

### Task 6: Multivariable regression (Notebook 6)

**Files:**
- Create: `6-multivariable-regression.ipynb`

- [ ] **Step 1: Create notebook 6**

Create `6-multivariable-regression.ipynb` with these cells:

**Cell 1 (markdown):**
```markdown
# Multivariable Regression: What Predicts HDB Resale Prices?

**Y = `log_resale_price`** (log10 of resale price)

From notebook 5, we identified which single predictors matter. Now we build up five models to see if these effects **survive** when we control for everything:

1. **Model 1** — Town + flat type (location baseline)
2. **Model 2** — + Floor area + storey (physical attributes)
3. **Model 3** — + Remaining lease (age/decay)
4. **Model 4** — + Flat model (DBSS, Premium, etc.)
5. **Model 5** — + Ends in 8 (lucky number test)

Then: residual analysis to identify WTF flats.
```

**Cell 2 (code — Python setup):**
```python
%load_ext rpy2.ipython
%load_ext autoreload
%autoreload 2

%matplotlib inline
from matplotlib import rcParams
rcParams['figure.figsize'] = (16, 100)

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from IPython.display import display, HTML
```

**Cell 3 (code — R setup):**
```
%%R

require('tidyverse')
require('broom')
require('scales')
require('stargazer')
```

**Cell 4 (code — R):**
```
%%R

df <- read_csv('data/hdb_analysis.csv', show_col_types = FALSE)
cat("Loaded", nrow(df), "transactions\n")
```

**Cell 5 (markdown):**
```markdown
### Model 1: `town` + `flat_type`

Location and flat type = baseline. These are the two variables everyone knows matter. How much of HDB price variation do they explain on their own?
```

**Cell 6 (code — R):**
```
%%R

model1 <- lm(log_resale_price ~ town + flat_type, data = df)
cat("=== Model 1: town + flat_type ===\n")
cat("R-squared:", round(summary(model1)$r.squared, 4), "\n")
cat("Adj R-squared:", round(summary(model1)$adj.r.squared, 4), "\n")
```

**Cell 7 (markdown):**
```markdown
### Model 2: + `floor_area_sqm` + `storey_mid`

Physical attributes — does knowing the size and floor add to what town and flat type already tell us?
```

**Cell 8 (code — R):**
```
%%R

model2 <- lm(log_resale_price ~ town + flat_type + floor_area_sqm + storey_mid, data = df)
cat("=== Model 2: + floor_area + storey ===\n")
cat("R-squared:", round(summary(model2)$r.squared, 4), "\n")
cat("Adj R-squared:", round(summary(model2)$adj.r.squared, 4), "\n")

cat("\nStorey premium: each additional floor adds approximately",
    round((10^(coef(model2)["storey_mid"]) - 1) * 100, 2), "%\n")

cat("Size premium: each additional sqm adds approximately",
    round((10^(coef(model2)["floor_area_sqm"]) - 1) * 100, 2), "%\n")
```

**Cell 9 (markdown):**
```markdown
### Model 3: + `remaining_lease_years`

The lease decay question: after controlling for location, type, size, and floor, does remaining lease independently affect price? And by how much?
```

**Cell 10 (code — R):**
```
%%R

model3 <- lm(log_resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
               remaining_lease_years, data = df)
cat("=== Model 3: + remaining_lease_years ===\n")
cat("R-squared:", round(summary(model3)$r.squared, 4), "\n")
cat("Adj R-squared:", round(summary(model3)$adj.r.squared, 4), "\n")

cat("\nLease effect: each additional year of remaining lease adds approximately",
    round((10^(coef(model3)["remaining_lease_years"]) - 1) * 100, 2), "%\n")
```

**Cell 11 (markdown):**
```markdown
### Model 4: + `flat_model_grouped`

Do DBSS, Premium Apartment, and Maisonette command genuine premiums beyond what their size, location, and floor already explain?
```

**Cell 12 (code — R):**
```
%%R

model4 <- lm(log_resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
               remaining_lease_years + flat_model_grouped, data = df)
cat("=== Model 4: + flat_model ===\n")
cat("R-squared:", round(summary(model4)$r.squared, 4), "\n")
cat("Adj R-squared:", round(summary(model4)$adj.r.squared, 4), "\n")
summary(model4)
```

**Cell 13 (markdown):**
```markdown
### Model 5: + `ends_in_8` (the lucky number test)

The fun one. After controlling for town, flat type, size, floor, lease, and model — does having a price that ends in 8 predict a higher sale price?

If significant: sellers who use "auspicious" pricing get higher prices. But the causal interpretation is ambiguous — it could be that savvy sellers both price at 8 endings AND are better negotiators.
```

**Cell 14 (code — R):**
```
%%R

model5 <- lm(log_resale_price ~ town + flat_type + floor_area_sqm + storey_mid +
               remaining_lease_years + flat_model_grouped + ends_in_8, data = df)
cat("=== Model 5: + ends_in_8 ===\n")
cat("R-squared:", round(summary(model5)$r.squared, 4), "\n")

cat("\nLucky 8 coefficient:", round(coef(model5)["ends_in_8"], 6), "\n")
cat("Lucky 8 effect: prices ending in 8 are",
    round((10^(coef(model5)["ends_in_8"]) - 1) * 100, 2),
    "% higher/lower after controlling for everything.\n")
cat("p-value:", format(summary(model5)$coefficients["ends_in_8", "Pr(>|t|)"], scientific = TRUE), "\n")
```

**Cell 15 (markdown):**
```markdown
## Compare all five models
```

**Cell 16 (code — R):**
```
%%R

stargazer(model1, model2, model3, model4, model5, type = "text",
          title = "What predicts HDB resale prices? Building up from location to lucky numbers",
          column.labels = c("Location", "+ Physical", "+ Lease", "+ Model", "+ Lucky 8"),
          dep.var.labels = "log10(resale price)",
          omit = c("town", "flat_model_grouped"),
          omit.labels = c("Town dummies (26)", "Flat model dummies"))
```

**Cell 17 (code — R):**
```
%%R

cat("=== R-SQUARED PROGRESSION ===\n\n")
cat(sprintf("%-35s %10s %10s\n", "Model", "R²", "Adj R²"))
cat(paste(rep("-", 57), collapse = ""), "\n")

models <- list(model1, model2, model3, model4, model5)
labels <- c(
  "1: town + flat_type",
  "2: + floor_area + storey",
  "3: + remaining_lease",
  "4: + flat_model",
  "5: + ends_in_8 (lucky number)"
)
for (i in seq_along(models)) {
  s <- summary(models[[i]])
  cat(sprintf("%-35s %10.5f %10.5f\n", labels[i], s$r.squared, s$adj.r.squared))
}

cat("\n=== ANOVA: Is each step a significant improvement? ===\n")
print(anova(model1, model2, model3, model4, model5))
```

**Cell 18 (markdown):**
```markdown
## Residual analysis: WTF flats

**Data = Model + Error.** The model tells us what's "normal." The residuals tell us what's abnormal — and abnormal is the story.

Transactions with large positive residuals: the flat sold for **far more** than the model predicted, even after controlling for town, flat type, size, floor, lease, model, and lucky pricing. These are the WTF flats.
```

**Cell 19 (code — R):**
```
%%R -w 800 -h 500

model5_aug <- augment(model5)

ggplot(model5_aug) +
  aes(x = .fitted, y = .resid) +
  geom_point(alpha = 0.05, size = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs Fitted — looking for patterns the model missed",
       subtitle = "Points far from 0 = transactions the model got wrong. Investigation leads.",
       x = "Predicted log10(price)", y = "Residual") +
  theme_minimal()
```

**Cell 20 (code — R):**
```
%%R

df$predicted <- predict(model5, df)
df$residual <- df$log_resale_price - df$predicted
df$predicted_price <- round(10^df$predicted)
df$residual_pct <- round((10^df$residual - 1) * 100, 1)

cat("=== Top 20 WTF flats (highest positive residuals) ===\n")
cat("Sold for MORE than the model predicted — even after controlling for everything.\n\n")

df |>
  arrange(desc(residual)) |>
  slice_head(n = 20) |>
  mutate(
    address = paste(block, street_name),
    actual = paste0("$", format(resale_price, big.mark = ",")),
    predicted_fmt = paste0("$", format(predicted_price, big.mark = ",")),
    diff = paste0("+", residual_pct, "%")
  ) |>
  select(town, address, flat_type, flat_model, storey_range,
         floor_area_sqm, remaining_lease_years,
         actual, predicted_fmt, diff) |>
  print(n = 20)
```

**Cell 21 (code — R):**
```
%%R

cat("\n=== Top 20 BARGAIN flats (most negative residuals) ===\n")
cat("Sold for LESS than the model predicted.\n\n")

df |>
  arrange(residual) |>
  slice_head(n = 20) |>
  mutate(
    address = paste(block, street_name),
    actual = paste0("$", format(resale_price, big.mark = ",")),
    predicted_fmt = paste0("$", format(predicted_price, big.mark = ",")),
    diff = paste0(residual_pct, "%")
  ) |>
  select(town, address, flat_type, flat_model, storey_range,
         floor_area_sqm, remaining_lease_years,
         actual, predicted_fmt, diff) |>
  print(n = 20)
```

**Cell 22 (markdown):**
```markdown
## Final reflection

### What we found

[Fill in after running — summarise the R² progression and key findings]

### The lucky 8 verdict

[Fill in — was ends_in_8 significant after controlling for everything?]

### WTF flats: regression vs z-score

The regression-based WTF list is more rigorous than the current z-score approach because it controls for all factors simultaneously. A flat that the old system flagged as "WTF" might simply be on a high floor in a desirable model — the regression accounts for this.

### Story leads from residuals

[Fill in — which WTF flats are most interesting? Which towns appear most in the outlier list?]

### Next steps

1. **Phase 2**: Join PropertyGuru data to add MRT distance, furnishing, and green score
2. **Replace WTF tracker methodology**: Feed regression residuals into the auto-updating tracker
3. **Investigate outliers**: The top WTF flats are interview leads — what's special about these transactions?
```

- [ ] **Step 2: Run notebook 6**

Expected: all 5 models run, stargazer table prints, ANOVA shows R² progression, top 20 WTF/bargain flats identified.

- [ ] **Step 3: Commit**

```bash
git add 6-multivariable-regression.ipynb
git commit -m "feat: notebook 6 — multivariable regression and WTF flat residual analysis"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run all notebooks in sequence (1 through 6)**

Verify each notebook executes without errors and produces expected outputs.

- [ ] **Step 2: Fill in reflections in notebook 6**

After seeing actual results, complete the "Final reflection" section (Cell 22) with:
- Key findings from the R² progression
- Lucky 8 verdict
- Most interesting WTF flat outliers
- Story leads

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete 6-notebook HDB regression pipeline"
```
