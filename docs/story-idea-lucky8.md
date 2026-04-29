# Story idea: Only one superstition survives scrutiny — and it's the rich who pay for it

## The pitch

We tested four superstition effects in 50,000 HDB resale transactions over two years (May 2024 – April 2026). All four looked statistically significant in the initial model. Then we stress-tested them with eight different methods — median regression, outlier removal, hold-out validation, interaction models, multicollinearity checks, AIC/BIC model selection, coefficient stability, and Q-Q diagnostics.

Three collapsed. One survived everything: the lucky 8.

## What collapsed

- **"168" prices (一路发 "prosperity all the way")** — the apparent $33K premium was driven by a handful of very expensive transactions. Median regression (which ignores outliers) killed it: p = 0.198. Not a robust finding.
- **CNY month timing** — the $59K "festive premium" was a data artifact. March 2026 was the only CNY month in the dataset, and the coefficient was just absorbing that month's general price level. Once the aliasing was resolved, the true effect was ~$880 — indistinguishable from zero.
- **Block number 4 (死 "death")** — looked like a $10K discount for blocks containing the digit 4. But 60% of the effect disappeared once the model accounted for the fact that block-4 flats happen to cluster in cheaper town/storey combinations. The remaining ~$4K was too fragile to publish with confidence.

## What survived

**Trailing 8s in the resale price: +$1,710 per trailing 8 digit** (p < 0.001 across every specification).

The coefficient actually *strengthened* the more we controlled for:
- Model 10 (original): $1,070
- After removing influential outliers (Cook's distance): $1,354
- Under median regression (LAD): $1,371
- Final model with town-level interactions (Model 12): $1,710

## The twist: it's a rich-buyer phenomenon

The $1,710 average masks sharp variation by price level:

| Price quartile | Predicted price range | 8-premium |
|---|---|---|
| Q1 (cheapest 25%) | Below ~$509K | **-$1,795** (discount!) |
| Q2 | $509K – $641K | +$1,482 |
| Q3 | $641K – $761K | +$1,955 |
| Q4 (most expensive 25%) | Above ~$761K | **+$2,166** |

Sellers pricing with trailing 8s at the low end actually leave money on the table. The premium only kicks in above ~$500K, and grows with price. Lucky 8s are valued by buyers who can afford to meet a psychologically curated price — not by budget-conscious comparison shoppers.

## Expert validation

- **Mohan Sandrasegeran** (Head of Research, PropertyGuru): "The approach is directionally sound... a useful framework for understanding broad pricing patterns."
- **Prof Sing Tien Foo** (NUS Real Estate): No major issues with the model. Pointed to the Management Science paper on superstition and housing in Singapore.
- **Nicholas Mak** (former ERA research head): "Some academic knowledge are not appreciated by property agents. But it will be appreciated by academic circles and by me."
- **Mark Hensen**: Recommended the LAD regression and hold-out validation that ultimately killed the weaker findings. His advice directly shaped the robustness checks.

## What the model explains (and doesn't)

Model 12 explains 93.7% of HDB resale price variation (95.4% in the log specification). The typical prediction is off by ~$27K (about 4-5% of a median flat's price).

What drives prices (in order of importance): town, remaining lease, floor area, storey, distance to CBD/MRT, flat model/type, distance to amenities and feng shui factors. The superstition premium (trailing 8s) is real but small — it adds only ~0.2% to the model's explanatory power. It's statistically significant because n = 50,718, but economically modest.

What the model can't see: renovation quality, views, facing, exact floor (only storey ranges), micro-neighbourhood effects within a town, and seller/buyer negotiation dynamics. That's most of the remaining ~6%.

## Methodology (for the notebook)

- 12 models built incrementally (Notebooks 6, 11, 14)
- 5 diagnostic checks (Notebook 12): Q-Q plots, Cook's distance, VIF, AIC/BIC, coefficient stability
- 3 robustness checks (Notebook 13): hold-out validation (test R² = 0.9029), L1/LAD regression, price-quartile interaction
- HC1 robust standard errors throughout
- All code at: github.com/[repo]

## Potential angles

1. **The lucky 8 piece** — update of PT's earlier reporting, now with full robustness checks showing only trailing 8s survive. The rich-buyer twist is new.
2. **What didn't survive** — a methodological transparency piece. "We thought we found four superstition effects. Three were wrong. Here's how we caught them." This is the Sarah Cohen / data journalism accountability angle.
3. **Lease decay varies by town** — the interaction finding. A year of lease in Bukit Timah is worth $25,500 more than in Yishun. This is a separate story about where lease decay hurts most.
4. **The outlier hunt** — use Model 12's residuals to find genuinely unusual transactions (Boon Tiong Road, Changi Village heritage flats, 3Gen bargain flats) and report on what's driving them.
