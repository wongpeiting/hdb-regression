# When to Use Regression (and When Not To)

A quick-reference tipsheet for data journalists.

---

## 1. Linear Regression: The 6-Notebook Pipeline

**Source:** Dhrumil Mehta's EDA-with-regression pipeline, Columbia J-School ([class.data4news.com](https://class.data4news.com))

**The 6 notebooks:**

1. Download your data
2. Explore distributions (histograms, summary stats, spot problems early)
3. Download supplementary/census data (the variables you want to control for)
4. Merge your datasets together
5. Single-variable regressions (test each X against Y, one at a time)
6. Multivariable regression (put everything into one model)

### When to use it

Use linear regression when **Y is continuous** -- a number like a price, score, amount, or rate. You are asking: **"What predicts how much / how many?"**

### Singapore story ideas

| Story question | Y variable |
|---|---|
| What predicts HDB resale prices? | price ($) |
| What predicts government tender overpayment? | bid_premium (% above estimate) |
| What predicts school performance? | PSLE aggregate score |
| What predicts COE prices? | COE premium ($) |
| What predicts ministerial salary? | salary amount ($) |

### Two kinds of stories come out of this

**Model stories (the coefficients).** The regression tells you the size of each effect, holding everything else constant. This is the bread and butter:

- "Every additional floor adds $X per sqm to HDB resale prices -- even after controlling for town, flat type, remaining lease, and proximity to MRT."
- "Tenders where only one company bid had a premium X percentage points higher than competitive tenders."

The coefficient gives you a number you can put in a headline.

**Error stories (the residuals).** Build a model of what's normal, then investigate what doesn't fit. This is the Dallas Morning News approach -- they built a model of expected school test scores based on demographics, then investigated schools that performed far worse than predicted.

- "These 10 HDB flats sold for far more than the model predicts -- why?"
- "This government agency consistently overpays on IT contracts compared to what the model expects."
- "These schools perform well below what their student demographics would predict."

The residuals give you a list of leads to report out.

---

## 2. Logistic Regression: Binary Outcomes

**Notebooks:** `/Users/wongpeiting/logistic-regression/`

These follow a similar EDA-to-regression structure, but for a different kind of Y.

### When to use it

Use logistic regression when **Y is binary** -- yes or no, 0 or 1, win or lose. You are asking: **"What predicts whether something happens?"**

### Singapore story ideas

| Story question | Y variable |
|---|---|
| What predicts whether a company wins a government tender? | won_bid (0/1) |
| What predicts whether an HDB flat sells above $1M? | million_dollar (0/1) |
| What predicts whether a student passes? | pass/fail (0/1) |
| What predicts whether a food stall gets a hygiene violation? | violated (0/1) |
| What predicts whether a company has government connections? | has_gov_connection (0/1) |

### Odds ratios in plain English

Logistic regression gives you **odds ratios** instead of simple coefficients. Here is how to read them:

- An odds ratio of **2.0** means: "For every one-unit increase in X, the odds of Y happening double."
- An odds ratio of **0.5** means: "For every one-unit increase in X, the odds of Y happening are cut in half."
- An odds ratio of **1.0** means: X has no effect.

In practice, you might write: "Companies that previously held government contracts had 3.2 times the odds of winning the next tender, even after controlling for bid price and company size."

The interpretation is always: **"For every X more, the odds multiply by Y."**

---

## 3. Stories That Don't Need Regression

Not everything needs a model. If you can answer the question with a simpler method, use the simpler method. Here is when to skip regression entirely.

**Pivot tables / group-by.** When you just need to compare averages across categories. "What is the median HDB resale price by town?" Group, summarise, make a bar chart. If the finding is obvious from the chart, you are done.

**Time series / trend lines.** When the story is about change over time. "How have COE prices changed since 2015?" Line charts and percent changes do the job.

**Counting and ranking.** "Which ministry spends the most on social media advertising?" Just sort and rank. No model needed.

**Mapping.** When geography IS the story and you do not need to control for anything else. "Where are the dengue clusters?" Put dots on a map.

**Simple correlation.** When you have one X and one Y, and the scatter plot tells the whole story. "Do constituencies with more elderly residents have higher voter turnout?" If there is only one variable, you may not need the full pipeline.

### The key test

> **Do you need to control for confounders?**
>
> If the answer is "no, the simple comparison is fair" -- you don't need regression. Use pivot tables, charts, or maps.
>
> If the answer is "yes, but maybe the effect disappears when you account for other factors" -- you need regression.

Example: "HDB flats near MRT stations sell for more." True, but is that just because MRT stations tend to be in mature estates with better amenities? To answer that, you need to control for town, flat age, flat type, and remaining lease. That is when you reach for regression.

---

## Decision Flowchart

```
START: What kind of question are you asking?
  |
  |-- Is your Y variable continuous (price, score, amount, rate)?
  |     --> Linear regression (6-notebook pipeline)
  |
  |-- Is your Y variable binary (yes/no, win/lose, pass/fail)?
  |     --> Logistic regression
  |
  |-- Is your Y variable a count (number of violations, complaints)?
  |     --> If counts are large (dozens+), linear regression can work
  |     --> If counts are small (0-10 range), consider Poisson regression
  |
  |-- Do you NOT need to control for confounders?
  |     --> Simple EDA: pivot tables, charts, maps, rankings
  |
  |-- Are you looking for outliers or anomalies?
        --> Build a regression model of what's "normal,"
            then investigate the residuals (the cases that
            don't fit the pattern)
```

---

## One-Paragraph Summary

If Y is a number, use linear regression. If Y is yes/no, use logistic regression. If you don't need to control for confounders, skip regression and use simpler tools. And remember: regression produces two stories, not one -- the model tells you what matters, and the residuals tell you what's weird.
