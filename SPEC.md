# Pre-Registered Specification

**Commit this file before running the backtest.** Git history is then evidence
of the ordering — that the primary specification was fixed in advance rather
than selected after seeing which variant looked best.

Status: `FRAMEWORK CALIBRATED — NOT YET RUN ON REAL DATA`

---

## 1. Primary hypothesis

The divergence between scripted (prepared remarks) and spontaneous (Q&A)
management language carries information about future cross-sectional equity
returns that is **not** subsumed by the earnings surprise.

Formally, with `resid` the text signal after Stage-1 controls:

    H0:  E[ rank_IC(resid, forward_return_21d) ] = 0
    H1:  E[ rank_IC(resid, forward_return_21d) ] > 0

## 2. Primary specification — fixed in advance

| Choice | Value | Why this and not another |
|---|---|---|
| Signal | equal-weight z-score composite | zero free parameters; nothing to overfit |
| Horizon | 21 trading days | underreaction horizon in the Lazy Prices literature |
| Universe | point-in-time S&P 500 constituents | survivor-only universes inflate results |
| Ranking | sector-neutral cross-sectional rank | tech and utilities genuinely talk differently |
| Portfolio | long/short top-minus-bottom quintile, equal-weighted | |
| Costs | 10 bp per side headline; 0/5/10/20 all reported | |
| Inference | rank IC, HAC standard errors, t critical values | overlapping horizons induce autocorrelation |
| Embedder | `all-MiniLM-L6-v2`, held fixed across the sample | choosing by backtest performance is look-ahead |
| Sample start | **chosen from the coverage report, recorded here before results** | |

## 3. Component signs — declared before any return is examined

| Component | Sign | Reasoning |
|---|---|---|
| `lm_tone_gap` | − | optimism in the script that cannot be sustained under questioning |
| `evasiveness` | − | answers drifting from questions asked |
| `hedging_density` | − | hedging under live pressure, not scripted boilerplate |
| `analyst_tone` | + | outside information management does not control |
| `semantic_novelty` | 0 | direction ambiguous a priori; tested separately, excluded from the composite |

Any post-hoc sign change must be recorded in section 7 with its date and reason.

## 4. Controls (Stage 1)

`sue`, `announcement_return`, `log_mktcap`, `book_to_market`, `momentum_12_1`,
sector dummies, period fixed effects.

Where analyst consensus is unavailable, SUE falls back to a standardised
announcement-window return. The substitution is recorded per row in
`sue_source` and reported in the results, not buried in a footnote.

## 5. Multiple testing

Every specification run is written to `results/all_variants.csv` — not only
those that worked. Benjamini–Hochberg at α = 0.05 across the family. The primary
specification above is reported regardless of outcome.

## 6. Pre-committed interpretation

| Result | What is reported |
|---|---|
| raw ✓, residual ✓ | genuine incremental information — headline finding |
| raw ✓, residual ✗ | the signal is a repackaging of the earnings surprise |
| neither | null result; contribution is the framework and an honest negative |

A null is a complete, publishable-in-spirit outcome. The framework's calibration
(power 100% on the `incremental` scenario) is what makes a null interpretable
rather than merely uninformative.

## 7. Amendment log

| Date | Change | Reason |
|---|---|---|
| — | initial specification | — |
