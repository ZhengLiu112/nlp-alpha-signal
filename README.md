# nlp-alpha-signal

Does the divergence between **scripted** and **spontaneous** management language
in earnings calls predict cross-sectional equity returns — *beyond* what is
already contained in the earnings surprise itself?

Prepared remarks are lawyered, rehearsed, and fully controlled by management.
Q&A answers are produced live, under analyst pressure. The gap between how
management speaks in the two sections is the object of study here, rather than
sentiment level, which is both well covered and hard to distinguish from
post-earnings-announcement drift.

---

## Status

The measurement framework is **built and calibrated**: 41/41 tests pass,
including the four multi-seed calibration tests. It has not yet been run on real
transcripts.

That ordering is deliberate. A backtest framework is a measurement device, and
an uncalibrated device produces numbers of unknown meaning. Before touching real
data, this pipeline is required to return the correct verdict on three synthetic
scenarios whose answers are known in advance.

| Scenario | Planted truth | Raw signal rejects | After orthogonalization | Verdict |
|---|---|---|---|---|
| `null` | nothing at all | 0% | **0%** | calibrated — no alpha manufactured from noise |
| `confounded` | signal is *purely* the earnings surprise | 100% | **0%** | the confound is fully stripped |
| `incremental` | genuine effect beyond the surprise | 100% | **100%** | adequate power |

*10 seeds each, 60 firms × 24 quarters. Nominal test size 5%.*
Reproduce with `make calibrate`.

The `confounded` row is the one that matters most. It is the failure this whole
project is designed to catch: a text signal that looks impressive and is nothing
but PEAD wearing an NLP costume. The protocol catches it.

---

## Two bugs the calibration harness caught

Both were invisible by inspection and both would have corrupted every number
downstream. They are recorded here rather than quietly fixed, because how a
framework fails is more informative than that it currently passes.

### 1. RNG stream coupling — spurious alpha from pure noise

On data where returns were **pure noise**, the pipeline reported a systematically
positive information coefficient: mean t of **+0.97** across seeds, where zero
was the correct answer.

Cause: `numpy.random.Generator.choice` uses rejection sampling, so the number of
raw random words it consumes depends on the length of the list being sampled.
The word lists used to synthesise transcripts have different lengths. The
generator's stream *position* after writing a paragraph therefore depended on
that paragraph's **content**, and the return drawn next inherited the
dependence — a deterministic leak from text into returns.

Fixed by drawing text and returns from independent `SeedSequence` children,
which makes the leak structurally impossible rather than merely unlikely. Mean t
moved to **−0.28**, with zero false positives across 8 seeds. Pinned by
`test_text_and_returns_use_independent_rng_streams`.

### 2. HAC over-rejection on short samples

The null scenario rejected at **25%** against a nominal 5% on a 10-quarter
sample. This one was not a coding error — it is a real property of asymptotic
HAC inference in small samples, and it would have inflated every borderline
t-statistic in the study.

Addressed three ways:
- finite-sample scaling `n/(n−k)` applied to the HAC covariance by default
- Student-*t* critical values with `n−1` df instead of normal
- a `small_sample_warning` flag attached to every IC result below 20 periods

After the corrections, null rejection sits at 0–5%.

---

## The orthogonalization protocol

The obvious objection to any earnings-text signal is that it is a noisy proxy
for the earnings surprise, and the surprise is already known to predict drift
(Bernard & Thomas, 1989). Addressed structurally rather than argued away:

```
Stage 1   text_signal ~ SUE + announcement_return + log_mktcap
                        + book_to_market + momentum_12_1
                        + sector dummies + period fixed effects
          residual = text_signal − fitted

Stage 2   forward_return ~ residual
```

Raw and residual results are always reported side by side. Three outcomes, all
of which are reportable:

| Outcome | Interpretation |
|---|---|
| raw ✓, residual ✓ | genuine incremental information |
| raw ✓, residual ✗ | the signal **is** the earnings surprise. Report it as such. |
| neither | null result. The contribution is the framework and an honest negative. |

Note from the synthetic study: with only a return-based SUE proxy the residual
retains part of the confound; with real consensus estimates it collapses to
zero. If consensus data is unavailable, that limitation is real and travels with
every row via the `sue_source` column.

---

## Signal components

Five, computed per firm-quarter. Signs are declared in
`alpha/features/composite.py` **before** any return is examined — flipping a
sign after seeing results is the most common unrecorded researcher degree of
freedom in this kind of study.

| Component | Construct | Prior |
|---|---|---|
| **Tone gap** | `sentiment(prepared) − sentiment(Q&A answers)` | − |
| **Evasiveness** | `1 − cos(embed(question), embed(answer))` | − |
| **Hedging density** | LM uncertainty + weak-modal rate, on answers only | − |
| **Analyst pressure** | tone of analyst *questions* (outside management's control) | + |
| **Linguistic novelty** | distance from the same firm's prior-quarter remarks | ambiguous, tested separately |

Composite is an equal-weight z-score. A supervised composite would almost
certainly backtest better, and that is exactly the problem — with five
components, three horizons and two tone models there is enough freedom to fit
noise, and no reader can tell from outside. Zero free parameters means nothing
to overfit.

---

## Quick start

```bash
pip install -r requirements.txt

make test          # fast suite (~1 min)
make calibrate     # framework validation (~15 min) — run before trusting anything
make demo          # end-to-end study on synthetic data
```

Real data:

```bash
python scripts/01_ingest.py --limit 500      # inspect schema + coverage FIRST
python scripts/run_study.py --data data/transcripts.jsonl
```

---

## Before reporting any real-data result

- [ ] **Licence checked** on the Hugging Face dataset. If unclear, publish
      derived features only — never raw transcript text.
- [ ] **~20 transcripts hand-inspected** via `inspect_sample`. Segmentation bugs
      do not raise; they move text between the prepared and Q&A pools and invert
      the tone gap.
- [ ] **Full Loughran–McDonald dictionary** downloaded to
      `data/lm_dictionary.csv`. The bundled seed lists are for testing; the
      pipeline warns loudly when it falls back to them.
- [ ] **Point-in-time universe** reconstructed. A dataset labelled "S&P 500
      companies" is a recent snapshot, so backtesting on it silently restricts
      the sample to survivors. Run both universes and **report the gap**.
- [ ] **Backtest start date chosen from the coverage report**, and written into
      `SPEC.md` before results are produced.
- [ ] **Entry timing** confirmed conservative — calls are typically after the
      close, and transcripts lag the call. Sensitivity at T+1 close and T+2 open.

---

## Known limitations

- **yfinance delisting gap.** Free price data serves currently-listed tickers,
  so firms delisted mid-sample lose their history. This partially undermines the
  point-in-time universe work and is not fully fixable without paid data.
- **TF-IDF fitted on the full corpus** in the lexical-novelty component, so IDF
  weights use future documents. The effect on a cosine distance between two
  documents of the same firm is second-order, but it is a genuine look-ahead
  channel. A strict version would refit on an expanding window.
- **SUE without consensus estimates** falls back to an announcement-return
  proxy, which is weaker. Recorded per row in `sue_source`.
- **The synthetic harness validates the inference, not the features.** It can
  prove the statistics are calibrated; it cannot prove that tone gap measures
  what it claims to on real transcripts. That is what hand-inspection is for.

---

## Layout

```
alpha/
├── stats.py                    OLS, Newey-West HAC (hand-rolled), BH correction
├── data/
│   ├── synthetic.py            three known-answer scenarios
│   └── ingest_transcripts.py   HF loader, schema normaliser, coverage report
├── features/
│   ├── segment.py              speaker attribution, prepared/Q&A split, Q-A pairing
│   ├── tone.py                 Loughran-McDonald + optional FinBERT
│   ├── evasiveness.py          question-answer semantic alignment
│   ├── novelty.py              quarter-over-quarter linguistic change
│   ├── composite.py            equal-weight z-score, signs pre-declared
│   └── pipeline.py             orchestration + rejection accounting
└── research/
    ├── orthogonalize.py        Stage 1 / Stage 2
    ├── evaluation.py           IC, decay, quantiles, FF alpha, cost sensitivity
    ├── calibration.py          multi-seed power and specificity study
    └── multiple_testing.py     BH across the specification family
```

---

## References

- Loughran & McDonald (2011), *JF* — finance-specific sentiment dictionaries
- Cohen, Malloy & Nguyen (2020), *JPE* — "Lazy Prices"; language *changes* predict returns
- Hassan, Hollander, van Lent & Tahoun (2019), *QJE* — firm risk from earnings calls
- Ke, Kelly & Xiu (2019), NBER — supervised sentiment extraction
- Bernard & Thomas (1989) — post-earnings-announcement drift, the null to beat
- Newey & West (1987) — the HAC estimator implemented in `alpha/stats.py`
