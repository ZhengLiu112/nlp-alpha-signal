"""
Synthetic transcript and return generator with a KNOWN, tunable alpha.

Why this exists, and why it is not a toy:

A backtest framework is a measurement device, and an unvalidated measurement
device produces numbers of unknown meaning. Before running on real data, this
generator answers two questions that no amount of staring at code will:

  1. POWER      -- with a planted effect of known size, does the pipeline
                   recover it? If not, a null result on real data means
                   nothing, because the framework could not have found an
                   effect even if one existed.

  2. SPECIFICITY -- with the effect set to zero, does the pipeline correctly
                   report nothing? If it finds "alpha" in pure noise, every
                   real result is suspect.

The second test is the important one and the one usually skipped. Same pattern
as planting conflicts in a collision checker to confirm the checker fires: a
test that can only pass is not a test.

The generator also plants a CONFOUND: part of the text signal is driven by the
earnings surprise, which itself drives returns. That lets the orthogonalization
protocol be tested on data where the right answer is known -- Stage 2 should
find the planted incremental effect and nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_POS_WORDS = ["strong", "growth", "improved", "record", "robust", "momentum",
              "confident", "pleased", "exceeded", "favorable", "solid", "opportunity"]
_NEG_WORDS = ["decline", "weak", "headwinds", "pressure", "challenging", "shortfall",
              "difficult", "delay", "concern", "miss", "adverse", "risk"]
_NEUTRAL = ["quarter", "revenue", "segment", "customers", "market", "business",
            "operations", "products", "region", "team", "results", "performance",
            "investment", "capacity", "demand", "pricing", "volume", "margin"]
_HEDGE = ["may", "might", "could", "approximately", "roughly", "somewhat",
          "generally", "typically", "uncertain", "depends"]

SECTORS = ["Technology", "Health Care", "Financials", "Industrials",
           "Consumer Discretionary", "Energy"]


@dataclass
class SyntheticConfig:
    n_firms: int = 120
    n_quarters: int = 24
    start_year: int = 2018

    # Planted effect sizes, in units of forward-return standard deviation per
    # unit of standardised signal.
    alpha_strength: float = 0.05      # TRUE incremental effect of tone gap
    pead_strength: float = 0.10       # effect of the earnings surprise itself
    confound_strength: float = 0.60   # how much the surprise contaminates the text

    return_vol: float = 0.12
    seed: int = 0

    # Text realism knobs
    words_prepared: int = 700
    words_per_answer: int = 110
    n_qa_pairs: int = 6


def _make_paragraph(rng: np.random.Generator, n_words: int, tone: float,
                    hedge_rate: float = 0.0) -> str:
    """
    Emit text whose dictionary tone tracks the requested latent tone.

    ``tone`` in [-1, 1] shifts the positive/negative word mix. The point is
    that the pipeline must RECOVER the latent tone from text rather than being
    handed it, so the full tokenise-score path is exercised.
    """
    p_sent = 0.16
    p_pos = p_sent * (0.5 + 0.5 * np.clip(tone, -1, 1))
    p_neg = p_sent - p_pos

    words = []
    for _ in range(n_words):
        u = rng.random()
        if u < p_pos:
            words.append(rng.choice(_POS_WORDS))
        elif u < p_pos + p_neg:
            words.append(rng.choice(_NEG_WORDS))
        elif u < p_pos + p_neg + hedge_rate:
            words.append(rng.choice(_HEDGE))
        else:
            words.append(rng.choice(_NEUTRAL))
    return " ".join(words).capitalize() + "."


def generate(config: SyntheticConfig | None = None) -> dict[str, pd.DataFrame]:
    """
    Build a synthetic panel with a known data-generating process.

    The DGP, stated explicitly so tests can assert against it:

        surprise            ~ N(0, 1)
        latent_quality      ~ N(0, 1)                     [the TRUE unobserved signal]

        tone_prepared       = 0.5*surprise + 0.3*latent_quality + noise
        tone_answers        = 0.5*surprise + 0.9*latent_quality + noise
        => tone_gap         mostly reflects -latent_quality, plus surprise noise

        forward_return      = pead_strength * surprise
                            + alpha_strength * latent_quality
                            + N(0, return_vol)

    So: the raw tone gap predicts returns partly through the surprise
    (confound) and partly through latent quality (real). After orthogonalising
    on the surprise, only the ``alpha_strength`` channel should remain.

    Returns dict with keys: transcripts, panel, factors, truth.
    """
    cfg = config or SyntheticConfig()

    # Two INDEPENDENT streams, and this is not a stylistic choice.
    #
    # An earlier version drew text and returns from a single generator. That
    # produced a spurious positive IC on data where returns were pure noise --
    # the specificity test caught it. Cause: np.random.Generator.choice uses
    # rejection sampling, so the number of raw random words it consumes depends
    # on the length of the list being sampled. Word lists here have different
    # lengths, so the generator's stream POSITION after writing a paragraph
    # depended on the paragraph's content, and the return drawn next inherited
    # that dependence. A deterministic leak from text into returns, invisible
    # by inspection and fatal to every number downstream.
    #
    # Independent streams make the leak structurally impossible rather than
    # merely unlikely.
    seeds = np.random.SeedSequence(cfg.seed).spawn(2)
    rng = np.random.default_rng(seeds[0])       # latent variables and returns
    trng = np.random.default_rng(seeds[1])      # text only

    tickers = [f"SYN{i:03d}" for i in range(cfg.n_firms)]
    sectors = {t: SECTORS[i % len(SECTORS)] for i, t in enumerate(tickers)}

    quarters = []
    y, q = cfg.start_year, 1
    for _ in range(cfg.n_quarters):
        quarters.append((y, q))
        q += 1
        if q > 4:
            q, y = 1, y + 1

    transcripts, rows = [], []

    for (fy, fq) in quarters:
        period = f"{fy}Q{fq}"
        month = {1: "02", 2: "05", 3: "08", 4: "11"}[fq]

        surprise = rng.normal(0, 1, cfg.n_firms)
        latent = rng.normal(0, 1, cfg.n_firms)

        for i, ticker in enumerate(tickers):
            s, lq = surprise[i], latent[i]

            # The confound loads ASYMMETRICALLY on the two sections. This is
            # deliberate: if it loaded equally it would cancel in the gap, and
            # the orthogonalization step would have nothing to remove -- the
            # test would pass without ever exercising the code it claims to
            # test. Management spins the script harder than it can spin live
            # answers, so the prepared section absorbs more of the surprise.
            tone_prep = np.tanh(0.90 * cfg.confound_strength * s + 0.30 * lq
                                + rng.normal(0, 0.35))
            tone_ans = np.tanh(0.25 * cfg.confound_strength * s + 0.90 * lq
                               + rng.normal(0, 0.35))
            hedge = float(np.clip(0.05 - 0.03 * lq + rng.normal(0, 0.01), 0.0, 0.25))

            day = int(rng.integers(3, 26))
            call_date = f"{fy}-{month}-{day:02d}"

            turns = [{"speaker": "Operator", "role": "operator",
                      "text": "Good day and welcome to the earnings conference call."}]
            turns.append({
                "speaker": "Jane Doe", "role": "Chief Executive Officer",
                "text": _make_paragraph(trng, cfg.words_prepared // 2, tone_prep),
            })
            turns.append({
                "speaker": "John Roe", "role": "Chief Financial Officer",
                "text": _make_paragraph(trng, cfg.words_prepared // 2, tone_prep),
            })
            turns.append({
                "speaker": "Operator", "role": "operator",
                "text": ("Thank you. We will now begin the question-and-answer session. "
                         "Our first question comes from the line of an analyst."),
            })

            for k in range(cfg.n_qa_pairs):
                turns.append({
                    "speaker": f"Analyst {k}", "role": "analyst",
                    "text": _make_paragraph(trng, 45, -0.2 * lq),
                })
                turns.append({
                    "speaker": "Jane Doe" if k % 2 == 0 else "John Roe",
                    "role": "Chief Executive Officer" if k % 2 == 0 else "Chief Financial Officer",
                    "text": _make_paragraph(trng, cfg.words_per_answer, tone_ans, hedge),
                })

            transcripts.append({
                "ticker": ticker, "call_date": call_date,
                "fiscal_year": fy, "fiscal_quarter": fq, "turns": turns,
            })

            fwd = (cfg.pead_strength * s
                   + cfg.alpha_strength * lq
                   + rng.normal(0, cfg.return_vol))

            # Analyst consensus and the reported number. Providing these
            # exercises the eps_consensus branch of build_announcement_controls
            # -- the "you have good data" case, where SUE is a near-exact
            # measure of the surprise rather than a noisy return proxy.
            #
            # Both paths matter. With consensus data the confound is almost
            # fully removable; with only the return proxy it is not, and the
            # residual retains part of the surprise. That difference is a real
            # property of the study, not an artefact, and the synthetic harness
            # reproduces both so the consequence is visible before it shows up
            # in a real result.
            eps_est = rng.normal(1.0, 0.25)
            eps_act = eps_est + 0.15 * s * abs(eps_est)

            rows.append({
                "ticker": ticker, "period": period, "call_date": call_date,
                "fiscal_year": fy, "fiscal_quarter": fq, "sector": sectors[ticker],
                "announcement_return": 0.035 * s + rng.normal(0, 0.012),
                "eps_actual": eps_act,
                "eps_estimate": eps_est,
                "log_mktcap": rng.normal(23, 1.2),
                "book_to_market": abs(rng.normal(0.45, 0.22)),
                "momentum_12_1": rng.normal(0.06, 0.20),
                "fwd_ret_5": 0.35 * fwd + rng.normal(0, cfg.return_vol * 0.5),
                "fwd_ret_21": fwd,
                "fwd_ret_63": 1.1 * fwd + rng.normal(0, cfg.return_vol * 0.6),
                "_true_surprise": s,
                "_true_latent": lq,
            })

    panel = pd.DataFrame(rows)

    periods = sorted(panel["period"].unique())
    frng = np.random.default_rng(cfg.seed + 99)
    factors = pd.DataFrame({
        "mkt_rf": frng.normal(0.015, 0.06, len(periods)),
        "smb": frng.normal(0.002, 0.03, len(periods)),
        "hml": frng.normal(0.001, 0.03, len(periods)),
        "rmw": frng.normal(0.002, 0.02, len(periods)),
        "cma": frng.normal(0.001, 0.02, len(periods)),
        "mom": frng.normal(0.004, 0.04, len(periods)),
    }, index=periods)

    truth = {
        "alpha_strength": cfg.alpha_strength,
        "pead_strength": cfg.pead_strength,
        "confound_strength": cfg.confound_strength,
        "n_firms": cfg.n_firms,
        "n_quarters": cfg.n_quarters,
        "note": ("tone_gap should predict returns raw (via surprise confound + latent) "
                 "and, if alpha_strength > 0, should retain predictive power after "
                 "orthogonalising on the surprise."),
    }

    return {"transcripts": transcripts, "panel": panel,
            "factors": factors, "truth": truth}


# Three presets, one for each outcome the orthogonalization protocol exists to
# distinguish. Running all three is how the framework earns the right to be
# pointed at real data: it must produce the correct verdict in every case, not
# just find something when something is there.
SCENARIOS: dict[str, dict] = {
    # Genuine incremental information: survives the controls.
    "incremental": {"alpha_strength": 0.15, "pead_strength": 0.10,
                    "confound_strength": 0.40},
    # The signal is nothing but the earnings surprise wearing an NLP costume.
    # Raw looks great; residual must collapse. This is the failure mode the
    # whole protocol is built to catch, and the one most likely to be true in
    # practice.
    "confounded": {"alpha_strength": 0.00, "pead_strength": 0.15,
                   "confound_strength": 0.80},
    # Pure noise. Anything significant here is a bug.
    "null": {"alpha_strength": 0.00, "pead_strength": 0.00,
             "confound_strength": 0.00},
}


def generate_scenario(name: str, config: SyntheticConfig | None = None) -> dict:
    """
    Build one of the named validation scenarios.

    Expected verdicts:

        incremental  raw significant, residual significant
        confounded   raw significant, residual NOT significant
        null         neither significant

    A framework that cannot reproduce all three verdicts cannot be trusted to
    tell them apart on real data, where the answer is unknown.
    """
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    base = config or SyntheticConfig()
    cfg = SyntheticConfig(**{**base.__dict__, **SCENARIOS[name]})
    out = generate(cfg)
    out["truth"]["scenario"] = name
    out["truth"]["expected_raw_significant"] = name in ("incremental", "confounded")
    out["truth"]["expected_residual_significant"] = name == "incremental"
    return out


def generate_null(config: SyntheticConfig | None = None) -> dict:
    """
    The specificity test: identical machinery, zero planted effect.

    If the pipeline reports a significant result here it has a bug, or the
    inference is wrong, and every result it produces on real data is
    untrustworthy. This is the test that is usually missing.
    """
    return generate_scenario("null", config)
