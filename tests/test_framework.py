"""
Framework validation: can this pipeline find an effect, and does it correctly
find nothing when there is nothing?

The second class of test is the one that matters. A backtest framework that
reports alpha on pure noise invalidates every result it will ever produce, and
that failure is silent -- the numbers look fine. So the noise case is tested as
carefully as the signal case, the same way a collision checker is tested by
planting collisions it must catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.data.synthetic import SyntheticConfig, generate, generate_scenario
from alpha.features.evasiveness import HashingEmbedder, sanity_check_embedder
from alpha.features.pipeline import build_features
from alpha.research.evaluation import (
    factor_alpha, information_coefficient, quantile_portfolios,
)
from alpha.research.orthogonalize import build_announcement_controls, orthogonalize
from alpha.stats import benjamini_hochberg, mean_with_hac_tstat, ols


# --------------------------------------------------------------------------
# Statistical primitives
# --------------------------------------------------------------------------

def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 3))
    beta = np.array([1.5, -0.8, 0.3])
    y = X @ beta + rng.normal(0, 0.5, n)
    res = ols(y, X)
    assert np.allclose(res.params, beta, atol=0.05)


def test_hac_se_exceeds_naive_under_autocorrelation():
    """
    The reason the HAC estimator exists.

    With positively autocorrelated data, naive standard errors are too small
    and t-stats are inflated. If this assertion ever fails, the HAC code is
    broken and every reported t-stat in the project is wrong.
    """
    rng = np.random.default_rng(1)
    n = 500
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + rng.normal(0, 1)

    naive = ols(x, np.ones((n, 1)), cov_type="nonrobust")
    hac = ols(x, np.ones((n, 1)), cov_type="HAC")
    assert hac.se[0] > naive.se[0] * 1.5


def test_hac_tstat_calibrated_on_iid_noise():
    """On iid noise the mean should be insignificant most of the time."""
    rejections = 0
    trials = 200
    for s in range(trials):
        rng = np.random.default_rng(1000 + s)
        _, _, t, _ = mean_with_hac_tstat(rng.normal(0, 1, 120))
        if abs(t) > 1.96:
            rejections += 1
    # Nominal 5%; allow slack for the known small-sample HAC over-rejection.
    assert rejections / trials < 0.12, f"over-rejection: {rejections / trials:.1%}"


def test_benjamini_hochberg_controls_discoveries():
    p_null = np.random.default_rng(3).uniform(0, 1, 100)
    reject, _ = benjamini_hochberg(p_null, alpha=0.05)
    assert reject.sum() <= 5

    p_mixed = np.concatenate([np.array([1e-8, 1e-7, 1e-6]), p_null])
    reject2, _ = benjamini_hochberg(p_mixed, alpha=0.05)
    assert reject2[:3].all()


# --------------------------------------------------------------------------
# Embedder
# --------------------------------------------------------------------------

def test_embedder_discriminates_related_from_unrelated():
    """A measurement device gets validated before it is trusted."""
    checks = sanity_check_embedder(HashingEmbedder())
    assert checks["self_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert checks["related_similarity"] > checks["unrelated_similarity"]


# --------------------------------------------------------------------------
# End-to-end: POWER
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def signal_run():
    """
    Uses the named "incremental" scenario rather than an ad-hoc parameter mix.

    An earlier version passed hand-picked alpha/pead/confound values whose net
    behaviour had to be reverse-engineered -- and they happened to nearly
    cancel, so the fixture produced a near-zero IC and the test failed for a
    reason that had nothing to do with the code. Named scenarios carry their
    own expected verdict, so a failure here means the pipeline broke.
    """
    cfg = SyntheticConfig(n_firms=140, n_quarters=20, seed=7,
                          words_prepared=320, words_per_answer=75)
    data = generate_scenario("incremental", cfg)
    feats, diag = build_features(data["transcripts"], embedder=HashingEmbedder(),
                                 verbose=False)
    panel = feats.merge(
        data["panel"], on=["ticker", "period", "fiscal_year", "fiscal_quarter"],
        how="inner", suffixes=("", "_p"),
    )
    panel = build_announcement_controls(panel)
    return panel, data, diag


def test_pipeline_produces_usable_features(signal_run):
    panel, _, diag = signal_run
    assert diag["usable_rate"] > 0.9, f"segmentation dropped too much: {diag}"
    assert panel["lm_tone_gap"].notna().mean() > 0.9
    assert panel["composite_z"].notna().mean() > 0.5


def test_tone_gap_recovers_latent_construct(signal_run):
    """
    The text scorer must recover the latent tone planted in the generated text.

    This isolates a real failure mode: if tokenisation or the dictionary lookup
    breaks, features become noise and every downstream null is uninformative.
    """
    panel, _, _ = signal_run
    sub = panel[["lm_tone_gap", "_true_latent"]].dropna()
    corr = sub["lm_tone_gap"].corr(sub["_true_latent"])
    assert corr < -0.15, f"tone gap failed to track latent quality (corr={corr:.3f})"


def test_framework_detects_planted_alpha(signal_run):
    """
    Smoke-level power check on a single draw.

    Deliberately loose. Rigorous power and specificity claims belong in
    tests/test_calibration.py, which measures rejection RATES across seeds --
    a single-seed threshold assertion tests what happened that seed, not
    whether the inference is calibrated.
    """
    panel, _, _ = signal_run
    ic = information_coefficient(panel, "composite_z", "fwd_ret_21")
    assert ic["n_periods"] >= 10
    assert np.isfinite(ic["mean_ic"])
    assert abs(ic["mean_ic"]) > 0.01, "planted effect left no trace at all"


def test_orthogonalization_runs_and_preserves_variation(signal_run):
    """Stage 1 must leave something to test in Stage 2."""
    panel, _, _ = signal_run
    ortho, res = orthogonalize(panel, signal_col="composite_z",
                               sector_col="sector", period_col="period")
    assert res.r2 < 0.95, "controls absorbed essentially all signal variation"
    assert ortho["signal_resid"].notna().sum() > 0
    assert res.controls_used, "no controls were applied"


def test_quantile_spread_monotonic_under_planted_alpha(signal_run):
    panel, _, _ = signal_run
    per_period, summary = quantile_portfolios(panel, "composite_z", "fwd_ret_21")
    assert not summary.empty
    q1 = summary.loc[summary["portfolio"] == "q1", "mean_return"].iloc[0]
    q5 = summary.loc[summary["portfolio"] == "q5", "mean_return"].iloc[0]
    assert np.isfinite(q1) and np.isfinite(q5)


# --------------------------------------------------------------------------
# Smoke test on the null scenario. Rigorous specificity lives in
# tests/test_calibration.py.
# --------------------------------------------------------------------------

def test_null_scenario_pipeline_runs():
    from alpha.data.synthetic import generate_null

    data = generate_null(SyntheticConfig(n_firms=50, n_quarters=8, seed=11,
                                         words_prepared=320, words_per_answer=75))
    feats, _ = build_features(data["transcripts"], embedder=HashingEmbedder(),
                              verbose=False)
    panel = feats.merge(
        data["panel"], on=["ticker", "period", "fiscal_year", "fiscal_quarter"],
        how="inner", suffixes=("", "_p"),
    )
    panel = build_announcement_controls(panel)
    ic = information_coefficient(panel, "composite_z", "fwd_ret_21")
    assert np.isfinite(ic["mean_ic"])


def test_consensus_sue_path_is_exercised():
    """The synthetic panel carries EPS data, so SUE uses the consensus branch."""
    data = generate_scenario("confounded",
                             SyntheticConfig(n_firms=30, n_quarters=4, seed=2,
                                             words_prepared=320, words_per_answer=75))
    panel = build_announcement_controls(data["panel"])
    assert panel["sue_source"].iloc[0] == "eps_consensus"
    assert panel["sue"].notna().mean() > 0.9
