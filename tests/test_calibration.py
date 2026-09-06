"""
Framework calibration tests.

These replace the single-seed threshold checks that an earlier version of this
suite used. Asserting ``t_stat > 2`` on one synthetic draw tests what happened
on that draw, not whether the inference is calibrated -- and it fails or passes
for reasons unrelated to the code being correct.

What is asserted here instead is the rejection RATE across seeds, which is the
quantity that has a correct value.

These are slow by nature. Mark them and skip in a fast loop:
    pytest -m "not slow"
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha.data.synthetic import SCENARIOS, SyntheticConfig, generate_scenario
from alpha.research.calibration import CALIBRATION_CONFIG, calibrate_scenario

# Kept small on purpose: calibration needs seeds far more than it needs firms.
# Fewer firms to keep runtime down, but the SAME 24 quarters as
# CALIBRATION_CONFIG. Trading quarters away is not a valid speed-up here: at 10
# quarters this project's own calibration measured HAC rejecting at 25% against
# a nominal 5%, so a shorter run tests the estimator's small-sample failure
# rather than the pipeline. Cut firms, never periods.
FAST_CONFIG = SyntheticConfig(
    n_firms=40, n_quarters=24, words_prepared=320,
    words_per_answer=75, n_qa_pairs=4,
)
N_SEEDS = 8


def test_scenarios_declare_their_expected_verdicts():
    """Every scenario states its own right answer, so nothing is graded after the fact."""
    for name in SCENARIOS:
        truth = generate_scenario(name, SyntheticConfig(n_firms=4, n_quarters=2))["truth"]
        assert "expected_raw_significant" in truth
        assert "expected_residual_significant" in truth
    inc = generate_scenario("incremental", SyntheticConfig(n_firms=4, n_quarters=2))["truth"]
    conf = generate_scenario("confounded", SyntheticConfig(n_firms=4, n_quarters=2))["truth"]
    null = generate_scenario("null", SyntheticConfig(n_firms=4, n_quarters=2))["truth"]
    assert inc["expected_residual_significant"] is True
    assert conf["expected_raw_significant"] is True
    assert conf["expected_residual_significant"] is False
    assert null["expected_raw_significant"] is False


def test_text_and_returns_use_independent_rng_streams():
    """
    Regression test for a real bug this suite caught.

    Text and returns were once drawn from one generator. Because
    Generator.choice uses rejection sampling, the number of raw random words it
    consumes depends on the length of the list sampled -- so the stream
    position after writing a paragraph depended on the paragraph's CONTENT, and
    the return drawn next inherited that dependence. The result was a
    systematically positive IC on data where returns were pure noise
    (mean t across seeds +0.97 instead of ~0).

    Pinned here: with the effect switched off, returns must be independent of
    everything the text was built from.
    """
    cfg = SyntheticConfig(n_firms=60, n_quarters=6, seed=3,
                          words_prepared=320, words_per_answer=75)
    data = generate_scenario("null", cfg)
    panel = data["panel"]
    for col in ("_true_latent", "_true_surprise"):
        corr = panel[col].corr(panel["fwd_ret_21"])
        assert abs(corr) < 0.10, (
            f"{col} correlates {corr:.3f} with returns in the null scenario; "
            "the RNG streams are coupled again"
        )


@pytest.mark.slow
def test_specificity_no_alpha_manufactured_from_noise():
    """
    The most important test here.

    A framework that finds alpha in noise invalidates every result it will ever
    produce, and it fails silently -- the numbers look perfectly reasonable.
    """
    cal = calibrate_scenario("null", n_seeds=N_SEEDS, config=FAST_CONFIG, verbose=False)
    ok, msg = cal.verdict()
    assert ok, msg
    assert abs(cal.mean_residual_t) < 1.0, (
        f"mean residual t across seeds is {cal.mean_residual_t:+.2f}, "
        "should sit near zero under the null"
    )


@pytest.mark.slow
def test_orthogonalization_removes_a_pure_confound():
    """
    The protocol's whole purpose.

    In the confounded scenario the text signal carries NOTHING but the earnings
    surprise. Raw must look strong and the residual must collapse. If it does
    not, the protocol cannot tell a repackaged PEAD from a real finding, which
    is the single failure this project is designed to avoid.
    """
    cal = calibrate_scenario("confounded", n_seeds=N_SEEDS, config=FAST_CONFIG,
                             verbose=False)
    ok, msg = cal.verdict()
    assert ok, msg
    assert cal.raw_rejection_rate > cal.residual_rejection_rate, (
        "orthogonalization did not attenuate the confound"
    )


@pytest.mark.slow
def test_power_framework_finds_a_real_incremental_effect():
    """
    POWER. Without this, a null result on real data means nothing -- it cannot
    be distinguished from a pipeline incapable of detecting anything.
    """
    cal = calibrate_scenario("incremental", n_seeds=N_SEEDS, config=FAST_CONFIG,
                             verbose=False)
    ok, msg = cal.verdict()
    assert ok, msg
    assert cal.mean_residual_ic > 0, "true incremental alpha lost in orthogonalization"


@pytest.mark.slow
def test_small_sample_warning_is_raised():
    """Short samples must be flagged, since HAC over-rejects there."""
    from alpha.features.evasiveness import HashingEmbedder
    from alpha.features.pipeline import build_features
    from alpha.research.evaluation import information_coefficient

    cfg = SyntheticConfig(n_firms=50, n_quarters=6, seed=1,
                          words_prepared=320, words_per_answer=75)
    data = generate_scenario("incremental", cfg)
    feats, _ = build_features(data["transcripts"], embedder=HashingEmbedder(),
                              verbose=False)
    panel = feats.merge(
        data["panel"], on=["ticker", "period", "fiscal_year", "fiscal_quarter"],
        how="inner", suffixes=("", "_m"),
    )
    ic = information_coefficient(panel, "composite_z", "fwd_ret_21")
    assert ic["small_sample_warning"] is True
