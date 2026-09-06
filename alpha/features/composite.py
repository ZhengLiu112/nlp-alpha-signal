"""
Composite signal construction.

Equal-weight z-score only, deliberately. A supervised composite (ridge, GBM)
would almost certainly backtest better, and that is exactly the problem: with
five components, three horizons and two tone models there are enough degrees of
freedom to fit noise, and a reviewer cannot tell the difference from outside.
Zero free parameters means there is nothing to overfit and nothing to defend.

If the equal-weight version shows nothing, a supervised version showing
something is far more likely to be fitting than finding.

Component signs are declared explicitly in SIGNAL_DIRECTIONS below, BEFORE any
returns are looked at. Flipping a sign after seeing results is the most common
unrecorded researcher degree of freedom in this kind of study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats import cross_sectional_zscore

# +1 : higher value hypothesised to predict HIGHER forward returns
# -1 : higher value hypothesised to predict LOWER forward returns
#
# Each sign is a prior committed in advance, with its reasoning. These are
# hypotheses to be tested, not conclusions.
SIGNAL_DIRECTIONS: dict[str, int] = {
    # Management talks up the script but cannot sustain it under questioning.
    # Read as unsustainable optimism -> negative.
    "lm_tone_gap": -1,
    # Answers that drift from the questions asked. Read as concealment -> negative.
    "evasiveness": -1,
    # Hedging under live questioning (not scripted boilerplate) -> negative.
    "hedging_density": -1,
    # Analysts sounding negative is outside information not controlled by
    # management -> negative.
    "analyst_tone": +1,
    # Language changed because the business changed, and the market underreacts
    # (Lazy Prices). Direction of the underreaction is ambiguous a priori;
    # tested as a magnitude effect and reported separately rather than being
    # forced into the composite with a guessed sign.
    "semantic_novelty": 0,
}

DEFAULT_COMPONENTS = [k for k, v in SIGNAL_DIRECTIONS.items() if v != 0]


def build_composite(
    df: pd.DataFrame,
    components: list[str] | None = None,
    period_col: str = "period",
    min_components: int = 2,
    min_names_per_period: int = 20,
) -> pd.DataFrame:
    """
    Cross-sectionally z-score each component within each period, apply the
    pre-declared sign, and average.

    ``min_components`` guards against a name whose composite rests on a single
    surviving component -- that is a different, noisier signal wearing the same
    name.

    ``min_names_per_period`` drops thin cross-sections. A z-score across eight
    names is not a cross-sectional rank, and early sample periods with sparse
    transcript coverage will otherwise contribute high-variance noise.
    """
    comps = components or DEFAULT_COMPONENTS
    missing = [c for c in comps if c not in df.columns]
    if missing:
        raise KeyError(f"missing component columns: {missing}")

    out = df.copy()
    z_cols: list[str] = []

    for comp in comps:
        zc = f"z_{comp}"
        out[zc] = np.nan
        sign = SIGNAL_DIRECTIONS.get(comp, 1)
        for period, grp in out.groupby(period_col):
            if len(grp) < min_names_per_period:
                continue
            z = cross_sectional_zscore(grp[comp].to_numpy())
            out.loc[grp.index, zc] = z * sign
        z_cols.append(zc)

    n_valid = out[z_cols].notna().sum(axis=1)
    out["n_components"] = n_valid
    out["composite"] = np.where(
        n_valid >= min_components, out[z_cols].mean(axis=1, skipna=True), np.nan
    )

    # Re-standardise so the composite is comparable across periods regardless
    # of how many components happened to survive.
    out["composite_z"] = np.nan
    for period, grp in out.groupby(period_col):
        if grp["composite"].notna().sum() < min_names_per_period:
            continue
        out.loc[grp.index, "composite_z"] = cross_sectional_zscore(
            grp["composite"].to_numpy()
        )
    return out


def component_correlations(df: pd.DataFrame, components: list[str] | None = None,
                           period_col: str = "period") -> pd.DataFrame:
    """
    Average within-period rank correlation between components.

    If two components correlate above ~0.8 they are one component with two
    names, and the equal-weight composite is silently double-counting it.
    """
    comps = components or DEFAULT_COMPONENTS
    comps = [c for c in comps if c in df.columns]
    mats = []
    for _, grp in df.groupby(period_col):
        sub = grp[comps].dropna()
        if len(sub) < 10:
            continue
        mats.append(sub.rank().corr().to_numpy())
    if not mats:
        return pd.DataFrame(index=comps, columns=comps, dtype=float)
    return pd.DataFrame(np.nanmean(mats, axis=0), index=comps, columns=comps)
