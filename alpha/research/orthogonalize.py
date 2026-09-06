"""
The orthogonalization protocol.

This is the module that decides whether the project is a real study or a
repackaging of post-earnings-announcement drift.

The obvious objection to any earnings-text signal: it is a noisy proxy for the
earnings surprise, and the earnings surprise is already known to predict drift
(Bernard & Thomas, 1989). Unless that is addressed structurally, the whole
result is a rediscovery of a 1989 anomaly through an expensive NLP pipeline.

    Stage 1   text_signal ~ SUE + announcement_return + size + value
                            + momentum + sector + period FE
              residual = text_signal - fitted

    Stage 2   forward_return ~ residual

Both raw and residual results are reported side by side. Three outcomes, all
publishable in spirit:

  raw works, residual works       -> genuine incremental information
  raw works, residual does not    -> the signal IS the earnings surprise.
                                     Report it. Saying it first is far better
                                     than having an interviewer find it.
  neither works                   -> null result. The contribution becomes the
                                     measurement framework and the finding that
                                     a widely-assumed effect does not survive
                                     proper controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..stats import ols

# Controls, declared up front. Anything absent from the frame is skipped and
# recorded in `controls_used`, so a result computed without SUE is never
# mistaken for one computed with it.
DEFAULT_CONTROLS = [
    "sue",                    # standardised unexpected earnings
    "announcement_return",    # 3-day return around the announcement (PEAD proxy)
    "log_mktcap",             # size
    "book_to_market",         # value
    "momentum_12_1",          # 12-1 month momentum
]


@dataclass
class OrthogonalizationResult:
    residual: pd.Series
    controls_used: list[str]
    controls_missing: list[str]
    r2: float
    coef_table: pd.DataFrame
    n_obs: int
    sector_dummies: bool
    period_fe: bool

    def summary(self) -> str:
        lines = [
            f"Stage 1: signal ~ controls    n={self.n_obs}  R2={self.r2:.4f}",
            f"  controls used   : {', '.join(self.controls_used) or '(none)'}",
        ]
        if self.controls_missing:
            lines.append(f"  MISSING         : {', '.join(self.controls_missing)}")
        lines.append(f"  sector dummies  : {self.sector_dummies}")
        lines.append(f"  period FE       : {self.period_fe}")
        if self.r2 > 0.5:
            lines.append(
                "  NOTE: controls explain >50% of the signal. Little independent "
                "variation remains; interpret Stage-2 results cautiously."
            )
        return "\n".join(lines)


def _design_matrix(df: pd.DataFrame, controls: list[str], sector_col: str | None,
                   period_col: str | None) -> tuple[np.ndarray, list[str]]:
    blocks, names = [np.ones((len(df), 1))], ["const"]

    for c in controls:
        blocks.append(df[[c]].to_numpy(dtype=float))
        names.append(c)

    # Drop-first dummy coding to keep the design full rank alongside the intercept.
    if sector_col and sector_col in df.columns:
        d = pd.get_dummies(df[sector_col].astype(str), prefix="sec", drop_first=True)
        if d.shape[1] > 0:
            blocks.append(d.to_numpy(dtype=float))
            names.extend(d.columns.tolist())

    if period_col and period_col in df.columns:
        d = pd.get_dummies(df[period_col].astype(str), prefix="per", drop_first=True)
        if d.shape[1] > 0:
            blocks.append(d.to_numpy(dtype=float))
            names.extend(d.columns.tolist())

    return np.column_stack(blocks), names


def orthogonalize(
    df: pd.DataFrame,
    signal_col: str = "composite_z",
    controls: list[str] | None = None,
    sector_col: str | None = "sector",
    period_col: str | None = "period",
    residual_col: str = "signal_resid",
    standardize_residual: bool = True,
) -> tuple[pd.DataFrame, OrthogonalizationResult]:
    """
    Stage 1. Strip from the text signal everything already known to predict returns.

    Period fixed effects absorb market-wide moves, so what remains is purely
    cross-sectional -- which is what a long/short book actually trades.

    The residual is re-standardised within period by default so it stays on a
    comparable scale to the raw signal and the two can be compared directly.
    """
    ctrl = [c for c in (controls or DEFAULT_CONTROLS) if c in df.columns]
    missing = [c for c in (controls or DEFAULT_CONTROLS) if c not in df.columns]

    work = df.copy()
    need = [signal_col] + ctrl
    mask = work[need].notna().all(axis=1)
    if sector_col and sector_col in work.columns:
        mask &= work[sector_col].notna()
    if mask.sum() < 30:
        raise ValueError(f"too few complete observations for Stage 1: {int(mask.sum())}")

    sub = work.loc[mask]
    X, names = _design_matrix(sub, ctrl, sector_col, period_col)
    y = sub[signal_col].to_numpy(dtype=float)

    res = ols(y, X, cov_type="HC0", names=names)

    work[residual_col] = np.nan
    work.loc[sub.index, residual_col] = res.resid

    if standardize_residual and period_col and period_col in work.columns:
        from ..stats import cross_sectional_zscore

        for _, grp in work.groupby(period_col):
            vals = grp[residual_col].to_numpy(dtype=float)
            if np.isfinite(vals).sum() >= 20:
                work.loc[grp.index, residual_col] = cross_sectional_zscore(vals)

    keep = len(ctrl) + 1
    coef_table = pd.DataFrame({
        "term": names[:keep],
        "coef": res.params[:keep],
        "se": res.se[:keep],
        "t": res.tstat[:keep],
        "p": res.pvalue[:keep],
    })

    return work, OrthogonalizationResult(
        residual=work[residual_col],
        controls_used=ctrl,
        controls_missing=missing,
        r2=res.r2,
        coef_table=coef_table,
        n_obs=res.nobs,
        sector_dummies=bool(sector_col and sector_col in df.columns),
        period_fe=bool(period_col and period_col in df.columns),
    )


def build_announcement_controls(
    df: pd.DataFrame,
    returns_col: str = "announcement_return",
    eps_actual: str = "eps_actual",
    eps_estimate: str = "eps_estimate",
    period_col: str = "period",
) -> pd.DataFrame:
    """
    Construct the PEAD controls.

    SUE is computed as the standardised earnings surprise where consensus
    estimates are available. Free consensus data at scale is the hardest item
    in this project; where it is missing, the 3-day announcement-window return
    carries the load as a PEAD proxy.

    That substitution is weaker and it is flagged in ``sue_source`` on every
    row, so the limitation travels with the data instead of living in a
    forgotten note.
    """
    out = df.copy()

    if eps_actual in out.columns and eps_estimate in out.columns:
        surprise = out[eps_actual] - out[eps_estimate]
        denom = out[eps_estimate].abs().replace(0, np.nan)
        raw_sue = surprise / denom
        out["sue"] = np.nan
        for _, grp in out.groupby(period_col):
            v = raw_sue.loc[grp.index]
            sd = v.std(ddof=1)
            if pd.notna(sd) and sd > 0:
                out.loc[grp.index, "sue"] = ((v - v.mean()) / sd).clip(-3, 3)
        out["sue_source"] = "eps_consensus"
    elif returns_col in out.columns:
        out["sue"] = np.nan
        for _, grp in out.groupby(period_col):
            v = out.loc[grp.index, returns_col]
            sd = v.std(ddof=1)
            if pd.notna(sd) and sd > 0:
                out.loc[grp.index, "sue"] = ((v - v.mean()) / sd).clip(-3, 3)
        out["sue_source"] = "announcement_return_proxy"
    else:
        out["sue"] = np.nan
        out["sue_source"] = "unavailable"

    return out


def compare_raw_vs_residual(
    df: pd.DataFrame,
    raw_col: str,
    resid_col: str,
    forward_return_col: str,
    period_col: str = "period",
) -> pd.DataFrame:
    """
    The headline table: does the signal survive its own controls?

    Returns one row per specification with mean IC, HAC t-stat and hit rate,
    so raw and residual sit side by side and the reader draws their own
    conclusion.
    """
    from .evaluation import information_coefficient

    rows = []
    for label, col in (("raw", raw_col), ("orthogonalized", resid_col)):
        if col not in df.columns:
            continue
        ic = information_coefficient(df, col, forward_return_col, period_col=period_col)
        rows.append({
            "specification": label,
            "signal": col,
            "mean_ic": ic["mean_ic"],
            "ic_se": ic["ic_se"],
            "t_stat_hac": ic["t_stat"],
            "p_value": ic["p_value"],
            "hac_lags": ic["hac_lags"],
            "n_periods": ic["n_periods"],
            "ic_hit_rate": ic["hit_rate"],
        })
    return pd.DataFrame(rows)
