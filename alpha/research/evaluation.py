"""
The evaluation battery.

These are the numbers a quant researcher actually asks for, in the order they
ask for them. Every one is computed with HAC standard errors where the
underlying series is serially correlated, which for overlapping holding
periods is always.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats import mean_with_hac_tstat, ols, spearman_ic

TRADING_DAYS = 252


def information_coefficient(
    df: pd.DataFrame,
    signal_col: str,
    forward_return_col: str,
    period_col: str = "period",
    min_names: int = 20,
) -> dict:
    """
    Cross-sectional rank IC, averaged over periods, with a HAC t-statistic.

    The HAC correction is the point. Overlapping holding periods make the IC
    series autocorrelated; a naive t-stat on that series is inflated, sometimes
    by a factor of two. Reporting one is the fastest way to lose credibility in
    an interview.
    """
    ics, periods = [], []
    for period, grp in df.groupby(period_col):
        sub = grp[[signal_col, forward_return_col]].dropna()
        if len(sub) < min_names:
            continue
        ic = spearman_ic(sub[signal_col].to_numpy(), sub[forward_return_col].to_numpy())
        if np.isfinite(ic):
            ics.append(ic)
            periods.append(period)

    if len(ics) < 3:
        return {
            "mean_ic": np.nan, "ic_se": np.nan, "t_stat": np.nan, "p_value": np.nan,
            "hac_lags": 0, "n_periods": len(ics), "hit_rate": np.nan,
            "small_sample_warning": True,
            "ic_std": np.nan, "ic_ir": np.nan, "ic_series": pd.Series(dtype=float),
        }

    arr = np.array(ics)
    mean_ic, se, tstat, lags = mean_with_hac_tstat(arr)
    from ..stats import _t_sf

    # HAC standard errors are asymptotic. With few periods they are biased
    # DOWNWARD, so t-stats are inflated and the test over-rejects -- measured
    # at roughly 17% against a nominal 5% on a 10-quarter sample in this
    # project's own calibration study (alpha/research/calibration.py).
    #
    # Surfaced on the result rather than left in a footnote, because a
    # borderline t-stat on a short sample is exactly where this matters and
    # exactly where it gets forgotten.
    small_sample = len(arr) < 20

    return {
        "mean_ic": mean_ic,
        "ic_se": se,
        "t_stat": tstat,
        # t critical values with (n_periods - 1) df, not normal: HAC is
        # asymptotic and this project's calibration measured it over-rejecting
        # on short samples.
        "p_value": (float(_t_sf(np.array([tstat]), df=len(arr) - 1)[0])
                    if np.isfinite(tstat) else np.nan),
        "hac_lags": lags,
        "n_periods": len(arr),
        "small_sample_warning": small_sample,
        "hit_rate": float((arr > 0).mean()),
        "ic_std": float(arr.std(ddof=1)),
        # Information ratio of the IC series itself (Grinold's "IC IR").
        "ic_ir": float(mean_ic / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else np.nan,
        "ic_series": pd.Series(arr, index=periods),
    }


def ic_decay(
    df: pd.DataFrame,
    signal_col: str,
    horizon_cols: dict[int, str],
    period_col: str = "period",
) -> pd.DataFrame:
    """
    IC as a function of holding horizon.

    Shape matters more than level. A signal that peaks at 1 day and dies is a
    reaction, not a drift, and is usually not tradeable after costs. A signal
    that builds to 21-63 days is the underreaction story the Lazy Prices
    literature describes.
    """
    rows = []
    for h in sorted(horizon_cols):
        col = horizon_cols[h]
        if col not in df.columns:
            continue
        ic = information_coefficient(df, signal_col, col, period_col=period_col)
        rows.append({
            "horizon_days": h, "mean_ic": ic["mean_ic"], "t_stat": ic["t_stat"],
            "p_value": ic["p_value"], "ic_ir": ic["ic_ir"], "n_periods": ic["n_periods"],
        })
    return pd.DataFrame(rows)


def quantile_portfolios(
    df: pd.DataFrame,
    signal_col: str,
    forward_return_col: str,
    period_col: str = "period",
    n_quantiles: int = 5,
    min_names: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort into quantiles each period; return per-period portfolio returns.

    Equal-weighted within bucket. ``min_names`` is enforced because quintiles
    of a 25-name cross-section are five-name portfolios, and their spread is
    idiosyncratic noise rather than a factor return.

    Returns (per-period returns, summary by quantile).
    """
    recs = []
    for period, grp in df.groupby(period_col):
        sub = grp[[signal_col, forward_return_col]].dropna()
        if len(sub) < min_names:
            continue
        ranks = sub[signal_col].rank(method="first")
        buckets = pd.qcut(ranks, n_quantiles, labels=False, duplicates="drop")
        if buckets.isna().all():
            continue
        row = {"period": period, "n_names": len(sub)}
        for q in range(n_quantiles):
            sel = sub.loc[buckets == q, forward_return_col]
            row[f"q{q + 1}"] = float(sel.mean()) if len(sel) else np.nan
        row["long_short"] = row.get(f"q{n_quantiles}", np.nan) - row.get("q1", np.nan)
        row["long_only"] = row.get(f"q{n_quantiles}", np.nan)
        recs.append(row)

    per_period = pd.DataFrame(recs).sort_values("period").reset_index(drop=True)
    if per_period.empty:
        return per_period, pd.DataFrame()

    summary_rows = []
    for col in [f"q{i + 1}" for i in range(n_quantiles)] + ["long_short", "long_only"]:
        if col not in per_period.columns:
            continue
        s = per_period[col].dropna()
        if len(s) < 3:
            continue
        m, se, t, lags = mean_with_hac_tstat(s.to_numpy())
        summary_rows.append({
            "portfolio": col, "mean_return": m, "se_hac": se, "t_stat": t,
            "hac_lags": lags, "std": float(s.std(ddof=1)),
            "hit_rate": float((s > 0).mean()), "n_periods": len(s),
        })
    return per_period, pd.DataFrame(summary_rows)


def annualize(per_period_returns: pd.Series, holding_days: int) -> dict:
    """
    Annualised statistics for a periodic return series.

    ``holding_days`` sets the compounding frequency. Getting it wrong is a
    common and embarrassing way to report a Sharpe that is off by sqrt(k).
    """
    s = pd.Series(per_period_returns).dropna()
    if len(s) < 3:
        return {k: np.nan for k in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "n_periods")}

    periods_per_year = TRADING_DAYS / max(holding_days, 1)
    mean_p, vol_p = float(s.mean()), float(s.std(ddof=1))
    ann_ret = mean_p * periods_per_year
    ann_vol = vol_p * np.sqrt(periods_per_year)

    equity = (1.0 + s).cumprod()
    dd = float((equity / equity.cummax() - 1.0).min())

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "max_drawdown": dd,
        "n_periods": len(s),
    }


def apply_transaction_costs(
    per_period: pd.DataFrame,
    cost_bps_per_side: float,
    turnover_per_period: float = 2.0,
    return_col: str = "long_short",
) -> pd.Series:
    """
    Subtract costs from gross returns.

    ``turnover_per_period`` defaults to 2.0 for a long/short book that fully
    rebalances (one side out, one side in, both legs). Event-driven signals
    turn over completely by construction, which is exactly why cost sensitivity
    can decide the whole result.
    """
    cost = (cost_bps_per_side / 10_000.0) * turnover_per_period
    return per_period[return_col] - cost


def cost_sensitivity(
    per_period: pd.DataFrame,
    holding_days: int,
    return_col: str = "long_short",
    cost_levels: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0),
    turnover_per_period: float = 2.0,
) -> pd.DataFrame:
    """Headline statistics at several cost assumptions. If it only works at 0 bp, say so."""
    rows = []
    for bps in cost_levels:
        net = apply_transaction_costs(per_period, bps, turnover_per_period, return_col)
        stats = annualize(net, holding_days)
        m, se, t, lags = mean_with_hac_tstat(net.dropna().to_numpy())
        rows.append({
            "cost_bps_per_side": bps, "mean_return": m, "t_stat": t,
            "ann_return": stats["ann_return"], "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
        })
    return pd.DataFrame(rows)


def factor_alpha(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    factor_cols: list[str] | None = None,
) -> dict:
    """
    Regress strategy returns on Fama-French factors plus momentum.

    The credibility test. If the returns load on known factors and alpha is
    insignificant, there is no edge here -- only repackaged beta with an NLP
    pipeline attached. Reported with HAC standard errors.

    Expects the strategy series and the factor frame to share an index.
    """
    cols = factor_cols or [c for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
                           if c in factors.columns]
    if not cols:
        return {"alpha": np.nan, "alpha_t": np.nan, "note": "no factor columns found"}

    joined = pd.concat([pd.Series(portfolio_returns).rename("ret"), factors[cols]],
                       axis=1, join="inner").dropna()
    if len(joined) < len(cols) + 5:
        return {"alpha": np.nan, "alpha_t": np.nan,
                "note": f"insufficient overlap: {len(joined)} periods"}

    res = ols(joined["ret"].to_numpy(), joined[cols].to_numpy(),
              cov_type="HAC", names=cols, add_const=True)

    return {
        "alpha": float(res.params[0]),
        "alpha_se": float(res.se[0]),
        "alpha_t": float(res.tstat[0]),
        "alpha_p": float(res.pvalue[0]),
        "r2": res.r2,
        "n_periods": res.nobs,
        "hac_lags": res.lags,
        "loadings": {c: float(res.params[i + 1]) for i, c in enumerate(cols)},
        "loading_t": {c: float(res.tstat[i + 1]) for i, c in enumerate(cols)},
        "note": "",
    }


def subperiod_stability(
    per_period: pd.DataFrame,
    breakpoints: list[str],
    return_col: str = "long_short",
    period_col: str = "period",
) -> pd.DataFrame:
    """
    Split the sample and report each regime separately.

    A signal that worked only in one regime is a story about that regime. This
    is where most "great" backtests quietly fall apart, so it belongs in the
    main results rather than an appendix.
    """
    df = per_period.copy()
    df[period_col] = df[period_col].astype(str)
    edges = ["", *sorted(breakpoints), "~"]

    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = df[(df[period_col] >= lo) & (df[period_col] < hi)]
        s = sel[return_col].dropna()
        if len(s) < 3:
            continue
        m, se, t, lags = mean_with_hac_tstat(s.to_numpy())
        rows.append({
            "subperiod": f"{lo or 'start'} to {hi if hi != '~' else 'end'}",
            "n_periods": len(s), "mean_return": m, "t_stat": t,
            "hit_rate": float((s > 0).mean()),
        })
    return pd.DataFrame(rows)
