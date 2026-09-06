"""
Statistical primitives, implemented directly rather than pulled from statsmodels.

Two reasons this is hand-rolled:
  1. Overlapping holding periods induce serial correlation in the IC time series.
     Naive OLS standard errors are then badly understated, which is the single
     most common way a backtest reports a t-stat it has not earned.
  2. Writing the HAC estimator out makes the bandwidth choice explicit and
     auditable instead of hidden behind a default argument.

Reference: Newey & West (1987), "A Simple, Positive Semi-Definite,
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OLSResult:
    """Coefficients plus the covariance flavour actually used for inference."""

    params: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    pvalue: np.ndarray
    resid: np.ndarray
    fitted: np.ndarray
    nobs: int
    rank: int
    r2: float
    cov_type: str
    lags: int | None = None
    names: list[str] = field(default_factory=list)

    def summary(self) -> str:
        width = max([len(n) for n in self.names] + [8]) if self.names else 8
        head = (
            f"OLS  n={self.nobs}  rank={self.rank}  R2={self.r2:.4f}  "
            f"cov={self.cov_type}"
            + (f"(lags={self.lags})" if self.lags is not None else "")
        )
        lines = [head, "-" * max(len(head), width + 40)]
        lines.append(f"{'term'.ljust(width)}  {'coef':>10} {'se':>10} {'t':>8} {'p':>8}")
        for i, name in enumerate(self.names or [f"x{i}" for i in range(len(self.params))]):
            lines.append(
                f"{name.ljust(width)}  {self.params[i]:>10.5f} {self.se[i]:>10.5f} "
                f"{self.tstat[i]:>8.3f} {self.pvalue[i]:>8.4f}"
            )
        return "\n".join(lines)


def _t_sf(t: np.ndarray, df: int) -> np.ndarray:
    """
    Two-sided tail probability under Student's t with ``df`` degrees of freedom.

    Used instead of the normal because HAC inference is asymptotic and this
    project's own calibration study measured it over-rejecting badly on short
    samples -- 25% against a nominal 5% on ten periods. Normal critical values
    make that worse; the t distribution is the standard partial remedy.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if df <= 0:
        return np.full_like(t, np.nan)
    try:
        from scipy import stats

        return 2.0 * stats.t.sf(np.abs(t), df)
    except ImportError:
        return _normal_sf(t)


def _normal_sf(z: np.ndarray) -> np.ndarray:
    """Two-sided tail probability under the normal, via erfc. Avoids a scipy import."""
    from math import erfc, sqrt

    z = np.atleast_1d(np.asarray(z, dtype=float))
    return np.array([erfc(abs(v) / sqrt(2.0)) for v in z])


def newey_west_lags(nobs: int) -> int:
    """
    Automatic bandwidth, the common 4*(n/100)^(2/9) rule of thumb.

    Stated explicitly so the number appearing in results is reproducible and
    can be defended, rather than inherited from a library default.
    """
    if nobs <= 1:
        return 0
    return int(np.floor(4.0 * (nobs / 100.0) ** (2.0 / 9.0)))


def ols(
    y: np.ndarray,
    X: np.ndarray,
    *,
    cov_type: str = "nonrobust",
    lags: int | None = None,
    names: list[str] | None = None,
    add_const: bool = False,
) -> OLSResult:
    """
    Least squares with optional HC0 (White) or HAC (Newey-West) covariance.

    Parameters
    ----------
    cov_type : {"nonrobust", "HC0", "HAC"}
        "HAC" is what you want for any time series of overlapping returns.
    lags : int, optional
        HAC bandwidth. Defaults to :func:`newey_west_lags` when omitted.
    add_const : bool
        Prepend an intercept column.

    Uses lstsq rather than a normal-equation inverse, so rank-deficient designs
    degrade gracefully instead of producing silent nonsense.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]

    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
        names = ["const"] + list(names or [f"x{i}" for i in range(X.shape[1] - 1)])
    if names is None:
        names = [f"x{i}" for i in range(X.shape[1])]

    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[mask], X[mask]
    n, k = X.shape
    if n <= k:
        raise ValueError(f"not enough observations: n={n} <= k={k}")

    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted

    XtX_inv = np.linalg.pinv(X.T @ X)

    if cov_type == "nonrobust":
        sigma2 = resid @ resid / (n - rank)
        cov = sigma2 * XtX_inv
        used_lags = None
    elif cov_type == "HC0":
        S = (X * (resid**2)[:, None]).T @ X
        cov = XtX_inv @ S @ XtX_inv
        used_lags = None
    elif cov_type == "HAC":
        used_lags = newey_west_lags(n) if lags is None else int(lags)
        u = X * resid[:, None]
        S = u.T @ u
        for L in range(1, used_lags + 1):
            w = 1.0 - L / (used_lags + 1.0)  # Bartlett kernel
            G = u[L:].T @ u[:-L]
            S += w * (G + G.T)
        # Finite-sample scaling. The plain HAC sandwich is biased downward in
        # short samples, which inflates t-stats exactly where the sample is too
        # short to support them. n/(n-k) is the standard correction and it is
        # applied by default rather than offered as an option.
        S *= n / max(n - rank, 1)
        cov = XtX_inv @ S @ XtX_inv
    else:
        raise ValueError(f"unknown cov_type: {cov_type!r}")

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, np.nan)
    pval = _t_sf(tstat, df=n - rank)

    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else np.nan

    return OLSResult(
        params=beta, se=se, tstat=tstat, pvalue=pval, resid=resid, fitted=fitted,
        nobs=n, rank=int(rank), r2=r2, cov_type=cov_type, lags=used_lags, names=names,
    )


def mean_with_hac_tstat(x: np.ndarray, lags: int | None = None) -> tuple[float, float, float, int]:
    """
    Mean of a serially-correlated series with a HAC t-statistic.

    This is the right way to test "is the average IC different from zero" when
    holding periods overlap. Returns (mean, se, tstat, lags_used).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan, 0)
    res = ols(x, np.ones((len(x), 1)), cov_type="HAC", lags=lags, names=["const"])
    return float(res.params[0]), float(res.se[0]), float(res.tstat[0]), int(res.lags or 0)


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, float]:
    """
    Benjamini-Hochberg step-up procedure controlling the false discovery rate.

    Returns (reject_flags, critical_pvalue). Any specification search over
    models, horizons and components is a multiple-testing problem; reporting
    which results survive correction is what separates a study from a fishing
    expedition.
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.nan
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(n, dtype=bool), 0.0
    kmax = int(np.max(np.where(passed)[0]))
    crit = ranked[kmax]
    return p <= crit, float(crit)


def spearman_ic(signal: np.ndarray, forward_return: np.ndarray) -> float:
    """
    Rank correlation between a signal and forward returns.

    Rank-based rather than Pearson because cross-sectional return distributions
    are fat-tailed, and a handful of extreme names should not decide the metric.
    """
    s = np.asarray(signal, dtype=float)
    r = np.asarray(forward_return, dtype=float)
    mask = np.isfinite(s) & np.isfinite(r)
    s, r = s[mask], r[mask]
    if len(s) < 3:
        return np.nan
    from scipy.stats import rankdata

    sr, rr = rankdata(s), rankdata(r)
    sr = sr - sr.mean()
    rr = rr - rr.mean()
    denom = np.sqrt((sr**2).sum() * (rr**2).sum())
    return float((sr @ rr) / denom) if denom > 0 else np.nan


def cross_sectional_zscore(x: np.ndarray, clip: float | None = 3.0) -> np.ndarray:
    """Demean and scale within a cross-section, with optional winsorising."""
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if mask.sum() < 2:
        return out
    v = x[mask]
    sd = v.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        out[mask] = 0.0
        return out
    z = (v - v.mean()) / sd
    if clip is not None:
        z = np.clip(z, -clip, clip)
    out[mask] = z
    return out
