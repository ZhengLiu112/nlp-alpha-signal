#!/usr/bin/env python3
"""
Run the full study end to end.

    python scripts/run_study.py --synthetic       # validate the framework
    python scripts/run_study.py --synthetic-null  # confirm it finds nothing in noise
    python scripts/run_study.py --data data/transcripts.jsonl

Order matters and is deliberate: the framework is validated on synthetic data
with a known answer BEFORE it is pointed at real data. A null result from an
unvalidated pipeline is uninterpretable, because it cannot be distinguished
from a pipeline that would miss a real effect.

Every specification run is written to results/all_variants.csv, not just the
ones that worked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha.features.evasiveness import HashingEmbedder, sanity_check_embedder  # noqa: E402
from alpha.features.composite import DEFAULT_COMPONENTS, component_correlations  # noqa: E402
from alpha.features.pipeline import build_features  # noqa: E402
from alpha.research.evaluation import (  # noqa: E402
    annualize, cost_sensitivity, factor_alpha, ic_decay,
    information_coefficient, quantile_portfolios, subperiod_stability,
)
from alpha.research.orthogonalize import (  # noqa: E402
    build_announcement_controls, compare_raw_vs_residual, orthogonalize,
)
from alpha.stats import benjamini_hochberg  # noqa: E402

HORIZONS = {5: "fwd_ret_5", 21: "fwd_ret_21", 63: "fwd_ret_63"}


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def load_synthetic(null: bool, seed: int = 7):
    from alpha.data.synthetic import SyntheticConfig, generate, generate_null

    cfg = SyntheticConfig(n_firms=150, n_quarters=24, seed=seed)
    data = generate_null(cfg) if null else generate(cfg)
    return data["transcripts"], data["panel"], data["factors"], data["truth"]


def load_real(path: Path):
    """
    Load real transcripts plus the market panel.

    Expects JSONL transcripts matching the schema in alpha/features/segment.py
    and a parquet/csv panel keyed on (ticker, period) carrying forward returns
    and the control variables.
    """
    transcripts = [json.loads(line) for line in path.open() if line.strip()]

    panel_path = path.with_name("panel.parquet")
    if not panel_path.exists():
        panel_path = path.with_name("panel.csv")
    if not panel_path.exists():
        raise FileNotFoundError(
            f"need a market panel next to {path.name} (panel.parquet or panel.csv). "
            "Build it with scripts/01_build_panel.py"
        )
    panel = (pd.read_parquet(panel_path) if panel_path.suffix == ".parquet"
             else pd.read_csv(panel_path))

    factors_path = path.with_name("factors.csv")
    factors = (pd.read_csv(factors_path, index_col=0)
               if factors_path.exists() else pd.DataFrame())
    return transcripts, panel, factors, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run on synthetic data with a planted alpha (power test)")
    ap.add_argument("--synthetic-null", action="store_true",
                    help="run on synthetic noise (specificity test)")
    ap.add_argument("--data", type=Path, help="path to transcripts.jsonl")
    ap.add_argument("--horizon", type=int, default=21, choices=list(HORIZONS))
    ap.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    if not (args.synthetic or args.synthetic_null or args.data):
        ap.error("choose one of --synthetic, --synthetic-null, or --data")

    args.outdir.mkdir(parents=True, exist_ok=True)
    mode = ("synthetic-null" if args.synthetic_null
            else "synthetic" if args.synthetic else "real")

    banner(f"NLP ALPHA SIGNAL STUDY   mode={mode}")

    if mode == "real":
        transcripts, market, factors, truth = load_real(args.data)
    else:
        transcripts, market, factors, truth = load_synthetic(args.synthetic_null)

    if truth:
        print(f"Planted effect: alpha={truth['alpha_strength']}, "
              f"pead={truth['pead_strength']}, confound={truth['confound_strength']}")
        if truth["alpha_strength"] == 0:
            print("EXPECTATION: this run should find NOTHING. A significant "
                  "result here means the framework is broken.")

    # ---- 0. validate the measurement device before using it -------------
    banner("0.  EMBEDDER SANITY CHECK")
    embedder = HashingEmbedder()
    checks = sanity_check_embedder(embedder)
    for k, v in checks.items():
        print(f"  {k:24} {v:.4f}")
    if checks["discrimination_gap"] < 0.05:
        print("  WARNING: embedder barely separates related from unrelated text. "
              "Evasiveness scores will be close to noise.")

    # ---- 1. features ----------------------------------------------------
    banner("1.  FEATURE CONSTRUCTION")
    features, diag = build_features(transcripts, embedder=embedder)
    print(f"  usable calls     : {diag['n_usable']}/{diag['n_input']} "
          f"({diag['usable_rate']:.1%})")
    print(f"  embedder         : {diag['embedder']}")
    print(f"  LM dictionary    : {diag['lm_dictionary_source']}")

    panel = features.merge(
        market, on=["ticker", "period", "fiscal_year", "fiscal_quarter"],
        how="inner", suffixes=("", "_mkt"),
    )
    panel = build_announcement_controls(panel)
    print(f"  merged panel     : {len(panel)} firm-quarters, "
          f"{panel['period'].nunique()} periods")
    print(f"  SUE source       : {panel['sue_source'].iloc[0]}")

    corr = component_correlations(panel)
    if not corr.empty:
        print("\n  Component rank correlations (>0.8 means double counting):")
        print(corr.round(3).to_string().replace("\n", "\n    "))

    # ---- 2. orthogonalization -------------------------------------------
    banner("2.  ORTHOGONALIZATION  (Stage 1)")
    panel, ortho_res = orthogonalize(panel, signal_col="composite_z",
                                     sector_col="sector", period_col="period")
    print(ortho_res.summary())
    print("\n  Control loadings on the text signal:")
    print(ortho_res.coef_table.round(4).to_string(index=False).replace("\n", "\n    "))

    # ---- 3. headline: raw vs orthogonalized -----------------------------
    banner("3.  HEADLINE  —  DOES THE SIGNAL SURVIVE ITS OWN CONTROLS?")
    fwd = HORIZONS[args.horizon]
    headline = compare_raw_vs_residual(panel, "composite_z", "signal_resid", fwd)
    print(headline.round(4).to_string(index=False))

    raw_t = headline.loc[headline.specification == "raw", "t_stat_hac"].iloc[0]
    res_t = headline.loc[headline.specification == "orthogonalized", "t_stat_hac"].iloc[0]
    print("\n  Reading:")
    if abs(raw_t) > 2 and abs(res_t) > 2:
        if np.sign(raw_t) == np.sign(res_t):
            print("    Raw works, residual works, same direction -> genuine incremental information.")
        else:
            print("    Raw and residual are BOTH significant but with OPPOSITE signs.")
            print("    The confound and true alpha work against each other; the residual")
            print("    reveals the signal that was masked in the raw.")
    elif abs(res_t) > 2 and abs(raw_t) <= 2:
        print("    Raw is weak, residual is strongly significant. Two channels in the raw")
        print("    signal (confound and true alpha) are working against each other and")
        print("    partly cancel. Once the confound is stripped, the true alpha shows through.")
        print("    Genuine incremental information -- report the residual result.")
    elif abs(raw_t) > 2 and abs(res_t) <= 2:
        print("    Raw works, residual does NOT -> the signal is a repackaging of")
        print("    the earnings surprise. Report it as such.")
    else:
        print("    Neither significant -> null result. The contribution is the")
        print("    measurement framework and an honest negative finding.")

    # ---- 4. decay --------------------------------------------------------
    banner("4.  IC DECAY BY HORIZON")
    decay = ic_decay(panel, "signal_resid", HORIZONS)
    print(decay.round(4).to_string(index=False))

    # ---- 5. portfolios ---------------------------------------------------
    banner("5.  QUINTILE PORTFOLIOS")
    per_period, summary = quantile_portfolios(panel, "signal_resid", fwd)
    if summary.empty:
        print("  Not enough names per period to form quintiles.")
        return 0
    print(summary.round(4).to_string(index=False))

    ls_stats = annualize(per_period["long_short"], args.horizon)
    print(f"\n  Long/short annualised: return={ls_stats['ann_return']:.2%}  "
          f"vol={ls_stats['ann_vol']:.2%}  Sharpe={ls_stats['sharpe']:.2f}  "
          f"maxDD={ls_stats['max_drawdown']:.2%}")

    # ---- 6. costs --------------------------------------------------------
    banner("6.  TRANSACTION COST SENSITIVITY")
    costs = cost_sensitivity(per_period, args.horizon)
    print(costs.round(4).to_string(index=False))
    if costs.iloc[0]["sharpe"] > 0 and costs.iloc[-1]["sharpe"] <= 0:
        print("\n  NOTE: the strategy does not survive realistic costs. "
              "Say this plainly rather than quoting the gross number.")

    # ---- 7. factor alpha -------------------------------------------------
    if len(factors):
        banner("7.  FAMA-FRENCH FACTOR ALPHA")
        series = per_period.set_index("period")["long_short"]
        fa = factor_alpha(series, factors)
        if np.isfinite(fa.get("alpha_t", np.nan)):
            print(f"  alpha = {fa['alpha']:.5f}  (t = {fa['alpha_t']:.2f}, "
                  f"HAC lags = {fa['hac_lags']})   R2 = {fa['r2']:.3f}")
            print("  loadings: " + ", ".join(
                f"{k}={v:+.3f}" for k, v in fa["loadings"].items()))
            if abs(fa["alpha_t"]) < 2:
                print("  Alpha is not significant after factor exposures. The returns")
                print("  are explained by known factors -- this is repackaged beta.")
        else:
            print(f"  {fa.get('note', 'unavailable')}")

    # ---- 8. subperiods ---------------------------------------------------
    banner("8.  SUBPERIOD STABILITY")
    edges = sorted(panel["period"].unique())
    if len(edges) >= 9:
        bps = [edges[len(edges) // 3], edges[2 * len(edges) // 3]]
        print(subperiod_stability(per_period, bps).round(4).to_string(index=False))
    else:
        print("  Sample too short to split meaningfully.")

    # ---- 9. multiple testing --------------------------------------------
    banner("9.  MULTIPLE TESTING ACROSS THE SPECIFICATION FAMILY")
    variants = []
    for sig in ["composite_z", "signal_resid"] + [f"z_{c}" for c in DEFAULT_COMPONENTS]:
        if sig not in panel.columns:
            continue
        for h, col in HORIZONS.items():
            ic = information_coefficient(panel, sig, col)
            variants.append({
                "signal": sig, "horizon": h, "mean_ic": ic["mean_ic"],
                "t_stat": ic["t_stat"], "p_value": ic["p_value"],
                "n_periods": ic["n_periods"],
            })
    vdf = pd.DataFrame(variants).dropna(subset=["p_value"])
    if not vdf.empty:
        reject, crit = benjamini_hochberg(vdf["p_value"].to_numpy(), alpha=0.05)
        vdf["survives_bh"] = reject
        print(f"  {len(vdf)} specifications tested, BH critical p = {crit:.5f}")
        print(vdf.round(4).to_string(index=False))
        print(f"\n  {int(reject.sum())}/{len(vdf)} survive correction. Every "
              "specification run is listed above, not only the ones that worked.")

        out = args.outdir / f"all_variants_{mode}.csv"
        vdf.to_csv(out, index=False)
        print(f"  -> {out}")

    panel.to_csv(args.outdir / f"panel_{mode}.csv", index=False)
    banner("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
