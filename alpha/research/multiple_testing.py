"""
Multiple-testing control across the specification family.

Any study that tries several signals, horizons and model choices is running
dozens of implicit tests, and the best of dozens looks good by chance alone.
Reporting only the winner is how a fishing expedition gets written up as a
finding.

The rule this project follows: every specification run appears in the output
table, and the table records which survive correction. A results table with two
dozen rows, most of them unremarkable, is far more persuasive than one row with
a great t-statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats import benjamini_hochberg


def correct_family(results: pd.DataFrame, pvalue_col: str = "p_value",
                   alpha: float = 0.05) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg across a family of specifications.

    FDR rather than Bonferroni: these tests are positively correlated (the same
    signal at 5, 21 and 63 days is not three independent questions), so
    controlling family-wise error would be needlessly conservative and would
    discard real findings.
    """
    out = results.copy()
    valid = out[pvalue_col].notna()
    out["survives_bh"] = False
    out["bh_critical_p"] = np.nan

    if valid.sum() == 0:
        return out

    reject, crit = benjamini_hochberg(out.loc[valid, pvalue_col].to_numpy(), alpha)
    out.loc[valid, "survives_bh"] = reject
    out["bh_critical_p"] = crit
    out["n_tests_in_family"] = int(valid.sum())
    return out


def family_summary(corrected: pd.DataFrame) -> str:
    """One-paragraph summary suitable for pasting into a README."""
    n = len(corrected)
    survivors = int(corrected.get("survives_bh", pd.Series(dtype=bool)).sum())
    crit = corrected["bh_critical_p"].dropna()
    crit_val = float(crit.iloc[0]) if len(crit) else float("nan")

    lines = [
        f"{n} specifications tested. BH critical p = {crit_val:.5f} at alpha=0.05.",
        f"{survivors}/{n} survive correction.",
    ]
    if survivors == 0:
        lines.append(
            "Nothing survives. Reported as a null result rather than by quoting "
            "the best uncorrected row."
        )
    elif survivors < n * 0.2:
        lines.append(
            "A small minority survive. Treat these as hypotheses for out-of-sample "
            "testing, not as established effects."
        )
    return " ".join(lines)
