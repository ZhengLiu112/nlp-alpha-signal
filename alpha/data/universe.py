"""
Point-in-time universe reconstruction.

A dataset described as "S&P 500 companies" is almost always constituents as of a
recent snapshot. Backtesting on it restricts the sample to firms that survived
to today, which inflates returns and is the first thing a reviewer asks about.

The fix is also the strongest talking point in the project: run the backtest on
both universes and quantify how much survivorship bias inflated the naive
result. Measuring your own bias is more convincing than claiming not to have it.

Historical S&P 500 additions and deletions are freely available; several
maintained GitHub mirrors track the changes list. Point ``changes_path`` at one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_pit_membership(changes_path: Path, current_members: list[str],
                         start: str = "2005-01-01") -> pd.DataFrame:
    """
    Reconstruct membership by walking the additions/deletions list backwards.

    Expects columns: ``date``, ``added_ticker``, ``removed_ticker``.

    Returns a long frame of (date, ticker) membership intervals.
    """
    changes = pd.read_csv(changes_path, parse_dates=["date"]).sort_values(
        "date", ascending=False
    )
    members = set(current_members)
    snapshots = [(pd.Timestamp.today().normalize(), sorted(members))]

    for _, row in changes.iterrows():
        if row["date"] < pd.Timestamp(start):
            break
        # Walking backwards inverts each change: an addition means the firm was
        # NOT a member before that date, and vice versa.
        if pd.notna(row.get("added_ticker")):
            members.discard(str(row["added_ticker"]).upper())
        if pd.notna(row.get("removed_ticker")):
            members.add(str(row["removed_ticker"]).upper())
        snapshots.append((row["date"], sorted(members)))

    rows = [{"date": d, "ticker": t} for d, mem in snapshots for t in mem]
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def was_member(membership: pd.DataFrame, ticker: str, date) -> bool:
    """Membership as of a date, using the most recent snapshot at or before it."""
    date = pd.Timestamp(date)
    prior = membership[membership["date"] <= date]
    if prior.empty:
        return False
    latest = prior["date"].max()
    return ticker.upper() in set(membership.loc[membership["date"] == latest, "ticker"])


def filter_to_pit_universe(panel: pd.DataFrame, membership: pd.DataFrame,
                           date_col: str = "call_date") -> pd.DataFrame:
    """Restrict a panel to firms that were index members at the time of the call."""
    keep = [
        was_member(membership, row["ticker"], row[date_col])
        for _, row in panel.iterrows()
    ]
    return panel.loc[keep].reset_index(drop=True)


def survivorship_gap(naive_result: dict, pit_result: dict) -> str:
    """
    Quantify how much the survivor-only universe inflated the headline.

    Report this number. Volunteering it is worth more than the result it
    qualifies.
    """
    lines = ["Survivorship bias impact (survivor-only vs point-in-time):"]
    for key in ("mean_ic", "t_stat", "sharpe", "ann_return"):
        if key in naive_result and key in pit_result:
            n, p = naive_result[key], pit_result[key]
            if p not in (0, None) and pd.notna(p) and pd.notna(n):
                lines.append(f"  {key:12} {n:+.4f} -> {p:+.4f}  "
                             f"(inflated by {(n - p):+.4f})")
    return "\n".join(lines)
