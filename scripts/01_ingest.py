#!/usr/bin/env python3
"""
Step 1: load real transcripts, inspect the schema, and report coverage.

Run this BEFORE anything else and read the output rather than skimming it. The
two decisions made here — whether the schema parsed correctly and where the
sample starts — determine whether every downstream number means anything.

    python scripts/01_ingest.py --limit 500     # quick schema check
    python scripts/01_ingest.py                 # full ingest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha.data.ingest_transcripts import (  # noqa: E402
    coverage_report, inspect_sample, load_huggingface, save_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Bose345/sp500_earnings_transcripts")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "transcripts.jsonl")
    ap.add_argument("--allow-flat-fallback", action="store_true",
                    help="parse unstructured transcripts by regex (worse; last resort)")
    args = ap.parse_args()

    print(f"Loading {args.dataset} ...")
    records, diag = load_huggingface(args.dataset, limit=args.limit,
                                     allow_flat_fallback=args.allow_flat_fallback)

    print("\n=== SCHEMA ===")
    print(f"  source rows     : {diag['n_source_rows']}")
    print(f"  normalised      : {diag['n_normalised']} ({diag['normalised_rate']:.1%})")
    print(f"  available fields: {', '.join(diag['available_fields'])}")
    if "WARNING" in diag:
        print(f"\n  WARNING: {diag['WARNING']}")

    if not records:
        print("\nNothing parsed. Compare available_fields above against the key "
              "lists in alpha/data/ingest_transcripts.py and extend them.")
        return 1

    print("\n=== SAMPLE (hand-check these) ===")
    print(inspect_sample(records, n=3))

    print("\n=== COVERAGE BY QUARTER ===")
    cov = coverage_report(records)
    print(cov.to_string(index=False))
    print("\n  Choose the backtest start from where coverage stabilises, and")
    print("  record it in SPEC.md BEFORE running the study.")

    save_jsonl(records, args.out)
    cov.to_csv(ROOT / "results" / "coverage.csv", index=False)
    print(f"\nWrote {len(records)} records -> {args.out}")

    print("\nBefore proceeding:")
    print("  [ ] licence checked on the dataset card")
    print("  [ ] ~20 transcripts hand-inspected, not just the 3 above")
    print("  [ ] full Loughran-McDonald dictionary at data/lm_dictionary.csv")
    print("  [ ] point-in-time universe built (alpha/data/universe.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
