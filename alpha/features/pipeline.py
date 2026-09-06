"""
Feature pipeline: raw transcript records -> a scored firm-quarter panel.

Segmentation is validated before anything is scored, and calls that fail are
dropped and counted rather than scored on a bad split. The rejection tally is
returned alongside the features so it always appears in the results, instead of
being a number nobody checked.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .composite import build_composite
from .evasiveness import HashingEmbedder, evasiveness_features
from .novelty import compute_novelty
from .segment import SegmentationError, segment_call, validate_call
from .tone import load_lm_dictionary, tone_gap_features


def build_features(
    transcripts: list[dict],
    embedder=None,
    dictionary=None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Score every transcript and assemble the firm-quarter feature panel.

    Returns (features, diagnostics). Always read the diagnostics: a low
    ``usable_rate`` means the segmentation assumptions do not match this
    dataset, and every downstream number is then built on a biased subsample of
    the calls that happened to parse.
    """
    emb = embedder or HashingEmbedder()
    lm = dictionary or load_lm_dictionary()

    rejected = Counter()
    rows = []

    for rec in transcripts:
        try:
            call = segment_call(rec)
        except SegmentationError:
            rejected["segmentation_error"] += 1
            continue

        problems = validate_call(call)
        if problems:
            for p in problems:
                rejected[p] += 1
            continue

        feats = {
            "ticker": call.ticker,
            "call_date": call.call_date,
            "fiscal_year": call.fiscal_year,
            "fiscal_quarter": call.fiscal_quarter,
            "period": f"{call.fiscal_year}Q{call.fiscal_quarter}",
            "prepared_text": call.prepared_text,
            "n_analysts": float(call.n_analysts),
            "boundary_method": call.boundary_method,
        }
        feats.update(tone_gap_features(
            call.prepared_text, call.mgmt_answer_text,
            call.analyst_question_text, dictionary=lm,
        ))
        feats.update(evasiveness_features(call.qa_pairs, emb))
        rows.append(feats)

    if not rows:
        raise ValueError(
            f"no usable calls. Rejection reasons: {dict(rejected)}. "
            "Check that the transcript schema matches alpha/features/segment.py."
        )

    df = pd.DataFrame(rows)
    df = compute_novelty(df, emb, text_col="prepared_text")
    df = build_composite(df)

    diagnostics = {
        "n_input": len(transcripts),
        "n_usable": len(df),
        "usable_rate": len(df) / len(transcripts) if transcripts else 0.0,
        "rejections": dict(rejected),
        "embedder": getattr(emb, "name", type(emb).__name__),
        "lm_dictionary_source": lm.source,
        "lm_is_seed": lm.is_seed,
    }

    if verbose:
        print(f"[features] {diagnostics['n_usable']}/{diagnostics['n_input']} calls usable "
              f"({diagnostics['usable_rate']:.1%})")
        if rejected:
            print(f"[features] rejections: {dict(rejected)}")
        if lm.is_seed:
            print("[features] WARNING: using SEED word lists, not the full "
                  "Loughran-McDonald dictionary. Download it into data/lm_dictionary.csv "
                  "before reporting any result.")

    # Drop the heavy text column once novelty has consumed it.
    return df.drop(columns=["prepared_text"]), diagnostics
