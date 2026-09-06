"""
Linguistic novelty: how much did this firm's language change since last quarter?

Cohen, Malloy & Nguyen (2020, JPE) -- "Lazy Prices" -- found that year-over-year
CHANGES in filing language predict returns while language LEVELS do not. The
mechanism is that companies reuse boilerplate by default, so when the wording
moves, something in the business moved with it, and the market underreacts.

Two measures, because they capture different things:

  semantic_novelty  -- embedding distance. Catches meaning changes even when
                       the phrasing is rewritten.
  lexical_novelty   -- TF-IDF cosine distance on raw text. Catches wording
                       changes even when the meaning is unchanged.

A firm can score high on one and low on the other, and the difference is
informative: heavy rewriting with no semantic movement looks like a
communications team at work, while stable phrasing with semantic drift looks
like a real shift being downplayed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _quarter_ordinal(fy: int, fq: int) -> int:
    return int(fy) * 4 + int(fq)


def compute_novelty(
    calls_df: pd.DataFrame,
    embedder,
    text_col: str = "prepared_text",
    ticker_col: str = "ticker",
    fy_col: str = "fiscal_year",
    fq_col: str = "fiscal_quarter",
    require_consecutive: bool = True,
) -> pd.DataFrame:
    """
    Signal component 2.3, computed per firm-quarter against the prior quarter.

    ``require_consecutive`` matters more than it looks. If a firm is missing a
    quarter, comparing across the gap conflates one quarter of drift with two.
    With the flag on, non-consecutive pairs produce NaN rather than a number
    that means something different from its neighbours.

    Returns the input frame with novelty columns appended.
    """
    df = calls_df.copy()
    for col in (ticker_col, fy_col, fq_col, text_col):
        if col not in df.columns:
            raise KeyError(f"missing required column: {col}")

    df["_qord"] = [
        _quarter_ordinal(fy, fq) if pd.notna(fy) and pd.notna(fq) else np.nan
        for fy, fq in zip(df[fy_col], df[fq_col])
    ]
    df = df.sort_values([ticker_col, "_qord"]).reset_index(drop=True)

    texts = df[text_col].fillna("").tolist()
    vecs = embedder.encode(texts) if texts else np.zeros((0, 1))

    tfidf = _tfidf_matrix(texts)

    sem = np.full(len(df), np.nan)
    lex = np.full(len(df), np.nan)

    for _, idx in df.groupby(ticker_col).groups.items():
        idx = list(idx)
        for pos in range(1, len(idx)):
            cur, prev = idx[pos], idx[pos - 1]
            if require_consecutive:
                gap = df.at[cur, "_qord"] - df.at[prev, "_qord"]
                if not (pd.notna(gap) and gap == 1):
                    continue
            if not texts[cur].strip() or not texts[prev].strip():
                continue
            sem[cur] = 1.0 - _cos(vecs[cur], vecs[prev])
            if tfidf is not None:
                lex[cur] = 1.0 - _cos(tfidf[cur], tfidf[prev])

    df["semantic_novelty"] = sem
    df["lexical_novelty"] = lex
    # Positive when wording churned more than meaning did.
    df["novelty_divergence"] = df["lexical_novelty"] - df["semantic_novelty"]
    return df.drop(columns=["_qord"])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(a @ b / (na * nb))


def _tfidf_matrix(texts: list[str]) -> np.ndarray | None:
    """
    TF-IDF over the whole corpus.

    Note the caveat: IDF is fitted on the full sample, which technically uses
    future documents to weight past ones. The effect on a cosine DISTANCE
    between two documents of the same firm is second-order, but it is a real
    look-ahead channel and is recorded in the README rather than ignored. A
    strictly point-in-time version would refit IDF on an expanding window.
    """
    if not texts or all(not t.strip() for t in texts):
        return None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return None
    vec = TfidfVectorizer(
        lowercase=True, stop_words="english", max_features=20000,
        ngram_range=(1, 2), min_df=2,
    )
    try:
        return np.asarray(vec.fit_transform(texts).todense())
    except ValueError:
        return None
