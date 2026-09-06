"""
Evasiveness: does the answer actually address the question that was asked?

This is the component least likely to appear in a competing candidate's
project, and the one that needs real NLP work rather than a library call --
speaker attribution, turn pairing, and a defensible embedding choice all have
to be right before the number means anything.

Construct: for each (analyst question, management answer) pair, embed both and
take the cosine similarity. Low similarity means the answer drifted from the
question. Averaged over the call, this is an evasiveness score.

The embedding model is held FIXED across the entire sample. Choosing a model
because it backtests well is look-ahead bias laundered through model selection,
and it is very easy to do by accident.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    """Anything that turns a list of strings into a (n, d) array."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """
    Deterministic bag-of-words embedder with no downloads and no model risk.

    This exists so the pipeline is testable offline and so results are exactly
    reproducible on any machine. It captures lexical overlap only -- it cannot
    tell that "margin headwinds" and "cost pressure" are related -- so it is a
    LOWER BOUND on what the semantic component can detect, not a substitute for
    a real encoder.

    Report which embedder produced a result. A number from this one and a
    number from a sentence-transformer are not comparable.
    """

    name = "hashing"

    def __init__(self, dim: int = 512, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def _hash(self, token: str) -> int:
        h = hashlib.blake2b(f"{self.seed}:{token}".encode(), digest_size=8)
        return int.from_bytes(h.digest(), "big") % self.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import re

        out = np.zeros((len(texts), self.dim), dtype=float)
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z][a-z'-]+", (text or "").lower()):
                out[i, self._hash(tok)] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)


class SentenceTransformerEmbedder:
    """
    Real semantic encoder. Lazy-loaded so importing this module stays cheap.

    all-MiniLM-L6-v2 is the default: small, fast, and — importantly for a
    20-year backtest — trained on general text rather than on anything
    financial and post-dating the sample. A finance-specific encoder trained on
    data overlapping the backtest window would leak.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str | None = None, batch_size: int = 64):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def name(self) -> str:
        return self.model_name

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure()
        return np.asarray(
            model.encode(list(texts), batch_size=self.batch_size,
                         normalize_embeddings=True, show_progress_bar=False),
            dtype=float,
        )


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equally-shaped matrices."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    dots = np.einsum("ij,ij->i", a, b)
    return np.divide(dots, denom, out=np.full(len(dots), np.nan), where=denom > 0)


def evasiveness_features(qa_pairs, embedder: Embedder, min_words: int = 5) -> dict[str, float]:
    """
    Signal component 2.2.

    Pairs whose question or answer is shorter than ``min_words`` are dropped --
    "Thanks, that's helpful" is not a question and its similarity score is
    noise that would dilute the measure.

    Returns NaN (not 0) when nothing scoreable remains, so that a call with no
    usable pairs is excluded downstream instead of being ranked as perfectly
    responsive.
    """
    usable = [
        p for p in qa_pairs
        if len(p.question.text.split()) >= min_words
        and len(p.answer_text.split()) >= min_words
    ]
    if not usable:
        return {
            "evasiveness": np.nan, "qa_alignment_mean": np.nan,
            "qa_alignment_min": np.nan, "qa_alignment_std": np.nan,
            "n_qa_pairs_scored": 0.0, "followup_rate": np.nan,
        }

    questions = [p.question.text for p in usable]
    answers = [p.answer_text for p in usable]
    qv = embedder.encode(questions)
    av = embedder.encode(answers)
    sims = cosine_rows(qv, av)
    sims = sims[np.isfinite(sims)]
    if len(sims) == 0:
        return {
            "evasiveness": np.nan, "qa_alignment_mean": np.nan,
            "qa_alignment_min": np.nan, "qa_alignment_std": np.nan,
            "n_qa_pairs_scored": 0.0, "followup_rate": np.nan,
        }

    n_followup = sum(1 for p in qa_pairs if getattr(p, "is_followup", False))
    return {
        "evasiveness": float(1.0 - sims.mean()),
        "qa_alignment_mean": float(sims.mean()),
        "qa_alignment_min": float(sims.min()),
        "qa_alignment_std": float(sims.std(ddof=1)) if len(sims) > 1 else 0.0,
        "n_qa_pairs_scored": float(len(sims)),
        # A follow-up is evidence the first answer did not satisfy the analyst.
        "followup_rate": float(n_followup / len(qa_pairs)) if qa_pairs else np.nan,
    }


def sanity_check_embedder(embedder: Embedder) -> dict[str, float]:
    """
    Verify the embedder separates related from unrelated text before trusting it.

    Same control structure used elsewhere in this project: a measurement device
    gets validated against known-answer cases first. Self-similarity should be
    ~1.0 and unrelated-pair similarity should be clearly lower. If that gap is
    small, every evasiveness number downstream is noise.
    """
    a = "Our gross margin expanded 120 basis points on favourable product mix."
    a2 = "Gross margins improved by roughly 1.2 points, helped by mix."
    b = "The board approved a new share repurchase authorisation of two billion."

    v = embedder.encode([a, a2, b])
    self_sim = float(cosine_rows(v[0:1], v[0:1])[0])
    related = float(cosine_rows(v[0:1], v[1:2])[0])
    unrelated = float(cosine_rows(v[0:1], v[2:3])[0])
    return {
        "self_similarity": self_sim,
        "related_similarity": related,
        "unrelated_similarity": unrelated,
        "discrimination_gap": related - unrelated,
    }
