"""
Tone scoring: Loughran-McDonald dictionary plus an optional neural scorer.

Both are computed and both are reported. Loughran & McDonald (2011) showed
general-purpose sentiment dictionaries misclassify finance vocabulary --
"liability", "cost", "capital" are negative in Harvard IV and neutral in a
10-K. The LM lists exist precisely to fix that, and they have the advantage of
being fully interpretable: any score can be traced to the exact words that
produced it.

The neural scorer (FinBERT) is stronger on context but is a black box. Where
the two agree, the result is more credible. Where they disagree, that is itself
worth investigating rather than a reason to quietly keep the better-performing
one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_TOKEN_RE = re.compile(r"[a-z][a-z'-]+")

# Fallback seed lists. The real run should load the full LM master dictionary
# (~4k words) from data/lm_dictionary.csv -- see load_lm_dictionary. These
# seeds exist so the pipeline is testable offline and so a missing data file
# degrades to something honest rather than to zeros.
_SEED_NEGATIVE = {
    "loss", "losses", "decline", "declined", "declining", "weak", "weakness",
    "adverse", "adversely", "difficult", "challenging", "shortfall", "impairment",
    "litigation", "restructuring", "downturn", "deteriorate", "deteriorated",
    "disappointing", "unfavorable", "delay", "delayed", "concern", "concerns",
    "risk", "risks", "failure", "failed", "against", "negative", "shortfalls",
    "pressure", "headwind", "headwinds", "miss", "missed", "cut", "reduction",
}
_SEED_POSITIVE = {
    "gain", "gains", "growth", "growing", "strong", "strength", "improve",
    "improved", "improvement", "favorable", "opportunity", "opportunities",
    "success", "successful", "profitable", "outperform", "outperformed",
    "record", "exceeded", "exceeding", "robust", "solid", "momentum",
    "confident", "pleased", "excellent", "positive", "tailwind", "accelerate",
}
_SEED_UNCERTAINTY = {
    "approximately", "uncertain", "uncertainty", "uncertainties", "may",
    "might", "could", "possibly", "perhaps", "risk", "risks", "assume",
    "assumption", "assumptions", "believe", "believes", "estimate", "estimates",
    "fluctuate", "indefinite", "predict", "probable", "roughly", "seems",
    "somewhat", "tentative", "unclear", "unknown", "unpredictable", "variable",
    "volatile", "volatility", "depend", "depends", "depending", "contingent",
}
_SEED_WEAK_MODAL = {
    "may", "might", "could", "possibly", "perhaps", "appears", "seems",
    "suggest", "suggests", "somewhat", "sometimes", "occasionally", "nearly",
    "almost", "generally", "typically", "usually", "likely", "hopefully",
}


@dataclass
class LMDictionary:
    negative: set[str]
    positive: set[str]
    uncertainty: set[str]
    weak_modal: set[str]
    source: str

    @property
    def is_seed(self) -> bool:
        return self.source == "seed"


@lru_cache(maxsize=1)
def load_lm_dictionary(path: str | None = None) -> LMDictionary:
    """
    Load the Loughran-McDonald master dictionary if present, else seed lists.

    The real file (LoughranMcDonald_MasterDictionary_*.csv) is freely available
    from the Notre Dame Software Repository for Accounting and Finance. It is
    not vendored here because of its licence terms; download it into
    ``data/lm_dictionary.csv``.

    When the seed lists are used, ``LMDictionary.is_seed`` is True and callers
    surface that in the results, so a run made without the real dictionary is
    never mistaken for one made with it.
    """
    candidates = [Path(path)] if path else [
        Path("data/lm_dictionary.csv"),
        Path(__file__).resolve().parents[2] / "data" / "lm_dictionary.csv",
    ]
    for p in candidates:
        if p and p.exists():
            import csv

            neg, pos, unc, weak = set(), set(), set(), set()
            with open(p, newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    w = (row.get("Word") or row.get("word") or "").strip().lower()
                    if not w:
                        continue
                    def _hit(col: str) -> bool:
                        v = row.get(col) or row.get(col.lower()) or "0"
                        try:
                            return int(float(v)) != 0
                        except ValueError:
                            return False
                    if _hit("Negative"):
                        neg.add(w)
                    if _hit("Positive"):
                        pos.add(w)
                    if _hit("Uncertainty"):
                        unc.add(w)
                    if _hit("Weak_Modal"):
                        weak.add(w)
            return LMDictionary(neg, pos, unc, weak, source=str(p))
    return LMDictionary(
        _SEED_NEGATIVE, _SEED_POSITIVE, _SEED_UNCERTAINTY, _SEED_WEAK_MODAL, source="seed"
    )


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def lm_tone(text: str, dictionary: LMDictionary | None = None) -> dict[str, float]:
    """
    Dictionary tone scores, all normalised by token count.

    ``tone`` is the standard (pos - neg) / (pos + neg) polarity, which is scale
    free and does not reward verbosity. Returns NaN for empty text rather than
    0.0, because "no words" and "perfectly neutral" are different states and
    collapsing them biases the cross-section.
    """
    d = dictionary or load_lm_dictionary()
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return {k: np.nan for k in ("tone", "neg_rate", "pos_rate", "uncertainty_rate", "weak_modal_rate", "n_words")}

    neg = sum(1 for t in toks if t in d.negative)
    pos = sum(1 for t in toks if t in d.positive)
    unc = sum(1 for t in toks if t in d.uncertainty)
    weak = sum(1 for t in toks if t in d.weak_modal)

    polarity = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0
    return {
        "tone": float(polarity),
        "neg_rate": neg / n,
        "pos_rate": pos / n,
        "uncertainty_rate": unc / n,
        "weak_modal_rate": weak / n,
        "n_words": float(n),
    }


class FinBertScorer:
    """
    Optional neural tone scorer.

    Lazy-loaded so the pipeline runs end-to-end without downloading a model,
    and so tests stay offline. Chunks long text to the model's window and
    averages, since an earnings call vastly exceeds 512 tokens.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from transformers import pipeline  # imported lazily on purpose

            self._pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device,
                truncation=True,
                max_length=512,
            )
        return self._pipe

    def score(self, text: str, chunk_words: int = 320) -> float:
        """Signed tone in [-1, 1]: mean over chunks of P(positive) - P(negative)."""
        if not text or not text.strip():
            return np.nan
        words = text.split()
        chunks = [
            " ".join(words[i : i + chunk_words]) for i in range(0, len(words), chunk_words)
        ] or [text]
        pipe = self._ensure()
        scores = []
        for out in pipe(chunks, top_k=None):
            probs = {d["label"].lower(): d["score"] for d in out}
            scores.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
        return float(np.mean(scores)) if scores else np.nan


def tone_gap_features(prepared_text: str, answer_text: str, question_text: str = "",
                      dictionary: LMDictionary | None = None) -> dict[str, float]:
    """
    Signal components 2.1 (tone gap), 2.4 (analyst pressure), 2.5 (hedging).

    The gap is the point: prepared remarks are lawyered and fully controlled,
    Q&A answers are produced under live analyst pressure. A large positive gap
    means management is projecting confidence in the script that it cannot
    sustain when questioned.

    Hedging is measured on answers specifically, not the whole call -- hedging
    in a scripted section is legal boilerplate, hedging under questioning is
    informative.
    """
    d = dictionary or load_lm_dictionary()
    prep = lm_tone(prepared_text, d)
    ans = lm_tone(answer_text, d)
    q = lm_tone(question_text, d) if question_text else {}

    return {
        "lm_tone_prepared": prep["tone"],
        "lm_tone_answers": ans["tone"],
        "lm_tone_gap": prep["tone"] - ans["tone"],
        "hedging_uncertainty": ans["uncertainty_rate"],
        "hedging_weak_modal": ans["weak_modal_rate"],
        "hedging_density": ans["uncertainty_rate"] + ans["weak_modal_rate"],
        "analyst_tone": q.get("tone", np.nan),
        "prepared_words": prep["n_words"],
        "answer_words": ans["n_words"],
    }
