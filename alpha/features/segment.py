"""
Transcript segmentation: the foundation everything else stands on.

A silent bug here corrupts every downstream number without ever raising, so
this module is written defensively and is the most heavily tested part of the
project. Segmentation is validated before any feature is computed
(see :func:`validate_call`), and the pipeline refuses to score a call that
fails validation rather than emitting a quietly wrong feature.

Expected input schema (normalise to this in alpha/data/ingest_transcripts.py):

    {
      "ticker": "AAPL",
      "call_date": "2023-11-02",
      "fiscal_year": 2023,
      "fiscal_quarter": 4,
      "turns": [
          {"speaker": "Operator",   "role": "operator",   "text": "..."},
          {"speaker": "Tim Cook",   "role": "executive",  "text": "..."},
          {"speaker": "Katy Huberty","role": "analyst",   "text": "..."},
      ]
    }

The `role` field is what the whole module keys on. Community transcript
datasets label roles inconsistently, so :func:`normalise_role` handles the
common variants and anything unrecognised becomes UNKNOWN rather than being
guessed into a category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    OPERATOR = "operator"
    EXECUTIVE = "executive"
    ANALYST = "analyst"
    UNKNOWN = "unknown"


_EXEC_PAT = re.compile(
    r"\b(ceo|cfo|coo|cto|chief|president|chairman|founder|vp|vice president|"
    r"executive|officer|head of|treasurer|controller|director of)\b",
    re.I,
)
_ANALYST_PAT = re.compile(r"\b(analyst|research|equity|securities)\b", re.I)
_OPERATOR_PAT = re.compile(r"\boperator\b", re.I)

# Phrases that open the Q&A section. Ordered most- to least-specific.
_QA_MARKERS = [
    r"we (?:will|'ll) now (?:begin|open|take).{0,40}(?:question|q\s*&\s*a)",
    r"(?:begin|open|start).{0,30}question[- ]and[- ]answer",
    r"question[- ]and[- ]answer session",
    r"(?:first|our first) question comes from",
    r"the floor is (?:now )?open (?:for|to) questions",
    r"\bq\s*&\s*a\s+(?:session|portion)\b",
    r"ladies and gentlemen.{0,60}question",
]
_QA_RE = re.compile("|".join(f"(?:{m})" for m in _QA_MARKERS), re.I | re.S)


@dataclass
class Turn:
    speaker: str
    role: Role
    text: str
    index: int

    @property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass
class QAPair:
    """One analyst question with the management response that followed it."""

    question: Turn
    answers: list[Turn]
    is_followup: bool = False

    @property
    def answer_text(self) -> str:
        return " ".join(a.text for a in self.answers)


@dataclass
class SegmentedCall:
    ticker: str
    call_date: str
    fiscal_year: int | None
    fiscal_quarter: int | None
    prepared: list[Turn] = field(default_factory=list)
    qa: list[Turn] = field(default_factory=list)
    qa_pairs: list[QAPair] = field(default_factory=list)
    qa_boundary_index: int | None = None
    boundary_method: str = "none"

    @property
    def key(self) -> str:
        return f"{self.ticker}:{self.fiscal_year}Q{self.fiscal_quarter}"

    @property
    def prepared_text(self) -> str:
        return " ".join(t.text for t in self.prepared if t.role is Role.EXECUTIVE)

    @property
    def mgmt_answer_text(self) -> str:
        return " ".join(t.text for t in self.qa if t.role is Role.EXECUTIVE)

    @property
    def analyst_question_text(self) -> str:
        return " ".join(t.text for t in self.qa if t.role is Role.ANALYST)

    @property
    def n_analysts(self) -> int:
        return len({t.speaker for t in self.qa if t.role is Role.ANALYST})


class SegmentationError(ValueError):
    """Raised when a transcript cannot be segmented reliably enough to score."""


def normalise_role(speaker: str, raw_role: str | None = None) -> Role:
    """
    Map a free-text speaker/role label onto a Role.

    Unrecognised labels return UNKNOWN. Deliberately conservative: guessing a
    role wrong silently reassigns text between the prepared and Q&A pools,
    which flips the sign of the tone gap.
    """
    blob = f"{raw_role or ''} {speaker or ''}".strip()
    if not blob:
        return Role.UNKNOWN
    if _OPERATOR_PAT.search(blob):
        return Role.OPERATOR
    low = (raw_role or "").strip().lower()
    if low in {"executive", "management", "company", "corporate"}:
        return Role.EXECUTIVE
    if low in {"analyst", "sell-side", "sellside"}:
        return Role.ANALYST
    if _EXEC_PAT.search(blob):
        return Role.EXECUTIVE
    if _ANALYST_PAT.search(blob):
        return Role.ANALYST
    return Role.UNKNOWN


def find_qa_boundary(turns: list[Turn]) -> tuple[int | None, str]:
    """
    Locate the first Q&A turn.

    Two strategies, tried in order:
      1. Marker phrase, usually spoken by the operator.
      2. First analyst turn, as a fallback when no marker is present.

    Returns (index, method). ``(None, "none")`` means the call has no
    detectable Q&A section, which is legitimate for some smaller-cap calls and
    means the call is dropped rather than scored on a bad split.
    """
    for t in turns:
        if _QA_RE.search(t.text):
            # The marker itself belongs to the prepared section; Q&A starts next.
            return min(t.index + 1, len(turns)), "marker"
    for t in turns:
        if t.role is Role.ANALYST:
            return t.index, "first_analyst"
    return None, "none"


def pair_questions_answers(qa_turns: list[Turn]) -> list[QAPair]:
    """
    Group Q&A turns into (analyst question, management answers) pairs.

    An analyst turn opens a pair; every executive turn until the next analyst
    turn is part of the answer. Operator turns are skipped -- they carry the
    call-handling boilerplate and no informational content.

    A pair is marked ``is_followup`` when the same analyst speaks again without
    an intervening operator hand-off. That matters on its own: a follow-up is
    evidence the first answer did not satisfy the questioner.
    """
    pairs: list[QAPair] = []
    current: QAPair | None = None
    last_analyst: str | None = None
    saw_operator_since_question = False

    for turn in qa_turns:
        if turn.role is Role.ANALYST:
            if current is not None and current.answers:
                pairs.append(current)
            is_fu = (
                last_analyst is not None
                and turn.speaker == last_analyst
                and not saw_operator_since_question
            )
            current = QAPair(question=turn, answers=[], is_followup=is_fu)
            last_analyst = turn.speaker
            saw_operator_since_question = False
        elif turn.role is Role.EXECUTIVE and current is not None:
            current.answers.append(turn)
        elif turn.role is Role.OPERATOR:
            saw_operator_since_question = True

    if current is not None and current.answers:
        pairs.append(current)
    return pairs


def segment_call(record: dict) -> SegmentedCall:
    """Turn one raw transcript record into a :class:`SegmentedCall`."""
    raw_turns = record.get("turns") or []
    if not raw_turns:
        raise SegmentationError(f"{record.get('ticker')}: no turns in record")

    turns = [
        Turn(
            speaker=(rt.get("speaker") or "").strip(),
            role=normalise_role(rt.get("speaker", ""), rt.get("role")),
            text=(rt.get("text") or "").strip(),
            index=i,
        )
        for i, rt in enumerate(raw_turns)
    ]

    boundary, method = find_qa_boundary(turns)
    prepared = turns if boundary is None else turns[:boundary]
    qa = [] if boundary is None else turns[boundary:]

    return SegmentedCall(
        ticker=record.get("ticker", ""),
        call_date=str(record.get("call_date", "")),
        fiscal_year=record.get("fiscal_year"),
        fiscal_quarter=record.get("fiscal_quarter"),
        prepared=prepared,
        qa=qa,
        qa_pairs=pair_questions_answers(qa),
        qa_boundary_index=boundary,
        boundary_method=method,
    )


# Minimum content for a call to be scoreable. Set from the coverage report
# rather than by intuition -- see scripts/01_ingest.py.
MIN_PREPARED_WORDS = 200
MIN_ANSWER_WORDS = 200
MIN_QA_PAIRS = 2


def validate_call(call: SegmentedCall) -> list[str]:
    """
    Return a list of reasons this call should not be scored. Empty means usable.

    Called before feature computation so that a malformed transcript is dropped
    loudly and counted, rather than producing a feature value that looks fine.
    """
    problems: list[str] = []
    if call.qa_boundary_index is None:
        problems.append("no_qa_section")
    if len(call.prepared_text.split()) < MIN_PREPARED_WORDS:
        problems.append("prepared_too_short")
    if len(call.mgmt_answer_text.split()) < MIN_ANSWER_WORDS:
        problems.append("answers_too_short")
    if len(call.qa_pairs) < MIN_QA_PAIRS:
        problems.append("too_few_qa_pairs")
    unknown = sum(1 for t in call.prepared + call.qa if t.role is Role.UNKNOWN)
    total = len(call.prepared) + len(call.qa)
    if total and unknown / total > 0.5:
        problems.append("majority_unknown_roles")
    return problems


def segmentation_report(calls: list[SegmentedCall]) -> dict:
    """Aggregate diagnostics. Print this before trusting any downstream result."""
    from collections import Counter

    reasons = Counter()
    methods = Counter()
    usable = 0
    for c in calls:
        methods[c.boundary_method] += 1
        probs = validate_call(c)
        if probs:
            for p in probs:
                reasons[p] += 1
        else:
            usable += 1
    return {
        "n_calls": len(calls),
        "n_usable": usable,
        "usable_rate": usable / len(calls) if calls else 0.0,
        "boundary_methods": dict(methods),
        "rejection_reasons": dict(reasons),
    }
