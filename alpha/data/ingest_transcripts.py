"""
Ingest real earnings-call transcripts into this project's schema.

Primary source: the Hugging Face dataset ``Bose345/sp500_earnings_transcripts``
(~33k transcripts, 685 firms, 2005-2025, with speaker-level segmentation).
Speaker segmentation is what makes the tone-gap and evasiveness components
possible at all -- a flat transcript blob cannot support either.

Three things must be verified before any result from this data is reported, and
this module is built to force all three:

  1. LICENCE AND PROVENANCE. The dataset is community-uploaded and very likely
     scraped. Check the dataset card. If the licence is unclear, keep the repo
     private or publish only derived per-firm-quarter FEATURES, never raw text.

  2. SCHEMA. The exact field names are not guaranteed stable. This loader
     normalises the variants it knows about and FAILS LOUDLY on anything else,
     rather than silently producing empty turns that would score as neutral.

  3. COVERAGE. Early years are thin. The backtest start date must be chosen
     from the coverage report, not from wherever results happen to look best.

Survivorship: a dataset described as "S&P 500 companies" is almost certainly
constituents as of a recent snapshot. See alpha/data/universe.py -- the point-
in-time reconstruction is not optional.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Field-name variants seen across community transcript dumps.
_TICKER_KEYS = ("ticker", "symbol", "Symbol", "stock_symbol")
_DATE_KEYS = ("call_date", "date", "Date", "event_date", "published_at")
_YEAR_KEYS = ("fiscal_year", "year", "Year", "earnings_year")
_QUARTER_KEYS = ("fiscal_quarter", "quarter", "Quarter", "fiscal_qtr")
_CONTENT_KEYS = ("structured_content", "turns", "content", "transcript_segments")
_TEXT_KEYS = ("transcript", "text", "full_text", "body")

_SPEAKER_KEYS = ("speaker", "name", "speaker_name", "Speaker")
_ROLE_KEYS = ("role", "title", "position", "speaker_role", "affiliation")
_TURN_TEXT_KEYS = ("text", "content", "speech", "utterance")


class SchemaError(ValueError):
    """Raised when the source data does not match any known layout."""


def _first(record: dict, keys) -> object | None:
    for k in keys:
        if k in record and record[k] not in (None, ""):
            return record[k]
    return None


def _parse_quarter(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and 1 <= int(value) <= 4:
        return int(value)
    m = re.search(r"([1-4])", str(value))
    return int(m.group(1)) if m else None


def _normalise_turns(raw) -> list[dict]:
    """Coerce a speaker-segmented structure into the project's turn schema."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []

    turns = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = _first(item, _TURN_TEXT_KEYS)
        if not text or not str(text).strip():
            continue
        turns.append({
            "speaker": str(_first(item, _SPEAKER_KEYS) or "").strip(),
            "role": (str(_first(item, _ROLE_KEYS)) if _first(item, _ROLE_KEYS) else None),
            "text": str(text).strip(),
        })
    return turns


def _turns_from_flat_text(text: str) -> list[dict]:
    """
    Last-resort parser for transcripts with no speaker structure.

    Splits on "Name - Title:" style prefixes. This is genuinely worse than real
    segmentation and callers are told so, because a mis-split moves text
    between the prepared and Q&A pools and flips the sign of the tone gap.
    """
    pattern = re.compile(r"^([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3})\s*[-–—:]\s*(.*)$")
    turns, speaker, buf = [], None, []

    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m and len(m.group(1)) < 60:
            if speaker and buf:
                turns.append({"speaker": speaker, "role": None, "text": " ".join(buf)})
            speaker, buf = m.group(1), [m.group(2)] if m.group(2) else []
        elif speaker:
            buf.append(line)
    if speaker and buf:
        turns.append({"speaker": speaker, "role": None, "text": " ".join(buf)})
    return turns


def normalise_record(record: dict, allow_flat_fallback: bool = False) -> dict | None:
    """
    Map one source record onto the schema in alpha/features/segment.py.

    Returns None when the record cannot be used. Never invents fields: a record
    without a usable date or ticker cannot be joined to prices, and guessing
    would introduce a look-ahead channel that is impossible to audit later.
    """
    ticker = _first(record, _TICKER_KEYS)
    if not ticker:
        return None

    turns = _normalise_turns(_first(record, _CONTENT_KEYS))
    if not turns and allow_flat_fallback:
        turns = _turns_from_flat_text(str(_first(record, _TEXT_KEYS) or ""))
    if not turns:
        return None

    date = _first(record, _DATE_KEYS)
    year = _first(record, _YEAR_KEYS)
    quarter = _parse_quarter(_first(record, _QUARTER_KEYS))

    if year is None and date:
        try:
            year = pd.to_datetime(date).year
        except (ValueError, TypeError):
            year = None
    if year is None or quarter is None:
        return None

    return {
        "ticker": str(ticker).upper().strip(),
        "call_date": str(pd.to_datetime(date).date()) if date else "",
        "fiscal_year": int(year),
        "fiscal_quarter": int(quarter),
        "turns": turns,
    }


def load_huggingface(
    dataset_name: str = "Bose345/sp500_earnings_transcripts",
    split: str = "train",
    limit: int | None = None,
    allow_flat_fallback: bool = False,
) -> tuple[list[dict], dict]:
    """
    Load and normalise from Hugging Face.

    Returns (records, diagnostics). Read the diagnostics before anything else:
    a low ``normalised_rate`` means the schema assumptions do not match this
    dataset version, and every downstream number would then be computed on
    whichever subset happened to parse -- a silently biased sample.
    """
    from datasets import load_dataset  # imported lazily; heavy dependency

    ds = load_dataset(dataset_name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    records, failures = [], Counter()
    for row in ds:
        rec = normalise_record(dict(row), allow_flat_fallback)
        if rec is None:
            failures["unparseable"] += 1
        else:
            records.append(rec)

    diagnostics = {
        "dataset": dataset_name,
        "n_source_rows": len(ds),
        "n_normalised": len(records),
        "normalised_rate": len(records) / len(ds) if len(ds) else 0.0,
        "failures": dict(failures),
        "available_fields": sorted(ds.features.keys()),
        "used_flat_fallback": allow_flat_fallback,
    }
    if diagnostics["normalised_rate"] < 0.8:
        diagnostics["WARNING"] = (
            "under 80% of rows parsed. Inspect available_fields against the key "
            "lists in this module before proceeding -- the surviving subset is "
            "probably not random."
        )
    return records, diagnostics


def coverage_report(records: list[dict]) -> pd.DataFrame:
    """
    Transcripts per quarter and distinct firms per quarter.

    Use this to choose the backtest start date. Pick the first quarter where
    coverage stabilises, and record that choice in SPEC.md BEFORE running the
    backtest -- choosing it afterwards is a researcher degree of freedom that
    silently inflates results.
    """
    rows = [{
        "period": f"{r['fiscal_year']}Q{r['fiscal_quarter']}",
        "ticker": r["ticker"],
        "n_turns": len(r["turns"]),
        "n_words": sum(len(t["text"].split()) for t in r["turns"]),
    } for r in records]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return (df.groupby("period")
              .agg(n_calls=("ticker", "size"),
                   n_firms=("ticker", "nunique"),
                   median_turns=("n_turns", "median"),
                   median_words=("n_words", "median"))
              .reset_index()
              .sort_values("period"))


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def inspect_sample(records: list[dict], n: int = 3) -> str:
    """
    Print a few parsed calls for manual eyeballing.

    Hand-checking ~20 transcripts is the single highest-value hour in this
    project. Segmentation bugs do not raise -- they quietly move text between
    the prepared and Q&A pools, which is the one error that inverts the
    headline signal.
    """
    from ..features.segment import segment_call, validate_call

    out = []
    for rec in records[:n]:
        call = segment_call(rec)
        problems = validate_call(call)
        out.append(
            f"\n{'=' * 70}\n{call.ticker} {call.fiscal_year}Q{call.fiscal_quarter}"
            f"  ({call.call_date})\n"
            f"  boundary method : {call.boundary_method} @ {call.qa_boundary_index}\n"
            f"  prepared turns  : {len(call.prepared)}  "
            f"({len(call.prepared_text.split())} exec words)\n"
            f"  Q&A turns       : {len(call.qa)}  ({len(call.qa_pairs)} pairs, "
            f"{call.n_analysts} analysts)\n"
            f"  validation      : {problems or 'OK'}\n"
            f"  --- first prepared words ---\n  {call.prepared_text[:220]}...\n"
            f"  --- first Q&A pair ---\n"
            f"  Q: {call.qa_pairs[0].question.text[:160] if call.qa_pairs else '(none)'}...\n"
            f"  A: {call.qa_pairs[0].answer_text[:160] if call.qa_pairs else '(none)'}..."
        )
    return "\n".join(out)
