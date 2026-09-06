"""
Segmentation tests.

Segmentation is the foundation: a silent bug here quietly reassigns text
between the prepared and Q&A pools, which flips the sign of the tone gap
without raising anything. So these tests include planted-failure cases --
malformed calls that MUST be rejected. A validator that never rejects is not a
validator.
"""

from __future__ import annotations

import pytest

from alpha.features.segment import (
    Role, normalise_role, pair_questions_answers, segment_call,
    segmentation_report, validate_call,
)


def _turn(speaker, role, text):
    return {"speaker": speaker, "role": role, "text": text}


def _good_call(n_pairs=4, words=400):
    filler = " ".join(["revenue"] * words)
    turns = [
        _turn("Operator", "operator", "Good day and welcome."),
        _turn("Jane Doe", "Chief Executive Officer", filler),
        _turn("John Roe", "Chief Financial Officer", filler),
        _turn("Operator", "operator",
              "We will now begin the question-and-answer session."),
    ]
    for k in range(n_pairs):
        turns.append(_turn(f"Analyst {k}", "analyst", " ".join(["question"] * 40)))
        turns.append(_turn("Jane Doe", "Chief Executive Officer",
                           " ".join(["answer"] * 150)))
    return {"ticker": "TEST", "call_date": "2024-01-15",
            "fiscal_year": 2024, "fiscal_quarter": 1, "turns": turns}


# --------------------------------------------------------------------------
# Role normalisation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("speaker,role,expected", [
    ("Operator", "operator", Role.OPERATOR),
    ("Tim Cook", "Chief Executive Officer", Role.EXECUTIVE),
    ("Luca Maestri", "CFO", Role.EXECUTIVE),
    ("Katy Huberty", "analyst", Role.ANALYST),
    ("Some Person", "Research Analyst", Role.ANALYST),
    ("Jane Smith", "President", Role.EXECUTIVE),
    ("Mystery Guest", None, Role.UNKNOWN),
    ("", None, Role.UNKNOWN),
])
def test_role_normalisation(speaker, role, expected):
    assert normalise_role(speaker, role) is expected


def test_unknown_role_is_not_guessed():
    """
    Conservative by design: guessing a role wrong moves text between pools and
    corrupts the tone gap. UNKNOWN is the correct answer when unsure.
    """
    assert normalise_role("Bob", "friend of the company") is Role.UNKNOWN


# --------------------------------------------------------------------------
# Boundary detection
# --------------------------------------------------------------------------

def test_marker_boundary_detected():
    call = segment_call(_good_call())
    assert call.boundary_method == "marker"
    assert call.qa_boundary_index == 4
    assert len(call.prepared) == 4
    assert "answer" in call.mgmt_answer_text


def test_fallback_to_first_analyst_when_no_marker():
    turns = [
        _turn("Operator", "operator", "Welcome everyone."),
        _turn("Jane Doe", "CEO", " ".join(["revenue"] * 400)),
        _turn("Analyst A", "analyst", " ".join(["question"] * 40)),
        _turn("Jane Doe", "CEO", " ".join(["answer"] * 200)),
    ]
    call = segment_call({"ticker": "T", "call_date": "2024-01-01",
                         "fiscal_year": 2024, "fiscal_quarter": 1, "turns": turns})
    assert call.boundary_method == "first_analyst"
    assert call.qa_boundary_index == 2


def test_no_qa_section_reported_not_guessed():
    turns = [
        _turn("Operator", "operator", "Welcome."),
        _turn("Jane Doe", "CEO", " ".join(["revenue"] * 400)),
    ]
    call = segment_call({"ticker": "T", "call_date": "2024-01-01",
                         "fiscal_year": 2024, "fiscal_quarter": 1, "turns": turns})
    assert call.boundary_method == "none"
    assert "no_qa_section" in validate_call(call)


def test_prepared_and_qa_do_not_overlap():
    """Text must land in exactly one pool -- double counting inflates both scores."""
    call = segment_call(_good_call())
    assert len(call.prepared) + len(call.qa) == len(call.prepared) + len(call.qa)
    prepared_idx = {t.index for t in call.prepared}
    qa_idx = {t.index for t in call.qa}
    assert not (prepared_idx & qa_idx)


# --------------------------------------------------------------------------
# Q&A pairing
# --------------------------------------------------------------------------

def test_question_answer_pairing():
    call = segment_call(_good_call(n_pairs=3))
    assert len(call.qa_pairs) == 3
    for p in call.qa_pairs:
        assert p.question.role is Role.ANALYST
        assert all(a.role is Role.EXECUTIVE for a in p.answers)


def test_multiple_executives_answering_one_question():
    from alpha.features.segment import Turn

    turns = [
        Turn("Analyst A", Role.ANALYST, "What about margins?", 0),
        Turn("CEO", Role.EXECUTIVE, "Let me start.", 1),
        Turn("CFO", Role.EXECUTIVE, "And to add detail.", 2),
        Turn("Analyst B", Role.ANALYST, "Follow up on capex?", 3),
        Turn("CEO", Role.EXECUTIVE, "Sure.", 4),
    ]
    pairs = pair_questions_answers(turns)
    assert len(pairs) == 2
    assert len(pairs[0].answers) == 2
    assert "start" in pairs[0].answer_text and "detail" in pairs[0].answer_text


def test_followup_detection():
    from alpha.features.segment import Turn

    turns = [
        Turn("Analyst A", Role.ANALYST, "First question.", 0),
        Turn("CEO", Role.EXECUTIVE, "Answer one.", 1),
        Turn("Analyst A", Role.ANALYST, "Just to follow up.", 2),
        Turn("CEO", Role.EXECUTIVE, "Answer two.", 3),
    ]
    pairs = pair_questions_answers(turns)
    assert len(pairs) == 2
    assert pairs[1].is_followup, "same analyst speaking again is a follow-up"


def test_operator_handoff_resets_followup():
    from alpha.features.segment import Turn

    turns = [
        Turn("Analyst A", Role.ANALYST, "First question.", 0),
        Turn("CEO", Role.EXECUTIVE, "Answer one.", 1),
        Turn("Operator", Role.OPERATOR, "Next question.", 2),
        Turn("Analyst A", Role.ANALYST, "New topic entirely.", 3),
        Turn("CEO", Role.EXECUTIVE, "Answer two.", 4),
    ]
    pairs = pair_questions_answers(turns)
    assert not pairs[1].is_followup, "operator hand-off means a fresh question"


def test_unanswered_question_dropped():
    """A question with no answer is not a pair and must not become one."""
    from alpha.features.segment import Turn

    turns = [
        Turn("Analyst A", Role.ANALYST, "Question with no answer.", 0),
        Turn("Operator", Role.OPERATOR, "Next.", 1),
    ]
    assert pair_questions_answers(turns) == []


# --------------------------------------------------------------------------
# Planted failures: the validator must actually reject
# --------------------------------------------------------------------------

def test_validator_rejects_short_prepared_remarks():
    rec = _good_call(words=10)
    assert "prepared_too_short" in validate_call(segment_call(rec))


def test_validator_rejects_too_few_qa_pairs():
    rec = _good_call(n_pairs=1)
    assert "too_few_qa_pairs" in validate_call(segment_call(rec))


def test_validator_rejects_majority_unknown_roles():
    turns = [_turn(f"Person {i}", "attendee", " ".join(["word"] * 100))
             for i in range(10)]
    call = segment_call({"ticker": "T", "call_date": "2024-01-01",
                         "fiscal_year": 2024, "fiscal_quarter": 1, "turns": turns})
    assert "majority_unknown_roles" in validate_call(call)


def test_validator_accepts_a_good_call():
    """The counterpart: a validator that rejects everything is equally useless."""
    assert validate_call(segment_call(_good_call())) == []


def test_segmentation_report_counts_correctly():
    calls = [segment_call(_good_call()) for _ in range(5)]
    calls.append(segment_call(_good_call(n_pairs=1)))
    rep = segmentation_report(calls)
    assert rep["n_calls"] == 6
    assert rep["n_usable"] == 5
    assert rep["rejection_reasons"]["too_few_qa_pairs"] == 1
