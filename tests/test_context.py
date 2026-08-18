import pytest

from pythonic.biomedical import EvidencePrediction
from pythonic.context import Citation, answer_question, select_context, validate_citations
from pythonic.data import Section, fixtures
from pythonic.retrieval import BM25, Hit


def test_context_budget_truncates_without_losing_provenance() -> None:
    section = Section("123", 0, "RESULTS", "A long passage about the trial results.")
    context = select_context("Trial?", [Hit(section, 1), Hit(section, 1)], budget=25, count=len)
    assert context.tokens <= 25
    assert len(context.citations) == 1
    assert context.text == section.text[:16]
    validate_citations(context.citations, {("123", 0): section})
    with pytest.raises(ValueError):
        validate_citations([Citation("123", 0, "Fabricated")], {("123", 0): section})


def test_missing_evidence_abstains_without_calling_classifier() -> None:
    def predict(question: str, context: str) -> EvidencePrediction:
        raise AssertionError("Classifier must not run without evidence.")

    answer = answer_question("Trial?", [], predict, budget=25, count=len)
    assert answer.decision is None and answer.citations == ()
    with pytest.raises(ValueError):
        select_context("Trial?", [], budget=2, count=len)


def test_answer_cites_actual_retrieved_text() -> None:
    sections = [section for paper in fixtures() for section in paper.sections]
    answer = answer_question(
        "fever",
        BM25(sections).search("fever"),
        lambda q, c: EvidencePrediction("yes", 0.7, len(c)),
        budget=100,
        count=len,
    )
    assert answer.decision == "yes"
    validate_citations(answer.citations, {(s.pmid, s.index): s for s in sections})
