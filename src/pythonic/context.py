from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from pythonic.biomedical import EvidencePrediction
from pythonic.data import DECISIONS, Decision, Section
from pythonic.retrieval import Hit


@dataclass(frozen=True)
class Citation:
    pmid: str
    section: int
    quote: str


@dataclass(frozen=True)
class Context:
    question: str
    text: str
    citations: tuple[Citation, ...]
    tokens: int


@dataclass(frozen=True)
class EvidenceAnswer:
    decision: Decision | None
    confidence: float | None
    citations: tuple[Citation, ...]
    reason: str | None = None


def select_context(
    question: str,
    hits: Sequence[Hit],
    budget: int,
    count: Callable[[str], int],
) -> Context:
    reserved = count(question) + 3
    if budget < reserved:
        raise ValueError("Question and special tokens exceed the budget.")
    passages: list[str] = []
    citations = []
    seen = set()
    for hit in hits:
        section = hit.section
        identity = section.pmid, section.index
        if identity in seen or not section.text.strip():
            continue
        seen.add(identity)
        prefix = "\n".join(passages) + ("\n" if passages else "")
        text = section.text
        if count(prefix + text) + reserved > budget:
            low, high = 0, len(text)
            while low < high:
                middle = (low + high + 1) // 2
                if count(prefix + text[:middle]) + reserved <= budget:
                    low = middle
                else:
                    high = middle - 1
            text = text[:low]
        if text.strip() and count(prefix + text) + reserved <= budget:
            passages.append(text)
            citations.append(Citation(section.pmid, section.index, text))
    combined = "\n".join(passages)
    return Context(question, combined, tuple(citations), reserved + count(combined))


def validate_citations(
    citations: Sequence[Citation], sources: Mapping[tuple[str, int], Section]
) -> None:
    for citation in citations:
        section = sources.get((citation.pmid, citation.section))
        if section is None or not citation.quote.strip() or citation.quote not in section.text:
            raise ValueError("Citation does not match its source section.")


def answer_question(
    question: str,
    hits: Sequence[Hit],
    predict: Callable[[str, str], EvidencePrediction],
    budget: int,
    count: Callable[[str], int],
) -> EvidenceAnswer:
    context = select_context(question, hits, budget, count)
    if not context.citations:
        return EvidenceAnswer(None, None, (), "No evidence fits the budget.")
    result = predict(question, context.text)
    if result.decision not in DECISIONS or not 0 <= result.confidence <= 1:
        raise ValueError("Classification decision or confidence is invalid.")
    return EvidenceAnswer(result.decision, result.confidence, context.citations)
