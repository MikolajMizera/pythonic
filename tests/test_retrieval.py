import pytest

from pythonic.data import Section, fixtures
from pythonic.retrieval import BM25, retrieval_metrics


def test_bm25_finds_source_sections_without_answer_fields() -> None:
    papers = fixtures()
    index = BM25([section for paper in papers for section in paper.sections])
    hit = index.search("treatment fever", limit=1)[0]
    assert hit.section.pmid == "0"
    assert hit.score > 0
    assert not hasattr(hit, "decision")
    assert retrieval_metrics(index, papers, limit=1) == {"recall": 1.0, "mrr": 1.0}
    assert index.search("unmatchedword") == ()


def test_empty_index_and_duplicate_sections() -> None:
    assert BM25([]).search("fever") == ()
    section = Section("1", 0, "RESULTS", "")
    assert BM25([section]).search("fever") == ()
    with pytest.raises(ValueError):
        BM25([section, section])
