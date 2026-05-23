import json
from dataclasses import replace
from pathlib import Path

import pytest

from pythonic.data import DECISIONS, abstract, fixtures, load_papers, split_papers


def test_fixtures_preserve_section_identity() -> None:
    papers = fixtures()
    assert {paper.decision for paper in papers} == set(DECISIONS)
    assert all(section.pmid == paper.pmid for paper in papers for section in paper.sections)


def test_splits_are_disjoint_reproducible_and_order_independent() -> None:
    papers = tuple(
        replace(paper, pmid=f"{paper.pmid}-{i}") for i in range(20) for paper in fixtures()
    )
    first = split_papers(papers)
    assert first == split_papers(tuple(reversed(papers)))
    groups = [set(p.pmid for p in group) for group in (first.train, first.validation, first.test)]
    assert sum(map(len, groups)) == len(papers)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert [len(first.train), len(first.validation), len(first.test)] == [42, 9, 9]
    with pytest.raises(ValueError):
        split_papers(papers + papers[:1])


def test_loader_excludes_conclusions_and_prediction_fields(tmp_path: Path) -> None:
    path = tmp_path / "papers.json"
    path.write_text(
        json.dumps(
            {
                "123": {
                    "QUESTION": "Does it work?",
                    "CONTEXTS": ["Observed result", "SECRET"],
                    "LABELS": ["RESULTS", "CONCLUSIONS"],
                    "final_decision": "yes",
                    "LONG_ANSWER": "SECRET",
                    "reasoning_free_pred": "SECRET",
                }
            }
        )
    )
    (paper,) = load_papers(path)
    assert abstract(paper) == "Observed result"
    assert "SECRET" not in repr(paper)
