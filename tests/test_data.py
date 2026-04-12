from pythonic.data import DECISIONS, fixtures


def test_fixtures_preserve_section_identity() -> None:
    papers = fixtures()
    assert {paper.decision for paper in papers} == set(DECISIONS)
    assert all(section.pmid == paper.pmid for paper in papers for section in paper.sections)
