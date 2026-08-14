from collections.abc import Sequence

import numpy as np

from pythonic.data import fixtures
from pythonic.dense import DenseIndex, Embeddings, HybridIndex
from pythonic.retrieval import BM25


def fixture_embeddings(texts: Sequence[str]) -> Embeddings:
    return np.array(
        [[text.lower().count(word) for word in ("fever", "fatigue", "recovery")] for text in texts],
        dtype=np.float32,
    ).reshape(-1, 3)


def test_dense_and_hybrid_preserve_source_identity() -> None:
    sections = [section for paper in fixtures() for section in paper.sections]
    dense = DenseIndex(sections, fixture_embeddings)
    hybrid = HybridIndex(BM25(sections), dense)
    assert dense.search("fever")[0].section.pmid == "0"
    assert hybrid.search("fever")[0].section.pmid == "0"
    assert len({(hit.section.pmid, hit.section.index) for hit in hybrid.search("fever")}) == len(
        hybrid.search("fever")
    )
    assert dense.search("unknown") == ()
    assert DenseIndex([], fixture_embeddings).search("fever") == ()
