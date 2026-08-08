import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pythonic.data import Paper, Section


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


@dataclass(frozen=True)
class Hit:
    section: Section
    score: float


class Retriever(Protocol):
    def search(self, query: str, limit: int = 5) -> tuple[Hit, ...]: ...


class BM25:
    def __init__(self, sections: Sequence[Section], k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 needs positive k1 and b between zero and one.")
        self.sections = tuple(sections)
        identities = {(section.pmid, section.index) for section in self.sections}
        if len(identities) != len(self.sections):
            raise ValueError("Section identities must be unique.")
        self.k1, self.b = k1, b
        self.counts = [Counter(words(section.text)) for section in self.sections]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = sum(self.lengths) / len(self.lengths) if self.lengths else 0
        self.frequency = Counter(token for count in self.counts for token in count)

    def search(self, query: str, limit: int = 5) -> tuple[Hit, ...]:
        if limit <= 0:
            raise ValueError("Search limit must be positive.")
        if not self.average:
            return ()
        tokens = words(query)
        hits = []
        for section, count, length in zip(self.sections, self.counts, self.lengths, strict=True):
            score = 0.0
            for token in tokens:
                frequency = count[token]
                if not frequency:
                    continue
                idf = math.log1p(
                    (len(self.sections) - self.frequency[token] + 0.5)
                    / (self.frequency[token] + 0.5)
                )
                normalizer = frequency + self.k1 * (1 - self.b + self.b * length / self.average)
                score += idf * frequency * (self.k1 + 1) / normalizer
            if score > 0:
                hits.append(Hit(section, score))
        hits.sort(key=lambda hit: (-hit.score, hit.section.pmid, hit.section.index))
        return tuple(hits[:limit])


def retrieval_metrics(
    retriever: Retriever, papers: Sequence[Paper], limit: int = 5
) -> dict[str, float]:
    if not papers:
        raise ValueError("Retrieval evaluation needs questions.")
    ranks = []
    for paper in papers:
        hits = retriever.search(paper.question, limit)
        rank = next(
            (index + 1 for index, hit in enumerate(hits) if hit.section.pmid == paper.pmid), None
        )
        ranks.append(rank)
    return {
        "recall": sum(rank is not None for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank if rank is not None else 0 for rank in ranks) / len(ranks),
    }
