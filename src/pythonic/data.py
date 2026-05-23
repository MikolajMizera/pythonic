import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

from pythonic.experiments import save_json

Decision = Literal["yes", "no", "maybe"]
DECISIONS: tuple[Decision, ...] = ("yes", "no", "maybe")
DATA_REVISION = "1cbae8e92f72f20c8d3747cbb3bf5bc53554d997"
DATA_SHA256 = "8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d"
DATA_URL = f"https://raw.githubusercontent.com/pubmedqa/pubmedqa/{DATA_REVISION}/data/ori_pqal.json"


@dataclass(frozen=True)
class Section:
    pmid: str
    index: int
    label: str
    text: str


@dataclass(frozen=True)
class Paper:
    pmid: str
    question: str
    sections: tuple[Section, ...]
    decision: Decision


def abstract(paper: Paper) -> str:
    return "\n".join(section.text for section in paper.sections)


def download_pubmedqa(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pubmedqa.json"
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(DATA_URL, timeout=60) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != DATA_SHA256:
        raise ValueError("PubMedQA checksum mismatch.")
    path.write_bytes(content)
    save_json(
        directory / "source.json",
        {"url": DATA_URL, "revision": DATA_REVISION, "sha256": DATA_SHA256},
    )
    return path


def load_papers(path: Path) -> tuple[Paper, ...]:
    raw = json.loads(path.read_text())
    papers = []
    for pmid, record in sorted(raw.items()):
        decision = record["final_decision"]
        if decision not in DECISIONS:
            raise ValueError(f"Unknown decision for {pmid}.")
        sections = tuple(
            Section(pmid, index, label, text)
            for index, (label, text) in enumerate(
                zip(record["LABELS"], record["CONTEXTS"], strict=True)
            )
            if label.upper() not in {"CONCLUSION", "CONCLUSIONS"}
        )
        papers.append(Paper(pmid, record["QUESTION"], sections, decision))
    return tuple(papers)


@dataclass(frozen=True)
class Split:
    train: tuple[Paper, ...]
    validation: tuple[Paper, ...]
    test: tuple[Paper, ...]


def split_papers(papers: tuple[Paper, ...], seed: int = 42) -> Split:
    if len({paper.pmid for paper in papers}) != len(papers):
        raise ValueError("PMIDs must be unique before splitting.")
    rng = random.Random(seed)
    train, validation, test = [], [], []
    for decision in DECISIONS:
        group = sorted(
            (paper for paper in papers if paper.decision == decision), key=lambda p: p.pmid
        )
        rng.shuffle(group)
        boundary = int(0.7 * len(group))
        second = boundary + int(0.15 * len(group))
        train.extend(group[:boundary])
        validation.extend(group[boundary:second])
        test.extend(group[second:])
    return Split(tuple(train), tuple(validation), tuple(test))


def save_split(split: Split, path: Path, seed: int = 42) -> None:
    save_json(
        path,
        {
            "seed": seed,
            "source_revision": DATA_REVISION,
            "source_sha256": DATA_SHA256,
            "train": [paper.pmid for paper in split.train],
            "validation": [paper.pmid for paper in split.validation],
            "test": [paper.pmid for paper in split.test],
        },
    )


def fixtures() -> tuple[Paper, ...]:
    examples: tuple[tuple[str, str, Decision], ...] = (
        ("Does treatment reduce fever?", "Treatment reduced fever in the trial.", "yes"),
        ("Does exercise reduce fatigue?", "Exercise did not reduce fatigue.", "no"),
        ("Does sleep improve recovery?", "The sample was too small to assess recovery.", "maybe"),
    )
    return tuple(
        Paper(str(i), question, (Section(str(i), 0, "RESULTS", text),), decision)
        for i, (question, text, decision) in enumerate(examples)
    )
