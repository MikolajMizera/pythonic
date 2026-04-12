from dataclasses import dataclass
from typing import Literal

Decision = Literal["yes", "no", "maybe"]
DECISIONS: tuple[Decision, ...] = ("yes", "no", "maybe")


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
