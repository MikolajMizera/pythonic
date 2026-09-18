import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pythonic.biomedical import EvidencePrediction
from pythonic.context import answer_question, validate_citations
from pythonic.data import Section
from pythonic.retrieval import Hit, Retriever


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchArguments(Arguments):
    query: str = Field(min_length=1, max_length=8000)
    limit: int = Field(default=5, ge=1, le=20)


class ReadArguments(Arguments):
    pmid: str = Field(min_length=1)
    section: int = Field(ge=0)


class ClassifyArguments(Arguments):
    question: str = Field(min_length=1, max_length=8000)
    references: list[ReadArguments] = Field(min_length=1, max_length=10)
    budget: int = Field(default=256, ge=8, le=512)


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    value: Any = None
    error: str | None = None


class EvidenceTools:
    def __init__(
        self,
        retriever: Retriever,
        sections: Sequence[Section],
        predict: Callable[[str, str, int], EvidencePrediction],
        count: Callable[[str], int],
        allowed: frozenset[str] = frozenset({"search", "read_section", "classify"}),
    ) -> None:
        self.retriever = retriever
        self.sources = {(section.pmid, section.index): section for section in sections}
        if len(self.sources) != len(sections):
            raise ValueError("Section identities must be unique.")
        self.predict, self.count = predict, count
        self.allowed = allowed
        self.source_digest = hashlib.sha256(
            json.dumps(
                [asdict(self.sources[key]) for key in sorted(self.sources)], sort_keys=True
            ).encode()
        ).hexdigest()
        self.arguments: dict[str, type[Arguments]] = {
            "search": SearchArguments,
            "read_section": ReadArguments,
            "classify": ClassifyArguments,
        }

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "parameters": args.model_json_schema()}
            for name, args in self.arguments.items()
            if name in self.allowed
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        if not call.call_id or call.name not in self.arguments:
            return ToolResult(call.call_id, call.name, error="Unknown tool or empty call ID.")
        if call.name not in self.allowed:
            return ToolResult(call.call_id, call.name, error="Tool is not permitted.")
        try:
            parsed = self.arguments[call.name].model_validate(call.arguments)
            if isinstance(parsed, SearchArguments):
                hits = self.retriever.search(parsed.query, parsed.limit)
                for hit in hits:
                    if self.sources.get((hit.section.pmid, hit.section.index)) != hit.section:
                        raise ValueError("Retrieved section does not match the source index.")
                value: Any = [asdict(hit) for hit in hits]
            elif isinstance(parsed, ReadArguments):
                value = asdict(self.sources[(parsed.pmid, parsed.section)])
            elif isinstance(parsed, ClassifyArguments):
                classification_hits = [
                    Hit(self.sources[(ref.pmid, ref.section)], 1.0) for ref in parsed.references
                ]
                answer = answer_question(
                    parsed.question,
                    classification_hits,
                    lambda question, context: self.predict(question, context, parsed.budget),
                    parsed.budget,
                    self.count,
                )
                validate_citations(answer.citations, self.sources)
                value = asdict(answer)
            else:
                raise ValueError("Unsupported arguments.")
            return ToolResult(call.call_id, call.name, value=value)
        except (ValidationError, ValueError, KeyError) as error:
            return ToolResult(call.call_id, call.name, error=f"{type(error).__name__}: {error}")
