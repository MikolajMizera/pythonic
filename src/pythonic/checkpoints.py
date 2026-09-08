import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from pythonic.agent import AgentState
from pythonic.tools import ToolCall, ToolResult


class CallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    call_id: str
    name: str
    arguments: dict[str, Any]


class ResultRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    call_id: str
    name: str
    value: Any = None
    error: str | None = None


class StateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: str
    calls: list[CallRecord]
    observations: list[ResultRecord]
    status: Literal["running", "complete", "stopped"]
    answer: Any
    reason: str | None


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1]
    source_digest: str
    state: StateRecord


def save_checkpoint(state: AgentState, path: Path, source_digest: str) -> None:
    content = {"version": 1, "source_digest": source_digest, "state": asdict(state)}
    parsed = Checkpoint.model_validate(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(parsed.model_dump_json(indent=2) + "\n")
    temporary.replace(path)


def load_checkpoint(path: Path, source_digest: str) -> AgentState:
    checkpoint = Checkpoint.model_validate_json(path.read_text())
    if checkpoint.source_digest != source_digest:
        raise ValueError("Checkpoint source index differs.")
    state = checkpoint.state
    if len(state.calls) != len(state.observations):
        raise ValueError("Checkpoint calls and observations differ.")
    identities = [(call.call_id, call.name) for call in state.calls]
    if identities != [(result.call_id, result.name) for result in state.observations] or len(
        {call.call_id for call in state.calls}
    ) != len(state.calls):
        raise ValueError("Checkpoint tool identities differ.")
    if state.status == "complete":
        classified = [
            result.value
            for result in state.observations
            if result.name == "classify" and result.error is None
        ]
        if not classified or json.dumps(state.answer, sort_keys=True) != json.dumps(
            classified[-1], sort_keys=True
        ):
            raise ValueError("Completed checkpoint has no matching classification.")
    return AgentState(
        state.question,
        [ToolCall(**call.model_dump()) for call in state.calls],
        [ToolResult(**result.model_dump()) for result in state.observations],
        state.status,
        state.answer,
        state.reason,
    )
