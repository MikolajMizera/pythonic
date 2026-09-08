import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pythonic.tools import EvidenceTools, ToolCall, ToolResult


@dataclass(frozen=True)
class Finish:
    reason: str | None = None


@dataclass
class AgentState:
    question: str
    calls: list[ToolCall] = field(default_factory=list)
    observations: list[ToolResult] = field(default_factory=list)
    status: Literal["running", "complete", "stopped"] = "running"
    answer: Any = None
    reason: str | None = None


class SearchReadClassify:
    def __init__(self, budget: int = 256) -> None:
        self.budget = budget

    def __call__(self, state: AgentState) -> ToolCall | Finish:
        successful = {result.name: result for result in state.observations if result.error is None}
        call_id = str(len(state.calls) + 1)
        if "search" not in successful:
            return ToolCall(call_id, "search", {"query": state.question, "limit": 3})
        hits = successful["search"].value
        if not hits:
            return Finish("No evidence found.")
        section = hits[0]["section"]
        if "read_section" not in successful:
            return ToolCall(
                call_id, "read_section", {"pmid": section["pmid"], "section": section["index"]}
            )
        if "classify" not in successful:
            read = successful["read_section"].value
            return ToolCall(
                call_id,
                "classify",
                {
                    "question": state.question,
                    "references": [{"pmid": read["pmid"], "section": read["index"]}],
                    "budget": self.budget,
                },
            )
        return Finish()


class Agent:
    def __init__(
        self,
        tools: EvidenceTools,
        select: Callable[[AgentState], ToolCall | Finish],
        max_turns: int = 8,
        max_errors: int = 3,
    ) -> None:
        if min(max_turns, max_errors) <= 0:
            raise ValueError("Agent limits must be positive.")
        self.tools, self.select = tools, select
        self.max_turns, self.max_errors = max_turns, max_errors

    def step(self, state: AgentState) -> AgentState:
        if state.status != "running":
            return state
        if len(state.calls) >= self.max_turns:
            state.status, state.reason = "stopped", "Turn limit reached."
            return state
        action = self.select(state)
        if isinstance(action, Finish):
            classified = [
                result
                for result in state.observations
                if result.name == "classify" and result.error is None
            ]
            if classified:
                state.status, state.answer = "complete", classified[-1].value
            else:
                state.status, state.reason = "stopped", action.reason or "No classified evidence."
            return state
        signature = json.dumps([action.name, action.arguments], sort_keys=True)
        previous = {json.dumps([call.name, call.arguments], sort_keys=True) for call in state.calls}
        if signature in previous or action.call_id in {call.call_id for call in state.calls}:
            state.status, state.reason = "stopped", "Repeated action or call ID."
            return state
        result = self.tools.execute(action)
        state.calls.append(action)
        state.observations.append(result)
        if sum(item.error is not None for item in state.observations) >= self.max_errors:
            state.status, state.reason = "stopped", "Tool error limit reached."
        return state

    def run(self, question: str, checkpoint: Path | None = None) -> AgentState:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        state = AgentState(question)
        return self.continue_run(state, checkpoint)

    def resume(self, checkpoint: Path) -> AgentState:
        from pythonic.checkpoints import load_checkpoint

        return self.continue_run(load_checkpoint(checkpoint, self.tools.source_digest), checkpoint)

    def continue_run(self, state: AgentState, checkpoint: Path | None = None) -> AgentState:
        from pythonic.checkpoints import save_checkpoint

        while state.status == "running":
            self.step(state)
            if checkpoint is not None:
                save_checkpoint(state, checkpoint, self.tools.source_digest)
        return state
