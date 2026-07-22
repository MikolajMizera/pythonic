from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    arrival: float = 0.0

    def __post_init__(self) -> None:
        if self.prompt_tokens <= 0 or self.output_tokens <= 0 or self.arrival < 0:
            raise ValueError("Requests need positive lengths and nonnegative arrival times.")


@dataclass
class Active:
    request: Request
    remaining_prompt: int
    generated: int = 0
    first_token: float | None = None


@dataclass(frozen=True)
class Completion:
    request_id: str
    ttft: float
    latency: float


@dataclass(frozen=True)
class ScheduleResult:
    completions: tuple[Completion, ...]
    duration: float
    output_tokens: int
    peak_sequences: int
    iterations: int

    @property
    def throughput(self) -> float:
        return self.output_tokens / self.duration if self.duration else 0.0


def simulate(
    requests: tuple[Request, ...],
    continuous: bool = True,
    max_sequences: int = 4,
    token_budget: int = 64,
    prefill_chunk: int = 32,
) -> ScheduleResult:
    if min(max_sequences, token_budget, prefill_chunk) <= 0 or token_budget < max_sequences:
        raise ValueError("Budget must cover one token per active sequence.")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("Request IDs must be unique.")
    pending = deque(sorted(requests, key=lambda request: request.arrival))
    active: list[Active] = []
    completions = []
    clock = 0.0
    iterations = peak = 0
    while pending or active:
        if not active and pending:
            clock = max(clock, pending[0].arrival)
        if continuous or not active:
            while pending and pending[0].arrival <= clock and len(active) < max_sequences:
                request = pending.popleft()
                active.append(Active(request, request.prompt_tokens))
        peak = max(peak, len(active))
        available = token_budget
        decoding = [state for state in active if state.remaining_prompt == 0]
        for state in decoding:
            state.generated += 1
            available -= 1
        for state in active:
            if state.remaining_prompt and available:
                count = min(state.remaining_prompt, prefill_chunk, available)
                state.remaining_prompt -= count
                available -= count
        clock += 1.0 + 0.01 * (token_budget - available)
        iterations += 1
        for state in decoding:
            if state.first_token is None:
                state.first_token = clock
        remaining = []
        for state in active:
            if state.generated == state.request.output_tokens:
                assert state.first_token is not None
                completions.append(
                    Completion(
                        state.request.request_id,
                        state.first_token - state.request.arrival,
                        clock - state.request.arrival,
                    )
                )
            else:
                remaining.append(state)
        active = remaining
    return ScheduleResult(
        tuple(completions), clock, sum(r.output_tokens for r in requests), peak, iterations
    )
