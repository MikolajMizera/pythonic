import pytest

from pythonic.scheduling import Request, simulate


def test_continuous_admission_uses_finished_slots() -> None:
    requests = tuple(Request(str(i), 4, count) for i, count in enumerate([20, 1, 1, 1, 1, 1]))
    static = simulate(requests, continuous=False, max_sequences=2)
    continuous = simulate(requests, continuous=True, max_sequences=2)
    assert continuous.duration < static.duration
    assert len(continuous.completions) == len(requests)
    assert continuous.peak_sequences == 2
    assert continuous.output_tokens == 25


def test_arrivals_and_small_token_budgets_complete_without_starvation() -> None:
    requests = (Request("long", 200, 3), Request("late", 2, 1, arrival=50))
    result = simulate(requests, max_sequences=2, token_budget=2, prefill_chunk=2)
    assert {item.request_id for item in result.completions} == {"long", "late"}
    assert all(item.latency >= item.ttft > 0 for item in result.completions)
    with pytest.raises(ValueError):
        simulate(requests, max_sequences=3, token_budget=2)
