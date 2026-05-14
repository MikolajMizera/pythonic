import pytest
import torch

from pythonic.experiments import benchmark


def test_benchmark_warmup_is_excluded() -> None:
    calls = []
    timing = benchmark(lambda: calls.append(1), torch.device("cpu"), warmup=2, repeats=5)
    assert len(calls) == 7
    assert len(timing.samples_ms) == 5
    assert timing.peak_bytes is None
    assert timing.p95_ms >= timing.median_ms >= 0
    with pytest.raises(ValueError):
        benchmark(lambda: None, torch.device("cpu"), repeats=0)
