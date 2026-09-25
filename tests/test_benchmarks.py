from pythonic.benchmarks import generation_benchmark
from pythonic.model import Decoder, DecoderConfig


def test_generation_benchmark_preserves_output_and_reports_real_samples() -> None:
    model = Decoder(DecoderConfig(width=16, layers=1, context=16))
    report = generation_benchmark(model, [1, 2, 3], max_new=4, repeats=2)
    cached = report["results"]["cached"]["runs"]
    uncached = report["results"]["full_context"]["runs"]
    assert len(cached) == 2
    assert cached[0]["output_tokens"] == uncached[0]["output_tokens"]
    assert cached[0]["ttft_ms"] > 0
    assert cached[0]["cache_bytes"] > 0
    assert uncached[0]["cache_bytes"] == 0
    assert cached[0]["peak_bytes"] is None
