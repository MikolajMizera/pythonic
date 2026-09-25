import statistics
import time
from typing import Any

import torch

from pythonic.experiments import metadata
from pythonic.generation import prefill, sample_token
from pythonic.model import ByteTokenizer, Decoder


@torch.inference_mode()
def generation_run(model: Decoder, prompt: list[int], max_new: int, cached: bool) -> dict[str, Any]:
    if not prompt or max_new <= 0 or len(prompt) + max_new > model.config.context:
        raise ValueError("Benchmark prompt and output must fit the context.")
    model.eval()
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    tokens = torch.tensor([prompt], device=device)
    outputs = []
    intervals = []
    caches = None
    begin = time.perf_counter()
    previous = begin
    for index in range(max_new):
        if cached:
            if index == 0:
                logits, caches = prefill(model, tokens)
            else:
                logits, caches = model.decode(tokens[:, -1:], caches)
                logits = logits[:, -1]
        else:
            logits = model(tokens)[:, -1]
        token = sample_token(logits[0])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        now = time.perf_counter()
        intervals.append((now - previous) * 1000)
        previous = now
        outputs.append(token)
        if token == ByteTokenizer.eos:
            break
        tokens = torch.cat((tokens, torch.tensor([[token]], device=device)), dim=1)
    elapsed = (time.perf_counter() - begin) * 1000
    return {
        "output_tokens": outputs,
        "ttft_ms": intervals[0],
        "tpot_ms": statistics.mean(intervals[1:]) if len(intervals) > 1 else None,
        "intervals_ms": intervals,
        "latency_ms": elapsed,
        "tokens_per_second": len(outputs) / (elapsed / 1000),
        "cache_bytes": sum(cache.bytes for cache in caches) if caches else 0,
        "peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def generation_benchmark(
    model: Decoder, prompt: list[int], max_new: int = 16, repeats: int = 10
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("Benchmark repetitions must be positive.")
    results = {}
    for cached in (False, True):
        for _ in range(3):
            generation_run(model, prompt, max_new, cached)
        runs = [generation_run(model, prompt, max_new, cached) for _ in range(repeats)]
        results["cached" if cached else "full_context"] = {
            "median_ttft_ms": statistics.median(run["ttft_ms"] for run in runs),
            "median_latency_ms": statistics.median(run["latency_ms"] for run in runs),
            "runs": runs,
        }
    if (
        results["cached"]["runs"][0]["output_tokens"]
        != results["full_context"]["runs"][0]["output_tokens"]
    ):
        raise ValueError("Greedy outputs differ; timing comparison is not quality preserving.")
    return {
        "metadata": metadata(next(model.parameters()).device),
        "model_config": vars(model.config),
        "prompt_tokens": prompt,
        "max_new": max_new,
        "results": results,
    }
