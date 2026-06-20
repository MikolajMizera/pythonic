import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class Timing:
    median_ms: float
    p95_ms: float
    samples_ms: tuple[float, ...]
    peak_bytes: int | None


def benchmark(
    operation: Callable[[], object], device: torch.device, warmup: int = 3, repeats: int = 10
) -> Timing:
    if warmup < 0 or repeats <= 0:
        raise ValueError("Warmup must be nonnegative and repeats positive.")
    for _ in range(warmup):
        operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return Timing(
        statistics.median(samples),
        ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        tuple(samples),
        peak,
    )


def metadata(device: torch.device, seed: int = 42) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "seed": seed,
    }


def save_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def attention_benchmark(
    device: str = "cpu", length: int = 128, repeats: int = 10
) -> dict[str, Any]:
    from torch.nn.functional import scaled_dot_product_attention

    from pythonic.attention import causal_attention
    from pythonic.tiled import tiled_attention

    torch.manual_seed(42)
    target = torch.device(device)
    query, key, value = [torch.randn(1, 4, length, 32, device=target) for _ in range(3)]
    operations: dict[str, Callable[[], object]] = {
        "dense": lambda: causal_attention(query, key, value),
        "tiled": lambda: tiled_attention(query, key, value),
        "sdpa": lambda: scaled_dot_product_attention(query, key, value, is_causal=True),
    }
    with torch.inference_mode():
        timings = {
            name: asdict(benchmark(call, target, repeats=repeats))
            for name, call in operations.items()
        }
    return {"metadata": metadata(target), "length": length, "timings": timings}
