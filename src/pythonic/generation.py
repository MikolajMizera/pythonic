import math
from collections.abc import Generator

import torch
from torch import Tensor

from pythonic.attention import KVCache
from pythonic.model import ByteTokenizer, Decoder


def sample_token(
    logits: Tensor, temperature: float = 0.0, generator: torch.Generator | None = None
) -> int:
    if (
        not math.isfinite(temperature)
        or temperature < 0
        or logits.ndim != 1
        or not torch.isfinite(logits).all()
    ):
        raise ValueError(
            "Sampling needs finite one-dimensional logits and nonnegative temperature."
        )
    if temperature == 0:
        return int(logits.argmax())
    probabilities = torch.softmax(logits.float() / temperature, dim=-1).cpu()
    return int(torch.multinomial(probabilities, 1, generator=generator))


def prefill(
    model: Decoder, tokens: Tensor, chunk_size: int = 32
) -> tuple[Tensor, tuple[KVCache, ...]]:
    if chunk_size <= 0 or tokens.size(1) == 0:
        raise ValueError("Prefill needs a nonempty prompt and positive chunk size.")
    caches = None
    logits = None
    for start in range(0, tokens.size(1), chunk_size):
        logits, caches = model.decode(tokens[:, start : start + chunk_size], caches)
    assert logits is not None and caches is not None
    return logits[:, -1], caches


@torch.inference_mode()
def generate_tokens(
    model: Decoder,
    prompt: list[int],
    max_new: int = 32,
    temperature: float = 0.0,
    chunk_size: int = 32,
    seed: int = 42,
) -> Generator[int, None, None]:
    if not prompt or max_new < 0 or len(prompt) + max_new > model.config.context:
        raise ValueError("Prompt and output must fit the context, with nonnegative output length.")
    if max_new == 0:
        return
    model.eval()
    device = next(model.parameters()).device
    inputs = torch.tensor([prompt], device=device)
    logits, caches = prefill(model, inputs, chunk_size)
    generator = torch.Generator().manual_seed(seed)
    for index in range(max_new):
        token = sample_token(logits[0], temperature, generator)
        if token == ByteTokenizer.eos:
            return
        yield token
        if index + 1 < max_new:
            logits, caches = model.decode(torch.tensor([[token]], device=device), caches)
            logits = logits[:, -1]
