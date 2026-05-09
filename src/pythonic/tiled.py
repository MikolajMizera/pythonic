import math

import torch
from torch import Tensor


def tiled_attention(query: Tensor, key: Tensor, value: Tensor, tile: int = 32) -> Tensor:
    if tile <= 0:
        raise ValueError("Tile size must be positive.")
    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError("This reference implementation requires equal Q, K, V shapes.")
    outputs = []
    length = query.size(-2)
    for start in range(0, length, tile):
        q = query[..., start : start + tile, :]
        rows = torch.arange(start, start + q.size(-2), device=query.device)
        maximum = torch.full_like(q[..., :1], -torch.inf)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(q)
        for column in range(0, start + q.size(-2), tile):
            k = key[..., column : column + tile, :]
            v = value[..., column : column + tile, :]
            columns = torch.arange(column, column + k.size(-2), device=query.device)
            scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            scores = scores.masked_fill(columns[None, :] > rows[:, None], -torch.inf)
            updated = torch.maximum(maximum, scores.amax(-1, keepdim=True))
            # Rescale previous partial sums when the running maximum changes.
            correction = torch.exp(maximum - updated)
            weights = torch.exp(scores - updated)
            numerator = correction * numerator + weights @ v
            denominator = correction * denominator + weights.sum(-1, keepdim=True)
            maximum = updated
        outputs.append(numerator / denominator)
    return torch.cat(outputs, dim=-2)
