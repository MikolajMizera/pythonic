import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class KVCache:
    key: Tensor
    value: Tensor

    @property
    def length(self) -> int:
        return self.key.size(-2)

    @property
    def bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in (self.key, self.value))


def kv_cache_bytes(
    layers: int, batch: int, length: int, kv_heads: int, head_dim: int, element_bytes: int = 4
) -> int:
    dimensions = (layers, batch, length, kv_heads, head_dim, element_bytes)
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError("Cache dimensions cannot be negative.")
    return 2 * math.prod(dimensions)


def causal_attention(query: Tensor, key: Tensor, value: Tensor, offset: int = 0) -> Tensor:
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    rows = torch.arange(query.size(-2), device=query.device) + offset
    columns = torch.arange(key.size(-2), device=query.device)
    allowed = columns[None, :] <= rows[:, None]
    scores = scores.masked_fill(~allowed, -torch.inf)
    return torch.softmax(scores, dim=-1) @ value


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int | None = None) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("Width must be positive and divisible by the head count.")
        self.heads = heads
        self.kv_heads = heads if kv_heads is None else kv_heads
        if self.kv_heads <= 0 or heads % self.kv_heads:
            raise ValueError("KV heads must divide query heads.")
        self.head_dim = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, self.kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(width, self.kv_heads * self.head_dim, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def split_heads(self, projected: Tensor, heads: int) -> Tensor:
        batch, length, _ = projected.shape
        return projected.view(batch, length, heads, self.head_dim).transpose(1, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_cached(inputs)[0]

    def forward_cached(
        self, inputs: Tensor, cache: KVCache | None = None
    ) -> tuple[Tensor, KVCache]:
        query = self.split_heads(self.query(inputs), self.heads)
        key = self.split_heads(self.key(inputs), self.kv_heads)
        value = self.split_heads(self.value(inputs), self.kv_heads)
        offset = 0
        if cache is not None:
            if cache.key.shape != cache.value.shape or cache.key.shape[:2] != key.shape[:2]:
                raise ValueError("Cache shape does not match this request.")
            offset = cache.length
            key = torch.cat((cache.key, key), dim=-2)
            value = torch.cat((cache.value, value), dim=-2)
        updated = KVCache(key, value)
        key = key.repeat_interleave(self.heads // self.kv_heads, dim=1)
        value = value.repeat_interleave(self.heads // self.kv_heads, dim=1)
        attended = causal_attention(query, key, value, offset)
        merged = attended.transpose(1, 2).contiguous().flatten(2)
        return self.output(merged), updated
