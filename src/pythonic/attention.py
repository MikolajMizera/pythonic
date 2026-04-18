import math

import torch
from torch import Tensor, nn


def causal_attention(query: Tensor, key: Tensor, value: Tensor, offset: int = 0) -> Tensor:
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    rows = torch.arange(query.size(-2), device=query.device) + offset
    columns = torch.arange(key.size(-2), device=query.device)
    allowed = columns[None, :] <= rows[:, None]
    scores = scores.masked_fill(~allowed, -torch.inf)
    return torch.softmax(scores, dim=-1) @ value


class Attention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("Width must be positive and divisible by the head count.")
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def split_heads(self, projected: Tensor) -> Tensor:
        batch, length, _ = projected.shape
        return projected.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        query = self.split_heads(self.query(inputs))
        key = self.split_heads(self.key(inputs))
        value = self.split_heads(self.value(inputs))
        attended = causal_attention(query, key, value)
        merged = attended.transpose(1, 2).contiguous().flatten(2)
        return self.output(merged)
