import copy
import math

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.base(inputs) + self.scale * (inputs @ self.a.T @ self.b.T)

    def merged(self) -> nn.Linear:
        result = copy.deepcopy(self.base)
        with torch.no_grad():
            result.weight.add_(self.scale * self.b @ self.a)
        return result


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> tuple[str, ...]:
    model.requires_grad_(False)
    replaced = []
    for path, module in list(model.named_modules()):
        for name, child in list(module.named_children()):
            if name in {"query", "value"} and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))
                replaced.append(f"{path}.{name}".lstrip("."))
    if not replaced:
        raise ValueError("No query/value linear projections found.")
    return tuple(replaced)
