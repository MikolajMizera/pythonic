import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Int8Linear(nn.Module):
    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        weight = source.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
        self.register_buffer("scale", scale)
        self.register_buffer("codes", (weight / scale).round().clamp(-127, 127).to(torch.int8))
        self.register_buffer("bias", None if source.bias is None else source.bias.detach().clone())

    def dequantize(self) -> Tensor:
        return self.codes.float() * self.scale

    def forward(self, inputs: Tensor) -> Tensor:
        bias = None if self.bias is None else self.bias.to(inputs.dtype)
        return F.linear(inputs, self.dequantize().to(inputs.dtype), bias)


def quantize_linears(model: nn.Module) -> nn.Module:
    result = copy.deepcopy(model)

    def replace(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if name == "output":
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, Int8Linear(child))
            else:
                replace(child)

    replace(result)
    return result.eval()


def storage_bytes(model: nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size() for tensor in (*model.parameters(), *model.buffers())
    )
