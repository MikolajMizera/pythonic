import torch
from torch import nn

from pythonic.lora import LoRALinear, inject_lora
from pythonic.model import Decoder, DecoderConfig


def test_lora_initialization_freezing_and_merge() -> None:
    base = nn.Linear(7, 5)
    inputs = torch.randn(3, 7)
    layer = LoRALinear(base, rank=2)
    torch.testing.assert_close(layer(inputs), base(inputs))
    layer(inputs).square().sum().backward()
    assert base.weight.grad is None
    assert layer.b.grad is not None and layer.b.grad.abs().sum() > 0
    with torch.no_grad():
        layer.b.normal_()
    torch.testing.assert_close(layer(inputs), layer.merged()(inputs))


def test_injection_preserves_initial_outputs_and_only_trains_adapters() -> None:
    model = Decoder(DecoderConfig(width=16, layers=1, context=8))
    inputs = torch.randint(0, 258, (1, 8))
    expected = model(inputs).detach()
    names = inject_lora(model, rank=2)
    assert len(names) == 2
    torch.testing.assert_close(model(inputs), expected)
    assert all(
        name.endswith((".a", ".b")) for name, p in model.named_parameters() if p.requires_grad
    )
