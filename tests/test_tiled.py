import pytest
import torch

from pythonic.attention import causal_attention
from pythonic.tiled import tiled_attention


@pytest.mark.parametrize("tile", [1, 3, 8, 32])
def test_tiled_outputs_and_gradients(tile: int) -> None:
    torch.manual_seed(9)
    inputs = [torch.randn(2, 3, 11, 5, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    reference = causal_attention(*inputs)
    tiled = tiled_attention(*inputs, tile=tile)
    torch.testing.assert_close(tiled, reference)
    actual_gradients = torch.autograd.grad(tiled.square().sum(), inputs)
    expected_gradients = torch.autograd.grad(reference.square().sum(), inputs)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_tiled_large_logits_remain_finite() -> None:
    query = torch.randn(1, 2, 13, 4) * 100
    assert tiled_attention(query, query, query, tile=4).isfinite().all()
