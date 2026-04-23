import torch
from torch.nn.functional import scaled_dot_product_attention

from pythonic.attention import Attention, causal_attention


def test_causal_attention_matches_reference_and_gradients() -> None:
    torch.manual_seed(1)
    query, key, value = [torch.randn(2, 3, 7, 4, requires_grad=True) for _ in range(3)]
    actual = causal_attention(query, key, value)
    expected = scaled_dot_product_attention(query, key, value, is_causal=True)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert all(
        tensor.grad is not None and tensor.grad.isfinite().all() for tensor in (query, key, value)
    )


def test_future_tokens_do_not_change_previous_outputs() -> None:
    model = Attention(16, 4)
    inputs = torch.randn(1, 6, 16)
    changed = inputs.clone()
    changed[:, 3:] += 100
    torch.testing.assert_close(model(inputs)[:, :3], model(changed)[:, :3])


def test_multi_query_matches_tied_multihead_weights() -> None:
    shared = Attention(16, 4, kv_heads=1)
    repeated = Attention(16, 4)
    with torch.no_grad():
        repeated.query.weight.copy_(shared.query.weight)
        repeated.output.weight.copy_(shared.output.weight)
        repeated.key.weight.copy_(shared.key.weight.repeat(4, 1))
        repeated.value.weight.copy_(shared.value.weight.repeat(4, 1))
    inputs = torch.randn(2, 7, 16)
    torch.testing.assert_close(shared(inputs), repeated(inputs))
