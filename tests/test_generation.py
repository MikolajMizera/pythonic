import pytest
import torch

from pythonic.generation import generate_tokens, prefill, sample_token
from pythonic.model import Decoder, DecoderConfig


def test_chunked_prefill_matches_full_prefill() -> None:
    model = Decoder(DecoderConfig(width=16, layers=2, context=32)).eval()
    tokens = torch.randint(0, 258, (1, 21))
    expected, _ = model.decode(tokens)
    for chunk_size in (1, 4, 32):
        actual, caches = prefill(model, tokens, chunk_size)
        torch.testing.assert_close(actual, expected[:, -1])
        assert all(cache.length == 21 for cache in caches)


def test_sampling_and_generation_are_reproducible_and_bounded() -> None:
    assert sample_token(torch.tensor([1.0, 3.0, 2.0])) == 1
    model = Decoder(DecoderConfig(width=16, layers=1, context=16))
    first = list(generate_tokens(model, [1, 2], max_new=8, temperature=0.8))
    assert first == list(generate_tokens(model, [1, 2], max_new=8, temperature=0.8))
    assert len(first) <= 8
    assert list(generate_tokens(model, [1], max_new=0)) == []
    with pytest.raises(ValueError):
        list(generate_tokens(model, [1] * 16, max_new=1))
    with pytest.raises(ValueError):
        sample_token(torch.tensor([float("nan")]))
    with pytest.raises(ValueError):
        sample_token(torch.tensor([1.0]), temperature=float("nan"))


def test_eos_stops_without_yielding_a_special_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pythonic.generation.sample_token", lambda *args: 257)
    model = Decoder(DecoderConfig(width=16, layers=1, context=8))
    assert list(generate_tokens(model, [1, 2], max_new=4)) == []
