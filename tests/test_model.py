import pytest
import torch

from pythonic.model import ByteTokenizer, Decoder, DecoderConfig


def test_decoder_tied_weights_causality_and_context() -> None:
    model = Decoder(DecoderConfig(width=16, layers=1, heads=4, kv_heads=2, context=8))
    assert model.output.weight is model.embedding.weight
    tokens = torch.randint(0, 258, (2, 7))
    changed = tokens.clone()
    changed[:, 4:] = 0
    torch.testing.assert_close(model(tokens)[:, :4], model(changed)[:, :4])
    assert model(tokens).shape == (2, 7, 258)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 9, dtype=torch.long))


def test_byte_tokenizer_handles_unicode_and_special_tokens() -> None:
    text = "β cells and Łódź"
    assert ByteTokenizer.decode([256, *ByteTokenizer.encode(text), 257]) == text
