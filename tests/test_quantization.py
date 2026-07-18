import torch
from torch import nn

from pythonic.model import Decoder, DecoderConfig
from pythonic.quantization import Int8Linear, quantize_linears, storage_bytes


def test_quantization_error_bound_and_zero_channels() -> None:
    source = nn.Linear(13, 7)
    with torch.no_grad():
        source.weight[0].zero_()
    quantized = Int8Linear(source)
    error = (source.weight - quantized.dequantize()).abs()
    assert torch.all(error <= quantized.scale / 2 + 1e-7)
    assert quantized.codes[0].count_nonzero() == 0
    assert quantized(torch.randn(2, 13)).isfinite().all()


def test_quantized_decoder_preserves_tied_embeddings_and_reduces_storage() -> None:
    model = Decoder(DecoderConfig(width=32, layers=2, context=16))
    quantized = quantize_linears(model)
    assert quantized.embedding.weight is quantized.output.weight
    assert storage_bytes(quantized) < storage_bytes(model)
    tokens = torch.randint(0, 258, (1, 8))
    assert (model(tokens) - quantized(tokens)).abs().max() < 0.02
    assert isinstance(model.blocks[0].attention.query, nn.Linear)
