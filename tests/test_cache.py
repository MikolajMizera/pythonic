import pytest
import torch

from pythonic.attention import kv_cache_bytes
from pythonic.model import Decoder, DecoderConfig


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_cached_chunks_match_full_logits(kv_heads: int) -> None:
    model = Decoder(DecoderConfig(width=16, layers=2, kv_heads=kv_heads, context=12)).eval()
    tokens = torch.randint(0, 258, (2, 11))
    full = model(tokens)
    first, caches = model.decode(tokens[:, :4])
    second, updated = model.decode(tokens[:, 4:7], caches)
    third, final = model.decode(tokens[:, 7:], updated)
    torch.testing.assert_close(torch.cat((first, second, third), dim=1), full)
    assert caches[0].length == 4
    assert final[0].length == 11
    assert sum(cache.bytes for cache in final) == kv_cache_bytes(2, 2, 11, kv_heads, 4)
    with pytest.raises(ValueError):
        model.decode(tokens[:, :2], final)


def test_request_caches_do_not_change_other_requests() -> None:
    model = Decoder(DecoderConfig(width=16, layers=1, context=8))
    _, first = model.decode(torch.tensor([[1, 2]]))
    before = first[0].key.clone()
    model.decode(torch.tensor([[3, 4, 5]]))
    torch.testing.assert_close(first[0].key, before)
