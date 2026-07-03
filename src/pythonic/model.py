from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pythonic.attention import Attention, KVCache


@dataclass(frozen=True)
class DecoderConfig:
    vocabulary: int = 258
    width: int = 256
    layers: int = 4
    heads: int = 4
    kv_heads: int = 2
    context: int = 256

    def __post_init__(self) -> None:
        if (
            min(self.vocabulary, self.width, self.layers, self.heads, self.kv_heads, self.context)
            <= 0
        ):
            raise ValueError("Model dimensions must be positive.")
        if self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("Width and KV heads must divide evenly into query heads.")


class Block(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = Attention(config.width, config.heads, config.kv_heads)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.mlp(self.mlp_norm(inputs))

    def forward_cached(self, inputs: Tensor, cache: KVCache | None) -> tuple[Tensor, KVCache]:
        attended, updated = self.attention.forward_cached(self.attention_norm(inputs), cache)
        inputs = inputs + attended
        return inputs + self.mlp(self.mlp_norm(inputs)), updated


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig | None = None) -> None:
        super().__init__()
        config = config or DecoderConfig()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.vocabulary, bias=False)
        self.output.weight = self.embedding.weight
        self.apply(self.initialize)

    @staticmethod
    def initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        length = tokens.size(1)
        if not 0 < length <= self.config.context:
            raise ValueError("Token length must fit the model context.")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.norm(hidden))

    def decode(
        self, tokens: Tensor, caches: tuple[KVCache, ...] | None = None
    ) -> tuple[Tensor, tuple[KVCache, ...]]:
        offset = 0
        if caches is not None:
            if len(caches) != self.config.layers or len({cache.length for cache in caches}) != 1:
                raise ValueError("Every decoder layer needs a cache of equal length.")
            offset = caches[0].length
        length = tokens.size(1)
        if not 0 < length or offset + length > self.config.context:
            raise ValueError("Cached token length must fit the model context.")
        positions = torch.arange(offset, offset + length, device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        updated = []
        for index, block in enumerate(self.blocks):
            hidden, cache = block.forward_cached(hidden, None if caches is None else caches[index])
            updated.append(cache)
        return self.output(self.norm(hidden)), tuple(updated)


class ByteTokenizer:
    bos = 256
    eos = 257

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return bytes(token for token in tokens if token < 256).decode("utf-8", errors="replace")
