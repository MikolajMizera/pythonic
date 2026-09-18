from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from pythonic.biomedical import MODEL_ID, MODEL_REVISION
from pythonic.data import Section
from pythonic.retrieval import Hit, Retriever

Embeddings = NDArray[np.float32]


class BiomedicalEmbedder:
    def __init__(
        self, encoder: nn.Module, tokenizer: Any, width: int, budget: int = 256, batch_size: int = 8
    ) -> None:
        if not 8 <= budget <= 512 or batch_size <= 0:
            raise ValueError("Embedding budget or batch size is invalid.")
        self.encoder = encoder.requires_grad_(False).eval()
        self.tokenizer = tokenizer
        self.budget, self.batch_size = budget, batch_size
        self.width = width

    @classmethod
    def load(cls, device: str = "cpu") -> "BiomedicalEmbedder":
        from transformers import AutoModel, AutoTokenizer

        encoder = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        return cls(encoder, tokenizer, encoder.config.hidden_size)

    @torch.inference_mode()
    def __call__(self, texts: Sequence[str]) -> Embeddings:
        if not texts:
            return np.empty((0, self.width), dtype=np.float32)
        vectors = []
        device = next(self.encoder.parameters()).device
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                list(texts[start : start + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=self.budget,
                return_tensors="pt",
            )
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            hidden = self.encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(pooled.float().cpu().numpy())
        return np.concatenate(vectors).astype(np.float32)


class DenseIndex:
    def __init__(
        self, sections: Sequence[Section], embed: Callable[[Sequence[str]], Embeddings]
    ) -> None:
        self.sections = tuple(sections)
        if len({(s.pmid, s.index) for s in self.sections}) != len(self.sections):
            raise ValueError("Section identities must be unique.")
        self.embed = embed
        matrix = embed([section.text for section in self.sections])
        if (
            matrix.ndim != 2
            or matrix.shape[0] != len(self.sections)
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("Embedding matrix shape or values are invalid.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        self.valid = norms[:, 0] > 0

    def search(self, query: str, limit: int = 5) -> tuple[Hit, ...]:
        if limit <= 0:
            raise ValueError("Search limit must be positive.")
        if not self.sections or not query.strip():
            return ()
        vector = self.embed([query])
        if vector.shape != (1, self.matrix.shape[1]) or not np.isfinite(vector).all():
            raise ValueError("Query embedding shape or values are invalid.")
        norm = np.linalg.norm(vector)
        if norm == 0:
            return ()
        scores = self.matrix @ (vector[0] / norm)
        hits = [
            Hit(section, float(score))
            for section, score, valid in zip(self.sections, scores, self.valid, strict=True)
            if valid
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.section.pmid, hit.section.index))
        return tuple(hits[:limit])


class HybridIndex:
    def __init__(self, sparse: Retriever, dense: Retriever, rank_offset: int = 60) -> None:
        if rank_offset <= 0:
            raise ValueError("Rank offset must be positive.")
        self.sparse, self.dense = sparse, dense
        self.rank_offset = rank_offset

    def search(self, query: str, limit: int = 5) -> tuple[Hit, ...]:
        if limit <= 0:
            raise ValueError("Search limit must be positive.")
        scores: dict[tuple[str, int], float] = {}
        sections: dict[tuple[str, int], Section] = {}
        for retriever in (self.sparse, self.dense):
            seen = set()
            for rank, hit in enumerate(retriever.search(query, max(20, limit)), start=1):
                identity = hit.section.pmid, hit.section.index
                if identity in seen:
                    continue
                seen.add(identity)
                sections[identity] = hit.section
                scores[identity] = scores.get(identity, 0) + 1 / (self.rank_offset + rank)
        ordered = sorted(scores, key=lambda identity: (-scores[identity], identity))
        return tuple(Hit(sections[identity], scores[identity]) for identity in ordered[:limit])
