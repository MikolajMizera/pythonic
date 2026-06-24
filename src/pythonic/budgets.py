import statistics
import time
from typing import Any

import torch

from pythonic.biomedical import (
    MODEL_ID,
    MODEL_REVISION,
    EvidenceClassifier,
    Strategy,
    predict_evidence,
)
from pythonic.data import DATA_REVISION, DATA_SHA256, Paper
from pythonic.experiments import metadata
from pythonic.metrics import classification_metrics


def budget_experiment(
    model: EvidenceClassifier,
    tokenizer: Any,
    papers: tuple[Paper, ...],
    budgets: tuple[int, ...] = (128, 256, 512),
    strategies: tuple[Strategy, ...] = ("question_only", "first", "results_first", "original"),
) -> dict[str, Any]:
    if not papers:
        raise ValueError("Budget experiment needs evaluation papers.")
    device = next(model.parameters()).device
    rows = []
    for strategy in strategies:
        for budget in budgets:
            predict_evidence(model, tokenizer, papers[0], budget, strategy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            predictions = []
            latencies = []
            tokens = []
            for paper in papers:
                start = time.perf_counter()
                result = predict_evidence(model, tokenizer, paper, budget, strategy)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latencies.append((time.perf_counter() - start) * 1000)
                predictions.append(result.decision)
                tokens.append(result.tokens)
            rows.append(
                {
                    "strategy": strategy,
                    "budget": budget,
                    **classification_metrics(predictions, [paper.decision for paper in papers]),
                    "median_ms": statistics.median(latencies),
                    "samples_ms": latencies,
                    "mean_tokens": statistics.mean(tokens),
                    "peak_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                }
            )
    return {
        "metadata": metadata(device),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_revision": DATA_REVISION,
        "source_sha256": DATA_SHA256,
        "evaluation_pmids": [paper.pmid for paper in papers],
        "rows": rows,
    }
