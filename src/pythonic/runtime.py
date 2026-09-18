import hashlib
from pathlib import Path

from pythonic.agent import Agent, SearchReadClassify
from pythonic.biomedical import (
    MODEL_ID,
    MODEL_REVISION,
    EvidencePrediction,
    load_evidence_checkpoint,
    predict_text,
)
from pythonic.data import load_papers
from pythonic.retrieval import BM25
from pythonic.tools import EvidenceTools
from pythonic.tracing import TraceWriter


def evidence_agent(
    data: Path,
    checkpoint: Path,
    device: str = "cpu",
    budget: int = 256,
    trace_path: Path | None = None,
) -> Agent:
    model, tokenizer = load_evidence_checkpoint(checkpoint, device)
    sections = [section for paper in load_papers(data) for section in paper.sections]
    trace = (
        None
        if trace_path is None
        else TraceWriter(
            trace_path,
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "adapter_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "device": device,
                "policy": "search-read-classify",
                "budget": budget,
            },
        )
    )

    def predict(question: str, context: str, limit: int) -> EvidencePrediction:
        if trace is None:
            return predict_text(model, tokenizer, question, context, limit)
        with trace.span("classification", {"budget": limit}) as attributes:
            result = predict_text(model, tokenizer, question, context, limit)
            attributes["input_tokens"] = result.tokens
            attributes["decision"] = result.decision
            return result

    tools = EvidenceTools(
        BM25(sections),
        sections,
        predict,
        lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    )
    return Agent(tools, SearchReadClassify(budget), trace=trace)
