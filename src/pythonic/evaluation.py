import time
from collections.abc import Sequence
from typing import Any

from pythonic.agent import Agent
from pythonic.context import Citation, validate_citations
from pythonic.data import Paper
from pythonic.metrics import classification_metrics
from pythonic.retrieval import retrieval_metrics


def evaluate_agent(agent: Agent, papers: Sequence[Paper]) -> dict[str, Any]:
    if not papers:
        raise ValueError("Agent evaluation needs questions.")
    rows = []
    predictions = []
    for paper in papers:
        start = time.perf_counter()
        state = agent.run(paper.question)
        valid = False
        decision = None
        if state.status == "complete" and state.answer:
            decision = state.answer["decision"]
            citations = [Citation(**item) for item in state.answer["citations"]]
            try:
                validate_citations(citations, agent.tools.sources)
                valid = bool(citations)
            except ValueError:
                pass
        predictions.append(decision)
        rows.append(
            {
                "pmid": paper.pmid,
                "status": state.status,
                "prediction": decision,
                "gold": paper.decision,
                "citation_provenance": valid,
                "correct": decision == paper.decision,
                "tool_calls": len(state.calls),
                "tool_errors": sum(result.error is not None for result in state.observations),
                "latency_ms": (time.perf_counter() - start) * 1000,
            }
        )
    return {
        "source_digest": agent.tools.source_digest,
        "classification": classification_metrics(predictions, [paper.decision for paper in papers]),
        "retrieval": retrieval_metrics(agent.tools.retriever, papers),
        "completion_rate": sum(row["status"] == "complete" for row in rows) / len(rows),
        "citation_provenance_rate": sum(row["citation_provenance"] for row in rows) / len(rows),
        "task_success": sum(row["correct"] and row["citation_provenance"] for row in rows)
        / len(rows),
        "rows": rows,
    }
