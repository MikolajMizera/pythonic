import json
from pathlib import Path

import pytest
from test_tools import make_tools

from pythonic.agent import Agent, SearchReadClassify
from pythonic.data import fixtures
from pythonic.evaluation import evaluate_agent
from pythonic.tracing import TraceWriter


def test_agent_evaluation_separates_provenance_and_correctness(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "trace.jsonl", {"policy": "scripted"})
    agent = Agent(make_tools(), SearchReadClassify(budget=150), trace=trace)
    report = evaluate_agent(agent, fixtures())
    assert report["citation_provenance_rate"] == 1
    assert report["classification"]["accuracy"] == pytest.approx(1 / 3)
    assert report["task_success"] == pytest.approx(1 / 3)
    spans = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert len([span for span in spans if span["name"] == "tool"]) == 9
    assert all(span["duration_ms"] >= 0 for span in spans)
    assert all(span["metadata"]["policy"] == "scripted" for span in spans)
