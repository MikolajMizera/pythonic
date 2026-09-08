import json
from pathlib import Path

import pytest
from test_tools import make_tools

from pythonic.agent import Agent, AgentState, SearchReadClassify
from pythonic.checkpoints import load_checkpoint, save_checkpoint


def test_agent_resumes_without_replaying_completed_tools(tmp_path: Path) -> None:
    tools = make_tools()
    agent = Agent(tools, SearchReadClassify(budget=100))
    state = agent.step(AgentState("Does treatment reduce fever?"))
    path = tmp_path / "state.json"
    save_checkpoint(state, path, tools.source_digest)
    resumed = agent.resume(path)
    assert resumed.status == "complete"
    assert [call.name for call in resumed.calls].count("search") == 1
    expected = agent.run(state.question)
    assert json.dumps(resumed.answer, sort_keys=True) == json.dumps(expected.answer, sort_keys=True)
    assert load_checkpoint(path, tools.source_digest).status == "complete"
    with pytest.raises(ValueError):
        load_checkpoint(path, "changed-source")


def test_checkpoint_rejects_incomplete_observations(tmp_path: Path) -> None:
    tools = make_tools()
    state = Agent(tools, SearchReadClassify()).step(AgentState("fever"))
    path = tmp_path / "state.json"
    save_checkpoint(state, path, tools.source_digest)
    content = json.loads(path.read_text())
    content["state"]["observations"] = []
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_checkpoint(path, tools.source_digest)
