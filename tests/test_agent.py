from pythonic.agent import Agent, Finish, SearchReadClassify
from pythonic.tools import ToolCall
from test_tools import make_tools


def test_scripted_agent_finishes_with_verified_classification() -> None:
    state = Agent(make_tools(), SearchReadClassify(budget=100)).run("Does treatment reduce fever?")
    assert state.status == "complete"
    assert [call.name for call in state.calls] == ["search", "read_section", "classify"]
    assert state.answer["citations"][0]["pmid"] == "0"


def test_repeated_actions_and_turn_limits_stop_execution() -> None:
    repeated = Agent(
        make_tools(), lambda state: ToolCall(str(len(state.calls)), "search", {"query": "fever"})
    ).run("fever")
    assert repeated.reason == "Repeated action or call ID."
    assert len(repeated.calls) == 1
    bounded = Agent(make_tools(), SearchReadClassify(), max_turns=1).run("fever")
    assert bounded.reason == "Turn limit reached."
    assert Agent(make_tools(), lambda state: Finish()).run("fever").status == "stopped"


def test_agent_recovers_using_a_different_action_after_tool_error() -> None:
    policy = SearchReadClassify(budget=100)

    def select(state):
        if not state.calls:
            return ToolCall("bad", "unknown", {})
        return policy(state)

    state = Agent(make_tools(), select).run("fever")
    assert state.status == "complete"
    assert state.observations[0].error is not None
