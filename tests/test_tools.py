from pythonic.biomedical import EvidencePrediction
from pythonic.data import fixtures
from pythonic.retrieval import BM25
from pythonic.tools import EvidenceTools, ToolCall


def make_tools(
    allowed: frozenset[str] = frozenset({"search", "read_section", "classify"}),
) -> EvidenceTools:
    sections = [section for paper in fixtures() for section in paper.sections]
    return EvidenceTools(
        BM25(sections),
        sections,
        lambda q, c, b: EvidencePrediction("yes", 0.7, len(c)),
        len,
        allowed,
    )


def test_tools_validate_arguments_names_and_permissions() -> None:
    tools = make_tools()
    assert tools.execute(ToolCall("1", "shell", {})).error is not None
    assert (
        tools.execute(ToolCall("2", "search", {"query": "fever", "limit": "3"})).error is not None
    )
    assert (
        tools.execute(ToolCall("3", "read_section", {"pmid": "unknown", "section": 0})).error
        is not None
    )
    restricted = make_tools(frozenset({"search"}))
    assert (
        restricted.execute(ToolCall("4", "read_section", {"pmid": "0", "section": 0})).error
        == "Tool is not permitted."
    )
    assert [item["name"] for item in restricted.schemas()] == ["search"]


def test_classification_uses_only_source_references() -> None:
    tools = make_tools()
    result = tools.execute(
        ToolCall(
            "5",
            "classify",
            {"question": "fever?", "references": [{"pmid": "0", "section": 0}], "budget": 100},
        )
    )
    assert result.error is None
    assert result.value["citations"][0]["quote"] == fixtures()[0].sections[0].text
    assert (
        tools.execute(
            ToolCall(
                "6",
                "classify",
                {"question": "fever?", "references": [{"pmid": "forged", "section": 0}]},
            )
        ).error
        is not None
    )


def test_classifier_runtime_errors_become_tool_observations() -> None:
    tools = make_tools()

    def failed_model(question: str, context: str, budget: int) -> EvidencePrediction:
        raise RuntimeError("Model execution failed.")

    tools.predict = failed_model
    result = tools.execute(
        ToolCall(
            "7",
            "classify",
            {
                "question": "fever?",
                "references": [{"pmid": "0", "section": 0}],
                "budget": 100,
            },
        )
    )
    assert result.error == "RuntimeError: Model execution failed."
