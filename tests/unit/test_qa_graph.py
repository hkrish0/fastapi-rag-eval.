from pytest_mock import MockerFixture

from rag_project.graph.qa_graph import QAState, build_qa_graph
from rag_project.retrieval.retriever import RetrievedChunk


def test_graph_retrieves_then_generates_with_retrieved_context(mocker: MockerFixture) -> None:
    fake_retriever = mocker.Mock()
    fake_retriever.similarity_search.return_value = [
        RetrievedChunk(
            content="Path params use curly braces.",
            source_path="tutorial/path-params.md",
            score=0.1,
        ),
    ]
    fake_llm = mocker.Mock()
    fake_llm.invoke.return_value = mocker.Mock(text="You use curly braces in the path.")
    mocker.patch("rag_project.graph.qa_graph.ChatAnthropic", return_value=fake_llm)

    graph = build_qa_graph(fake_retriever, generation_model="unused", api_key="unused", k=2)
    result = graph.invoke(QAState(question="How do I define a path parameter?"))

    assert result["answer"] == "You use curly braces in the path."
    assert result["chunks"] == fake_retriever.similarity_search.return_value
    fake_retriever.similarity_search.assert_called_once_with(
        "How do I define a path parameter?", k=2
    )

    prompt = fake_llm.invoke.call_args[0][0][1].content
    assert "tutorial/path-params.md" in prompt
    assert "Path params use curly braces." in prompt
    assert "How do I define a path parameter?" in prompt


def test_graph_generates_with_empty_context_when_nothing_retrieved(
    mocker: MockerFixture,
) -> None:
    fake_retriever = mocker.Mock()
    fake_retriever.similarity_search.return_value = []
    fake_llm = mocker.Mock()
    fake_llm.invoke.return_value = mocker.Mock(text="The context doesn't contain the answer.")
    mocker.patch("rag_project.graph.qa_graph.ChatAnthropic", return_value=fake_llm)

    graph = build_qa_graph(fake_retriever, generation_model="unused", api_key="unused")
    result = graph.invoke(QAState(question="What is the boiling point of tungsten?"))

    assert result["chunks"] == []
    assert result["answer"] == "The context doesn't contain the answer."
