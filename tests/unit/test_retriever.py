from langchain_core.documents import Document
from pytest_mock import MockerFixture

from rag_project.retrieval.retriever import Retriever


def test_similarity_search_returns_chunks_ordered_by_relevance(mocker: MockerFixture) -> None:
    fake_store = mocker.Mock()
    fake_store.similarity_search_with_score.return_value = [
        (Document(page_content="A", metadata={"source_path": "a.md"}), 0.1),
        (Document(page_content="B", metadata={"source_path": "b.md"}), 0.4),
    ]
    mocker.patch("rag_project.retrieval.retriever.get_vector_store", return_value=fake_store)

    retriever = Retriever(persist_dir="unused", embedding_model="unused")
    results = retriever.similarity_search("a query", k=2)

    assert [r.content for r in results] == ["A", "B"]
    assert [r.source_path for r in results] == ["a.md", "b.md"]
    assert [r.score for r in results] == [0.1, 0.4]
    fake_store.similarity_search_with_score.assert_called_once_with("a query", k=2)


def test_similarity_search_no_match_returns_empty_list(mocker: MockerFixture) -> None:
    fake_store = mocker.Mock()
    fake_store.similarity_search_with_score.return_value = []
    mocker.patch("rag_project.retrieval.retriever.get_vector_store", return_value=fake_store)

    retriever = Retriever(persist_dir="unused", embedding_model="unused")
    results = retriever.similarity_search("no matches for this")

    assert results == []


def test_similarity_search_defaults_missing_source_path_to_empty_string(
    mocker: MockerFixture,
) -> None:
    fake_store = mocker.Mock()
    fake_store.similarity_search_with_score.return_value = [
        (Document(page_content="A", metadata={}), 0.2),
    ]
    mocker.patch("rag_project.retrieval.retriever.get_vector_store", return_value=fake_store)

    retriever = Retriever(persist_dir="unused", embedding_model="unused")
    results = retriever.similarity_search("a query")

    assert results[0].source_path == ""
