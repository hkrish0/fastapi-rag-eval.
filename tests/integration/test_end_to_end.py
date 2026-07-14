import pytest
from fastapi.testclient import TestClient

from rag_project.api.main import app
from rag_project.config import get_settings
from rag_project.graph.qa_graph import QAState, build_qa_graph
from rag_project.retrieval.retriever import Retriever

pytestmark = pytest.mark.integration

REFUSAL_PHRASES = [
    "cannot answer",
    "can't answer",
    "doesn't contain",
    "does not contain",
    "no information",
    "don't have information",
    "not contain any information",
    "unable to answer",
]


def test_retriever_returns_relevant_chunk_for_known_query() -> None:
    settings = get_settings()
    retriever = Retriever(settings.chroma_persist_dir, settings.embedding_model)

    results = retriever.similarity_search("How do I define a path parameter?", k=4)

    assert results
    assert any("path-params" in r.source_path for r in results)


def test_qa_graph_returns_grounded_cited_answer_for_real_question() -> None:
    settings = get_settings()
    retriever = Retriever(settings.chroma_persist_dir, settings.embedding_model)
    graph = build_qa_graph(retriever, settings.generation_model, settings.anthropic_api_key)

    result = graph.invoke(QAState(question="How do I define a path parameter in FastAPI?"))

    assert result["answer"]
    assert result["chunks"]
    assert any("path-params" in c.source_path for c in result["chunks"])


def test_qa_graph_declines_when_context_lacks_the_answer() -> None:
    settings = get_settings()
    retriever = Retriever(settings.chroma_persist_dir, settings.embedding_model)
    graph = build_qa_graph(retriever, settings.generation_model, settings.anthropic_api_key)

    result = graph.invoke(QAState(question="What is the boiling point of tungsten?"))

    lowered = result["answer"].lower()
    assert any(phrase in lowered for phrase in REFUSAL_PHRASES)


def test_query_endpoint_returns_grounded_answer_end_to_end() -> None:
    client = TestClient(app)

    response = client.post(
        "/query", json={"question": "How do I define a path parameter in FastAPI?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["sources"]
