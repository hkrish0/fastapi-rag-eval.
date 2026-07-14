from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from rag_project.api.main import app
from rag_project.retrieval.retriever import RetrievedChunk

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_and_sources(mocker: MockerFixture) -> None:
    fake_graph = mocker.Mock()
    fake_graph.invoke.return_value = {
        "answer": "Use curly braces in the path.",
        "chunks": [
            RetrievedChunk(
                content="Path params use curly braces.",
                source_path="tutorial/path-params.md",
                score=0.1,
            ),
        ],
    }
    mocker.patch("rag_project.api.main.get_qa_graph", return_value=fake_graph)

    response = client.post("/query", json={"question": "How do I define a path parameter?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use curly braces in the path."
    assert body["sources"] == [
        {
            "content": "Path params use curly braces.",
            "source_path": "tutorial/path-params.md",
            "score": 0.1,
        }
    ]


def test_query_empty_question_returns_422(mocker: MockerFixture) -> None:
    fake_get_qa_graph = mocker.patch("rag_project.api.main.get_qa_graph")

    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422
    fake_get_qa_graph.assert_not_called()


def test_query_missing_question_field_returns_422(mocker: MockerFixture) -> None:
    fake_get_qa_graph = mocker.patch("rag_project.api.main.get_qa_graph")

    response = client.post("/query", json={})

    assert response.status_code == 422
    fake_get_qa_graph.assert_not_called()
