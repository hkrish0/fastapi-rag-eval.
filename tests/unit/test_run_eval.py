import json
from pathlib import Path

from pytest_mock import MockerFixture

from rag_project.eval.dataset import EvalRow
from rag_project.eval.run_eval import (
    build_eval_rows,
    build_report,
    run_eval,
    score_eval_rows,
    write_report,
)
from rag_project.retrieval.retriever import RetrievedChunk


def test_build_eval_rows_maps_question_and_graph_output(mocker: MockerFixture) -> None:
    fake_graph = mocker.Mock()
    fake_graph.invoke.return_value = {
        "answer": "Use HTTPException.",
        "chunks": [
            RetrievedChunk(content="chunk text", source_path="a.md", score=0.1),
        ],
    }
    rows = [EvalRow(question="How do I raise an error?", ground_truth="Raise HTTPException.")]

    eval_rows = build_eval_rows(rows, fake_graph)

    assert eval_rows == [
        {
            "user_input": "How do I raise an error?",
            "response": "Use HTTPException.",
            "retrieved_contexts": ["chunk text"],
            "reference": "Raise HTTPException.",
        }
    ]
    fake_graph.invoke.assert_called_once()


def test_build_report_computes_averages_and_reshapes_rows() -> None:
    scored_rows = [
        {
            "user_input": "Q1",
            "response": "A1",
            "reference": "R1",
            "retrieved_contexts": ["c1"],
            "faithfulness": 1.0,
            "answer_relevancy": 0.8,
            "context_precision": 0.5,
            "context_recall": 0.0,
        },
        {
            "user_input": "Q2",
            "response": "A2",
            "reference": "R2",
            "retrieved_contexts": ["c2"],
            "faithfulness": 0.0,
            "answer_relevancy": 0.6,
            "context_precision": 1.0,
            "context_recall": 1.0,
        },
    ]

    report = build_report(scored_rows)

    assert report["num_questions"] == 2
    assert report["average_scores"] == {
        "faithfulness": 0.5,
        "answer_relevancy": 0.7,
        "context_precision": 0.75,
        "context_recall": 0.5,
    }
    assert report["results"][0] == {
        "question": "Q1",
        "answer": "A1",
        "reference": "R1",
        "retrieved_contexts": ["c1"],
        "faithfulness": 1.0,
        "answer_relevancy": 0.8,
        "context_precision": 0.5,
        "context_recall": 0.0,
    }
    assert "generated_at" in report


def test_write_report_creates_timestamped_json_file(tmp_path: Path) -> None:
    report = {"num_questions": 1, "average_scores": {}, "results": []}

    report_path = write_report(report, tmp_path)

    assert report_path.parent == tmp_path
    assert report_path.name.startswith("eval_report_")
    assert report_path.name.endswith(".json")
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_write_report_is_safe_to_call_twice_without_overwriting(tmp_path: Path) -> None:
    first = write_report({"a": 1}, tmp_path)
    second = write_report({"a": 2}, tmp_path)

    assert first.exists()
    assert second.exists()


def test_score_eval_rows_returns_records_from_ragas_evaluate(mocker: MockerFixture) -> None:
    fake_result = mocker.Mock()
    fake_result.to_pandas.return_value.to_dict.return_value = [
        {"user_input": "Q1", "faithfulness": 1.0}
    ]
    fake_evaluate = mocker.patch("rag_project.eval.run_eval.evaluate", return_value=fake_result)
    mocker.patch(
        "rag_project.eval.run_eval.EvaluationDataset.from_list", return_value=mocker.Mock()
    )

    records = score_eval_rows(
        [{"user_input": "Q1"}], judge_llm=mocker.Mock(), judge_embeddings=mocker.Mock()
    )

    assert records == [{"user_input": "Q1", "faithfulness": 1.0}]
    fake_evaluate.assert_called_once()
    assert len(fake_evaluate.call_args.kwargs["metrics"]) == 4


def test_run_eval_applies_limit_and_writes_report(mocker: MockerFixture) -> None:
    rows = [
        EvalRow(question="Q1", ground_truth="R1"),
        EvalRow(question="Q2", ground_truth="R2"),
        EvalRow(question="Q3", ground_truth="R3"),
    ]
    mocker.patch("rag_project.eval.run_eval.get_settings")
    mocker.patch("rag_project.eval.run_eval.load_eval_set", return_value=rows)
    mocker.patch("rag_project.eval.run_eval.Retriever")
    mocker.patch("rag_project.eval.run_eval.build_qa_graph")
    mocker.patch("rag_project.eval.run_eval.ChatAnthropic")
    mocker.patch("rag_project.eval.run_eval.HuggingFaceEmbeddings")
    mocker.patch("rag_project.eval.run_eval.LangchainLLMWrapper")
    mocker.patch("rag_project.eval.run_eval.LangchainEmbeddingsWrapper")
    fake_build_eval_rows = mocker.patch(
        "rag_project.eval.run_eval.build_eval_rows", return_value=["eval_row"]
    )
    mocker.patch("rag_project.eval.run_eval.score_eval_rows", return_value=["scored_row"])
    fake_build_report = mocker.patch(
        "rag_project.eval.run_eval.build_report", return_value={"report": True}
    )
    fake_write_report = mocker.patch(
        "rag_project.eval.run_eval.write_report", return_value=Path("reports/eval/x.json")
    )

    result = run_eval(eval_set_path="unused", report_dir="unused", limit=2)

    assert result == Path("reports/eval/x.json")
    fake_build_eval_rows.assert_called_once()
    assert len(fake_build_eval_rows.call_args.args[0]) == 2
    fake_build_report.assert_called_once_with(["scored_row"])
    fake_write_report.assert_called_once_with({"report": True}, "unused")
