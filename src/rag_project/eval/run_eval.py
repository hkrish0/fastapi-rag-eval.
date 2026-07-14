import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph.state import CompiledStateGraph
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from rag_project.config import get_settings
from rag_project.eval.dataset import EvalRow, load_eval_set
from rag_project.graph.qa_graph import QAState, build_qa_graph
from rag_project.retrieval.retriever import Retriever

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

DEFAULT_EVAL_SET_PATH = Path(__file__).parent / "eval_set.jsonl"
DEFAULT_REPORT_DIR = Path("reports/eval")


def _ragas_metrics() -> list[Any]:
    """Imported lazily so importing this module doesn't require instantiating
    ragas metric classes (and triggering their deprecation warnings) at import time."""
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    return [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]


def build_eval_rows(
    rows: list[EvalRow], graph: CompiledStateGraph[QAState, None, QAState, QAState]
) -> list[dict[str, Any]]:
    """Run each eval question through the QA graph to get the answer and
    retrieved contexts RAGAS needs, alongside the hand-curated reference."""
    eval_rows = []
    for row in rows:
        result = graph.invoke(QAState(question=row.question))
        eval_rows.append(
            {
                "user_input": row.question,
                "response": result["answer"],
                "retrieved_contexts": [c.content for c in result["chunks"]],
                "reference": row.ground_truth,
            }
        )
    return eval_rows


def score_eval_rows(
    eval_rows: list[dict[str, Any]], judge_llm: Any, judge_embeddings: Any
) -> list[dict[str, Any]]:
    dataset = EvaluationDataset.from_list(eval_rows)
    result = cast(
        EvaluationResult,
        evaluate(
            dataset=dataset,
            metrics=_ragas_metrics(),
            llm=judge_llm,
            embeddings=judge_embeddings,
        ),
    )
    records: list[dict[str, Any]] = result.to_pandas().to_dict(orient="records")
    return records


def build_report(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    average_scores = {
        name: sum(row[name] for row in scored_rows) / len(scored_rows) for name in METRIC_NAMES
    }
    results = [
        {
            "question": row["user_input"],
            "answer": row["response"],
            "reference": row["reference"],
            "retrieved_contexts": row["retrieved_contexts"],
            **{name: row[name] for name in METRIC_NAMES},
        }
        for row in scored_rows
    ]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "num_questions": len(scored_rows),
        "average_scores": average_scores,
        "results": results,
    }


def write_report(report: dict[str, Any], report_dir: str | Path) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"eval_report_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def run_eval(
    eval_set_path: str | Path = DEFAULT_EVAL_SET_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    limit: int | None = None,
) -> Path:
    settings = get_settings()
    rows = load_eval_set(eval_set_path)
    if limit is not None:
        rows = rows[:limit]

    retriever = Retriever(settings.chroma_persist_dir, settings.embedding_model)
    graph = build_qa_graph(retriever, settings.generation_model, settings.anthropic_api_key)
    eval_rows = build_eval_rows(rows, graph)

    judge_llm = LangchainLLMWrapper(
        ChatAnthropic(model=settings.generation_model, api_key=settings.anthropic_api_key)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )
    scored_rows = score_eval_rows(eval_rows, judge_llm, judge_embeddings)

    report = build_report(scored_rows)
    return write_report(report, report_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAGAS eval suite against the QA graph")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N questions (for a cheap smoke test)",
    )
    args = parser.parse_args()

    report_path = run_eval(limit=args.limit)
    print(f"Wrote eval report to {report_path}")


if __name__ == "__main__":
    main()
