from pathlib import Path

import pytest

from rag_project.eval.dataset import EvalRow, load_eval_set


def test_load_eval_set_parses_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "eval_set.jsonl"
    path.write_text(
        '{"question": "Q1?", "ground_truth": "A1"}\n{"question": "Q2?", "ground_truth": "A2"}\n',
        encoding="utf-8",
    )

    rows = load_eval_set(path)

    assert rows == [
        EvalRow(question="Q1?", ground_truth="A1"),
        EvalRow(question="Q2?", ground_truth="A2"),
    ]


def test_load_eval_set_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "eval_set.jsonl"
    path.write_text('{"question": "Q1?", "ground_truth": "A1"}\n\n\n', encoding="utf-8")

    rows = load_eval_set(path)

    assert len(rows) == 1


def test_load_eval_set_rejects_malformed_row(tmp_path: Path) -> None:
    path = tmp_path / "eval_set.jsonl"
    path.write_text(
        '{"question": "Q1?", "ground_truth": "A1"}\n{"question": "missing ground truth"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Malformed eval row at .*:2"):
        load_eval_set(path)


def test_load_eval_set_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "eval_set.jsonl"
    path.write_text("not valid json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Malformed eval row at .*:1"):
        load_eval_set(path)


def test_real_eval_set_has_20_to_30_rows() -> None:
    eval_set_path = Path(__file__).parents[2] / "src/rag_project/eval/eval_set.jsonl"

    rows = load_eval_set(eval_set_path)

    assert 20 <= len(rows) <= 30
