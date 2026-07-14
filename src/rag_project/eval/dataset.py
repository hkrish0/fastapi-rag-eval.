from pathlib import Path

from pydantic import BaseModel, ValidationError


class EvalRow(BaseModel):
    question: str
    ground_truth: str


def load_eval_set(path: str | Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(EvalRow.model_validate_json(line))
        except ValidationError as e:
            raise ValueError(f"Malformed eval row at {path}:{line_number}: {e}") from e

    return rows
