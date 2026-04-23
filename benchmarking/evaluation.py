from __future__ import annotations

from typing import Any, Callable

from .datasets import BenchmarkRecord


Evaluator = Callable[..., Any]
EVALUATORS: dict[str, Evaluator] = {}


class EvaluationRegistryError(LookupError):
    pass


def register_evaluator(kind: str, evaluator: Evaluator) -> None:
    EVALUATORS[kind] = evaluator


def evaluate_record(
    record: BenchmarkRecord,
    *,
    short_answer_text: str,
    full_response_text: str,
    judge: object,
) -> Any:
    evaluator = EVALUATORS.get(record.grading.kind)
    if evaluator is None:
        evaluator = EVALUATORS.get("generic_semantic")
    if evaluator is None:
        raise EvaluationRegistryError(
            f"No evaluator registered for '{record.grading.kind}', and 'generic_semantic' fallback is unavailable."
        )
    return evaluator(
        record,
        short_answer_text=short_answer_text,
        full_response_text=full_response_text,
        judge=judge,
    )
