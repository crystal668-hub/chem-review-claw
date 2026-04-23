from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def _deep_copy_jsonish(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _deep_copy_jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_deep_copy_jsonish(item) for item in value]
    return value


def _grading_config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "preferred_score": _deep_copy_jsonish(payload.get("preferred_score")),
        "relative_tolerance": _deep_copy_jsonish(payload.get("relative_tolerance")),
        "track": _deep_copy_jsonish(payload.get("track")),
        "options": _deep_copy_jsonish(payload.get("options")),
        "reference_reasoning": _deep_copy_jsonish(payload.get("reference_reasoning")),
        "hidden_judge_spec_ref": _deep_copy_jsonish(payload.get("hidden_judge_spec_ref")),
        "modality": _deep_copy_jsonish(payload.get("modality")),
        "source_uuid": _deep_copy_jsonish(payload.get("source_uuid")),
    }


@dataclass(frozen=True)
class GradingSpec:
    kind: str
    reference_answer: str
    subset: str
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _deep_copy_jsonish(self.config))


class RecordValidationError(ValueError):
    pass


@dataclass
class BenchmarkRecord:
    record_id: str
    dataset: str
    source_file: str
    eval_kind: str
    prompt: str
    reference_answer: str
    payload: dict[str, Any]

    def __init__(
        self,
        *,
        record_id: str,
        dataset: str,
        source_file: str,
        prompt: str,
        grading: GradingSpec | None = None,
        raw_payload: dict[str, Any] | None = None,
        eval_kind: str | None = None,
        reference_answer: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        resolved_payload = _deep_copy_jsonish(payload if payload is not None else (raw_payload or {}))
        resolved_kind = str(
            eval_kind if eval_kind is not None else (grading.kind if grading is not None else resolved_payload.get("eval_kind") or "generic_semantic")
        ).strip() or "generic_semantic"
        resolved_reference = str(
            reference_answer
            if reference_answer is not None
            else (grading.reference_answer if grading is not None else (resolved_payload.get("answer") or resolved_payload.get("target") or ""))
        ).strip()
        self.record_id = record_id
        self.dataset = dataset
        self.source_file = source_file
        self.eval_kind = resolved_kind
        self.prompt = prompt
        self.reference_answer = resolved_reference
        self.payload = resolved_payload
        self._grading = grading or GradingSpec(
            kind=resolved_kind,
            reference_answer=resolved_reference,
            subset="",
            config=_grading_config_from_payload(resolved_payload),
        )

    @property
    def raw_payload(self) -> dict[str, Any]:
        return self.payload

    @property
    def grading(self) -> GradingSpec:
        return self._grading


def dataset_name_from_file(path: Path) -> str:
    return path.parent.parent.name


def classify_subset(record: BenchmarkRecord) -> str:
    subset = str(record.grading.subset or "").strip()
    if subset:
        return subset
    config = record.grading.config
    if record.dataset == "chembench":
        return "chembench"
    if record.dataset == "conformabench":
        return "conformabench"
    if record.dataset == "frontierscience":
        track = str(config.get("track") or record.payload.get("track") or "").strip().lower()
        if track == "olympiad" or record.grading.kind == "frontierscience_olympiad":
            return "frontierscience_Olympiad"
        if track == "research" or record.grading.kind == "frontierscience_research":
            return "frontierscience_Research"
    if record.dataset == "superchem":
        return "superchem_multimodal"
    return f"{record.dataset}:{record.grading.kind}"


def source_pair_key(record: BenchmarkRecord) -> str:
    source_uuid = str(record.grading.config.get("source_uuid") or record.payload.get("source_uuid") or "").strip()
    return source_uuid or record.record_id


def build_grading_spec(*, dataset: str, source_file: str, prompt: str, payload: dict[str, Any]) -> GradingSpec:
    reference_answer = str(payload.get("answer") or payload.get("target") or "").strip()
    if not reference_answer:
        raise RecordValidationError(f"Missing answer/target field in record: {payload.get('id') or dataset}")

    kind = str(payload.get("eval_kind") or "generic_semantic").strip() or "generic_semantic"
    config = {
        **_grading_config_from_payload(payload),
    }
    grading = GradingSpec(
        kind=kind,
        reference_answer=reference_answer,
        subset="",
        config=config,
    )
    preview = BenchmarkRecord(
        record_id=str(payload.get("id") or "preview"),
        dataset=dataset,
        source_file=source_file,
        prompt=prompt,
        grading=grading,
        raw_payload=payload,
    )
    return GradingSpec(
        kind=kind,
        reference_answer=reference_answer,
        subset=classify_subset(preview),
        config=config,
    )


def load_records(paths: Iterable[Path]) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for path in paths:
        dataset = dataset_name_from_file(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                record_id = str(payload.get("id") or f"{dataset}-{len(records)}")
                prompt = str(payload.get("prompt") or payload.get("problem") or payload.get("input") or payload.get("question") or "").strip()
                if not prompt:
                    raise RecordValidationError(f"Missing prompt/problem field in record: {record_id}")
                grading = build_grading_spec(
                    dataset=dataset,
                    source_file=str(path),
                    prompt=prompt,
                    payload=payload,
                )
                records.append(
                    BenchmarkRecord(
                        record_id=record_id,
                        dataset=dataset,
                        source_file=str(path),
                        prompt=prompt,
                        grading=grading,
                        raw_payload=payload,
                    )
                )
    return records
