from __future__ import annotations

from typing import Any, Protocol

from .datasets import BenchmarkRecord


class RuntimeBundleLike(Protocol):
    bundle_dir: Any
    question_markdown: Any
    image_files: list[Any]


def resolve_chemqa_answer_kind(record: BenchmarkRecord) -> str:
    eval_kind = str(getattr(record, "eval_kind", "") or "").strip()
    dataset = str(getattr(record, "dataset", "") or "").strip()
    payload = dict(getattr(record, "payload", {}) or {})
    config = dict(getattr(getattr(record, "grading", None), "config", {}) or {})
    explicit = str(payload.get("answer_kind") or config.get("answer_kind") or "").strip()
    if explicit:
        return explicit
    if eval_kind in {"chembench_open_ended", "frontierscience_olympiad"}:
        return "numeric_short_answer"
    if eval_kind == "frontierscience_research" or str(config.get("track") or payload.get("track") or "").strip().lower() == "research":
        return "multi_part_research_answer"
    if eval_kind == "superchem_multiple_choice_rpf":
        return "multiple_choice"
    if eval_kind == "conformabench_constructive":
        return "structure_answer"
    if dataset == "superchem" and isinstance(config.get("options") or payload.get("options"), dict):
        return "multiple_choice"
    return "generic_semantic_answer"


def build_single_llm_prompt(
    record: BenchmarkRecord,
    *,
    websearch_enabled: bool,
    input_bundle: RuntimeBundleLike | None = None,
) -> str:
    instructions = [
        "You are answering a chemistry benchmark question.",
        "Be careful, concise, and do not fabricate missing facts.",
    ]
    if websearch_enabled:
        instructions.append("You may use web search if it is genuinely helpful.")
    else:
        instructions.append("Do not use web search or external browsing.")

    if record.eval_kind == "superchem_multiple_choice_rpf":
        instructions.append("This is a chemistry multiple-choice question.")
        instructions.append("Show concise reasoning, then end with exactly one line formatted as: FINAL ANSWER: <option letters>.")
        instructions.append("If multiple options are correct, separate the letters with `|`.")
        if input_bundle is not None:
            instructions.append(f"Local file bundle: {input_bundle.bundle_dir}")
            instructions.append(f"Read the question bundle file first: {input_bundle.question_markdown}")
            if input_bundle.image_files:
                instructions.append("Inspect the local image files referenced in the bundle before answering.")
    elif record.eval_kind == "chembench_open_ended":
        instructions.append("Show brief reasoning if needed, then end with exactly one line formatted as: FINAL ANSWER: <answer>.")
    elif record.eval_kind == "frontierscience_olympiad":
        instructions.append("End with exactly one line formatted as: FINAL ANSWER: <answer>.")
    elif record.eval_kind == "conformabench_constructive":
        instructions.append("Propose one chemically valid molecule and end with exactly one line formatted as: FINAL ANSWER: <SMILES>.")
    else:
        instructions.append("Provide a complete answer. If you include a final answer line, use: FINAL ANSWER: <answer>.")

    return "\n".join(instructions) + "\n\nQUESTION:\n" + record.prompt.strip()


def build_chemqa_goal(
    record: BenchmarkRecord,
    *,
    websearch_enabled: bool,
    input_bundle: RuntimeBundleLike | None = None,
) -> str:
    instructions = [
        "Solve the following chemistry benchmark question.",
        "Return a final answer that is faithful to the prompt.",
    ]
    if websearch_enabled:
        instructions.append("Web search may be used if helpful.")
    else:
        instructions.append("Do not use web search or external browsing.")
    if record.eval_kind == "superchem_multiple_choice_rpf":
        instructions.append("This is a multiple-choice chemistry question.")
        instructions.append("End with a line `FINAL ANSWER: <option letters>`.")
        instructions.append("If multiple options are correct, separate the letters with `|`.")
        if input_bundle is not None:
            instructions.append(f"Use the local file bundle at `{input_bundle.bundle_dir}`.")
            instructions.append(f"Open `{input_bundle.question_markdown}` first and inspect any referenced images.")
    elif record.eval_kind == "conformabench_constructive":
        instructions.append("End with exactly one line `FINAL ANSWER: <SMILES>`.")
    elif record.eval_kind in {"chembench_open_ended", "frontierscience_olympiad"}:
        instructions.append("If appropriate, end with a line `FINAL ANSWER: <answer>`.")
    instructions.append(f"ChemQA Artifact Flow answer kind: {resolve_chemqa_answer_kind(record)}.")
    return "\n".join(instructions) + "\n\nQUESTION:\n" + record.prompt.strip()
