#!/usr/bin/env python3
"""Render DebateClaw ClawTeam templates for the supported workflows."""

from __future__ import annotations

import json
import textwrap


DEFAULT_RUNTIME_ROOT = "~/.clawteam/debateclaw/bin"


def slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("-")
    collapsed = "".join(cleaned).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed or "team"


def template_name_for(workflow: str, team_name: str | None = None) -> str:
    base = f"debate-{workflow}"
    if not team_name:
        return base
    return f"{base}-{slugify(team_name)}"


def _quote_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _quote_multiline(value: str) -> str:
    return _quote_string(value.strip())


def _template_header(name: str, description: str, command: str, backend: str) -> str:
    return textwrap.dedent(
        f"""
        [template]
        name = {_quote_string(name)}
        description = {_quote_string(description)}
        command = [{_quote_string(command)}]
        backend = {_quote_string(backend)}
        """
    ).strip()


def _render_command(command: list[str]) -> str:
    return json.dumps(command, ensure_ascii=True)


def _leader_section(task: str, *, command: list[str] | None = None) -> str:
    lines = [
        "",
        "[template.leader]",
        f'name = {_quote_string("debate-coordinator")}',
        f'type = {_quote_string("debate-coordinator")}',
    ]
    if command:
        lines.append(f"command = {_render_command(command)}")
    lines.append(f"task = {_quote_multiline(task)}")
    return "\n".join(lines).rstrip()


def _agent_section(name: str, agent_type: str, task: str, *, command: list[str] | None = None) -> str:
    lines = [
        "",
        "[[template.agents]]",
        f"name = {_quote_string(name)}",
        f"type = {_quote_string(agent_type)}",
    ]
    if command:
        lines.append(f"command = {_render_command(command)}")
    lines.append(f"task = {_quote_multiline(task)}")
    return "\n".join(lines).rstrip()


def _task_section(subject: str, owner: str) -> str:
    return textwrap.dedent(
        f"""
        [[template.tasks]]
        subject = {_quote_string(subject)}
        owner = {_quote_string(owner)}
        """
    ).rstrip()


def _parallel_leader_task(runtime_root: str) -> str:
    return textwrap.dedent(
        f"""
        You are the DebateClaw coordinator for an evidence-first parallel proposal debate.
        The debate objective is: {{goal}}

        You are not the final judge. The outer entry agent will read the completed debate
        record and choose or synthesize the final answer for the user.

        Runtime commands:
        - `{runtime_root}/debate_state.py status --team {{team_name}}`
        - `{runtime_root}/debate_state.py advance --team {{team_name}} --agent {{agent_name}}`
        - `{runtime_root}/debate_state.py summary --team {{team_name}}`

        Protocol:
        1. Keep your ClawTeam task in progress until the debate state reaches `done`.
        2. Do not write the final answer yourself. Your job is only to coordinate protocol completion.
        3. Poll the debate state. When all proposer agents have submitted exactly one proposal for the
           current epoch, run the `advance` command once.
        4. After the state reaches `done`, create `coordinator-summary.md` in your working directory
           with:
           - proposal count
           - proposal owners
           - evidence-policy reminder
           - any obvious gaps or unresolved assumptions
        5. Send a short completion note to the outer coordinator inbox:
           `clawteam inbox send {{team_name}} debate-coordinator "Parallel proposal stage completed; summary written to coordinator-summary.md."`
        6. Mark your task completed only after the state says `done`.

        If nothing is ready to advance, wait briefly, then poll again.
        """
    ).strip()


def _parallel_proposer_task(runtime_root: str) -> str:
    return textwrap.dedent(
        f"""
        You are an evidence-first proposer in a DebateClaw parallel proposal workflow.
        The debate objective is: {{goal}}

        Core rules:
        - Evidence first. Prefer verifiable repo facts, supplied documents, or cited external sources
          that are allowed by the evidence policy in the debate state.
        - Unsupported ideas are allowed only when clearly labeled as hypotheses, assumptions, or
          open questions.
        - Do not respond with a one-shot paragraph and stop. Work iteratively: inspect, compare,
          gather evidence, refine, then submit.
        - The outer entry agent is the final judge. Your job is to produce the strongest proposal you can.

        Runtime commands:
        - `{runtime_root}/debate_state.py status --team {{team_name}} --agent {{agent_name}}`
        - `{runtime_root}/debate_state.py next-action --team {{team_name}} --agent {{agent_name}} --json`
        - `{runtime_root}/debate_state.py submit-proposal --team {{team_name}} --agent {{agent_name}} --file proposal.md`

        Required procedure:
        1. Set your task to `in_progress`.
        2. Read the debate state before starting work.
        3. Investigate iteratively until you have an evidence-backed proposal.
        4. Write `proposal.md` in your current working directory using this structure:
           - first line: `Title: ...`
           - sections: `Claim`, `Evidence`, `Reasoning`, `Assumptions`, `Open Questions`
        5. Submit the file with `submit-proposal`.
        6. Send a one-line inbox update to `debate-coordinator`.
        7. Mark your task completed after your proposal is accepted by the state runtime.
        """
    ).strip()


def default_task_bundle(*, workflow: str, proposer_count: int, runtime_root: str = DEFAULT_RUNTIME_ROOT) -> dict[str, str]:
    if workflow == "parallel-judge":
        bundle = {"debate-coordinator": _parallel_leader_task(runtime_root)}
        for index in range(1, proposer_count + 1):
            bundle[f"proposer-{index}"] = _parallel_proposer_task(runtime_root)
        return bundle
    if workflow in {"review-loop", "chemqa-review"}:
        bundle = {"debate-coordinator": _review_loop_leader_task(runtime_root)}
        for index in range(1, proposer_count + 1):
            bundle[f"proposer-{index}"] = _review_loop_proposer_task(runtime_root)
        return bundle
    raise ValueError(f"Unsupported workflow for default task bundle: {workflow}")


def build_parallel_judge_template(
    *,
    name: str,
    proposer_count: int,
    command: str,
    backend: str,
    runtime_root: str = DEFAULT_RUNTIME_ROOT,
    agent_commands: dict[str, list[str]] | None = None,
    task_overrides: dict[str, str] | None = None,
) -> str:
    agent_commands = agent_commands or {}
    task_overrides = task_overrides or {}
    parts = [
        _template_header(
            name=name,
            description=f"Evidence-first parallel proposal debate with {proposer_count} proposers and outer-agent judging",
            command=command,
            backend=backend,
        ),
        _leader_section(
            task_overrides.get("debate-coordinator", _parallel_leader_task(runtime_root)),
            command=agent_commands.get("debate-coordinator"),
        ),
    ]

    for index in range(1, proposer_count + 1):
        agent_name = f"proposer-{index}"
        parts.append(
            _agent_section(
                agent_name,
                "debate-proposer",
                task_overrides.get(agent_name, _parallel_proposer_task(runtime_root)),
                command=agent_commands.get(agent_name),
            )
        )

    parts.append(
        _task_section(
            "Coordinate the DebateClaw parallel proposal workflow until all proposals are recorded",
            "debate-coordinator",
        )
    )
    for index in range(1, proposer_count + 1):
        parts.append(
            _task_section(
                f"Develop and submit an evidence-first proposal as proposer-{index}",
                f"proposer-{index}",
            )
        )
    return "\n".join(parts).strip() + "\n"


def _review_loop_leader_task(runtime_root: str) -> str:
    return textwrap.dedent(
        f"""
        You are the DebateClaw protocol coordinator for an evidence-first review/rebuttal debate.
        The debate objective is: {{goal}}

        You are not the final judge. Your job is to move the protocol forward, not to decide the
        final answer. The outer entry agent will select or synthesize the final answer after the
        debate reaches `done`.

        Runtime commands:
        - `{runtime_root}/debate_state.py next-action --team {{team_name}} --agent {{agent_name}} --json`
        - `{runtime_root}/debate_state.py status --team {{team_name}} --json`
        - `{runtime_root}/debate_state.py advance --team {{team_name}} --agent {{agent_name}}`
        - `{runtime_root}/debate_state.py summary --team {{team_name}}`

        Protocol:
        1. Keep your task `in_progress` until the debate state reaches `done`.
        2. Poll `next-action --json`.
        3. If `action` is `advance`, run `advance` exactly once, then inspect the new state.
        4. If `action` is `wait`, use `phase_progress` / `advance_ready` from `next-action` or `status --json`
           to understand what the runtime is still waiting on, then sleep briefly and poll again.
        5. If all proposals fail in an epoch, the runtime will reopen `propose` for the next epoch.
        6. When the debate reaches `done`, write `coordinator-summary.md` in your working directory with:
           - current epoch
           - final candidates
           - proposals that failed and why
           - any unresolved evidence gaps
        7. Mark your task completed only after the debate state says `done`.

        Do not guess whether a phase is complete from partial text output. Use the machine-readable
        `next-action` / `status --json` payloads to decide when advancement is ready.
        """
    ).strip()


def _review_loop_proposer_task(runtime_root: str) -> str:
    return textwrap.dedent(
        f"""
        You are a DebateClaw proposer in an evidence-first review/rebuttal workflow.
        The debate objective is: {{goal}}

        Core rules:
        - Evidence first. Prefer verifiable facts and explicitly allowed sources.
        - Hypotheses are allowed only when labeled as uncertain.
        - You must work iteratively. Do not answer once and exit.
        - Treat `debate_state.py next-action --json` as the protocol source of truth.
        - The proposal owner decides whether the proposal has failed. If you can no longer defend your
          own proposal after reviewing the attacks, concede it yourself.
        - Conceding or failing your own proposal does **not** end your participation. You must keep
          reviewing, waiting, and following `next-action` until it returns `stop`.
        - Review rounds are full cross-review rounds: review only the listed target proposals when asked,
          and never review your own proposal.
        - Rebuttals should be structured by attack theme, not by repeating the same answer separately
          for every reviewer.
        - If the same tool call fails twice with the same validation or schema error, stop retrying and
          report the blocker to `debate-coordinator` instead of looping.

        Runtime commands:
        - `{runtime_root}/debate_state.py status --team {{team_name}} --agent {{agent_name}}`
        - `{runtime_root}/debate_state.py next-action --team {{team_name}} --agent {{agent_name}} --json`
        - `{runtime_root}/debate_state.py submit-proposal --team {{team_name}} --agent {{agent_name}} --file proposal.md`
        - `{runtime_root}/debate_state.py submit-review --team {{team_name}} --agent {{agent_name}} --target proposer-X --blocking yes|no --file review.md`
        - `{runtime_root}/debate_state.py submit-rebuttal --team {{team_name}} --agent {{agent_name}} --file rebuttal.md`
        - `{runtime_root}/debate_state.py submit-rebuttal --team {{team_name}} --agent {{agent_name}} --concede --file rebuttal.md`

        Operating loop:
        1. Set your task to `in_progress` once at the beginning and keep it there until `next-action`
           returns `stop`.
        2. Poll `next-action`.
        3. If the action is `propose`:
           - inspect the provided history of your past proposals
           - avoid repeating the same idea
           - write `proposal.md` with first line `Title: ...` and sections `Claim`, `Evidence`,
             `Reasoning`, `Assumptions`, `Open Questions`
           - submit it
        4. If the action is `review`:
           - review each listed target proposal using the `targets` / `target_proposals` payload from
             `next-action`, including prior review/rebuttal history and known attack themes
           - review only the listed targets for this round; never review your own proposal
           - create one review file per target
           - include a `Summary` section plus an `Attack Points` section with bullet lines starting
             with `- `
           - use `--blocking yes` only if the review raises a substantive blocking objection
        5. If the action is `rebuttal`:
           - read the `proposal`, `current_round_reviews`, prior `review_history`, and
             `attack_registry` fields from `next-action`
           - write one rebuttal file that answers attack themes in a consolidated way
           - if you accept that your proposal fails, submit the rebuttal with `--concede`
        6. If the action is `wait`, sleep briefly and poll again. Do not mark your task completed,
           send a final "debate complete" note, or report final costs while `next-action` is still
           `wait`, `review`, or `rebuttal`.
        7. If the action is `stop`, and only then, send a concise final note to `debate-coordinator`,
           mark your task completed, and report final costs.
        """
    ).strip()


def build_review_loop_template(
    *,
    name: str,
    proposer_count: int,
    max_review_rounds: int,
    max_rebuttal_rounds: int,
    command: str,
    backend: str,
    runtime_root: str = DEFAULT_RUNTIME_ROOT,
    agent_commands: dict[str, list[str]] | None = None,
    task_overrides: dict[str, str] | None = None,
) -> str:
    agent_commands = agent_commands or {}
    task_overrides = task_overrides or {}
    description = (
        "Evidence-first proposal/review/rebuttal debate "
        f"with {proposer_count} proposers, {max_review_rounds} review rounds max, "
        f"and {max_rebuttal_rounds} rebuttal rounds max"
    )
    parts = [
        _template_header(
            name=name,
            description=description,
            command=command,
            backend=backend,
        ),
        _leader_section(
            task_overrides.get("debate-coordinator", _review_loop_leader_task(runtime_root)),
            command=agent_commands.get("debate-coordinator"),
        ),
    ]

    for index in range(1, proposer_count + 1):
        agent_name = f"proposer-{index}"
        parts.append(
            _agent_section(
                agent_name,
                "debate-proposer",
                task_overrides.get(agent_name, _review_loop_proposer_task(runtime_root)),
                command=agent_commands.get(agent_name),
            )
        )

    parts.append(
        _task_section(
            "Coordinate the DebateClaw review/rebuttal workflow until protocol completion",
            "debate-coordinator",
        )
    )
    for index in range(1, proposer_count + 1):
        parts.append(
            _task_section(
                f"Participate in the evidence-first debate loop as proposer-{index} until the coordinator stops the protocol",
                f"proposer-{index}",
            )
        )
    return "\n".join(parts).strip() + "\n"
