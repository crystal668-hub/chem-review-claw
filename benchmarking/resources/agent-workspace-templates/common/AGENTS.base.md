# Benchmark Attempt Workspace

This workspace belongs only to the current attempt.

- Use only the current prompt, current input bundle, current workspace, and scopes explicitly exposed by the role policy as evidence.
- Put every generated file, download, script, structure, output, and note under `scratch/`.
- Do not read, search, or list parent directories, other workspaces, benchmark results, archives, quarantine, agent sessions, raw datasets, or verifier resources.
- Do not modify absolute paths, guess run identifiers, or search the filesystem to locate resources.
- For `exec`, omit `workdir` and enter the runner-provided scratch environment inside the command.
- Use native `exec` for normal single-line shell commands and local tools.
- If `BENCHMARK_ATTEMPT_PYTHON` is set, use that fresh environment and do not create another venv. Otherwise, create Python virtual environments under `scratch/` with `python3 -m venv --copies venv`.
- Do not use heredocs, here-strings, or inline multiline `python -c` or `node -e` scripts.
- For a multiline script, use a structured file tool to write `scratch/tmp/<name>.<ext>`, then run that file with native `exec` from the runner-provided scratch environment.
- For structured file tools, use only workspace-relative `scratch/...` paths.
- After a path error, reuse the current relative path or runner-provided environment variable. Do not try parent, sibling, or similarly named run paths.
- If the guard blocks an operation or a tool fails, correct the current operation and continue.
- Do not use an external file as benchmark evidence even if it is visible.
- Do not run Git or create `.git` metadata.
- Return the final answer in the format required by the prompt.
