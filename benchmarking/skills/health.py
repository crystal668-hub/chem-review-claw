from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmarking.runtime import paths as runtime_paths

RunSubprocess = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class HealthRequirement:
    skill: str
    python_modules: tuple[str, ...] = ()
    pdf_backend_modules: tuple[tuple[str, ...], ...] = ()
    executables: tuple[str, ...] = ()
    api_keys: tuple[str, ...] = ()
    data_files: tuple[str, ...] = ()
    network_urls: tuple[str, ...] = ()
    network_probe_timeout_seconds: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


REQUIREMENT_OVERRIDES: dict[str, HealthRequirement] = {
    "rdkit": HealthRequirement(skill="rdkit", python_modules=("rdkit",), data_files=("skills/rdkit/SKILL.md",)),
    "chem-calculator": HealthRequirement(skill="chem-calculator", python_modules=("sympy",), data_files=("skills/chem-calculator/SKILL.md",)),
    "paper-retrieval": HealthRequirement(
        skill="paper-retrieval",
        python_modules=("requests",),
        data_files=("skills/paper-retrieval/SKILL.md",),
        network_urls=("https://api.openalex.org/works?per-page=1", "https://api.crossref.org/works?rows=0"),
    ),
    "paper-access": HealthRequirement(
        skill="paper-access",
        python_modules=("requests", "bs4"),
        data_files=("skills/paper-access/SKILL.md",),
        network_urls=("https://api.crossref.org/works?rows=0",),
    ),
    "paper-parse": HealthRequirement(
        skill="paper-parse",
        pdf_backend_modules=(("pymupdf", "fitz"),),
        data_files=("skills/paper-parse/SKILL.md",),
    ),
    "pubchem": HealthRequirement(
        skill="pubchem",
        python_modules=("requests",),
        data_files=("skills/pubchem/SKILL.md",),
        network_urls=("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/water/property/MolecularFormula/JSON",),
    ),
    "materials-project": HealthRequirement(
        skill="materials-project",
        python_modules=("mp_api",),
        api_keys=("MP_API_KEY",),
        data_files=("skills/materials-project/SKILL.md",),
    ),
    "pymatgen": HealthRequirement(skill="pymatgen", python_modules=("pymatgen",), data_files=("skills/pymatgen/SKILL.md",)),
    "ase": HealthRequirement(skill="ase", python_modules=("ase",), data_files=("skills/ase/SKILL.md",)),
    "cclib": HealthRequirement(skill="cclib", python_modules=("cclib",), data_files=("skills/cclib/SKILL.md",)),
    "xtb-cli": HealthRequirement(skill="xtb-cli", executables=("xtb",), data_files=("skills/xtb-cli/SKILL.md",)),
    "chembl-database": HealthRequirement(
        skill="chembl-database",
        python_modules=("chembl_webresource_client",),
        data_files=("skills/chembl-database/SKILL.md",),
        network_urls=("https://www.ebi.ac.uk/chembl/api/data/status.json",),
        network_probe_timeout_seconds=10,
    ),
}


@lru_cache(maxsize=4)
def _read_dotenv(path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = Path(path)
    if not env_path.is_file():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if not key:
            continue
        value = value.strip().strip("'\"")
        env[key] = value
    return env


def _health_environment(env: dict[str, str] | None) -> dict[str, str]:
    environment = os.environ.copy() if env is None else dict(env)
    for key, value in _read_dotenv(str(runtime_paths.openclaw_env)).items():
        if value and not str(environment.get(key) or "").strip():
            environment[key] = value
    return environment


def health_requirements_for_allowlist(allowlist: Iterable[str]) -> dict[str, HealthRequirement]:
    requirements: dict[str, HealthRequirement] = {}
    for skill in allowlist:
        key = str(skill)
        requirements[key] = REQUIREMENT_OVERRIDES.get(
            key,
            HealthRequirement(skill=key, data_files=(f"skills/{key}/SKILL.md",)),
        )
    return requirements


def check_skill_health(
    requirement: HealthRequirement,
    *,
    workspace_root: Path,
    env: dict[str, str] | None = None,
    run_subprocess: RunSubprocess = subprocess.run,
    network_timeout_seconds: int = 3,
) -> dict[str, Any]:
    environment = _health_environment(env)
    checks: dict[str, Any] = {"python_modules": {}, "pdf_backends": {}, "executables": {}, "api_keys": {}, "data_files": {}, "network": {}}
    unavailable: list[dict[str, str]] = []

    for module in requirement.python_modules:
        command = ["uv", "run", "python", "-c", f"import {module}"]
        completed = run_subprocess(
            command,
            cwd=str(workspace_root),
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=network_timeout_seconds,
        )
        ok = completed.returncode == 0
        checks["python_modules"][module] = {"ok": ok, "returncode": completed.returncode}
        if not ok:
            unavailable.append({"kind": "missing_dependency", "name": module, "reason": (completed.stderr or completed.stdout).strip()[:500]})

    for backend_modules in requirement.pdf_backend_modules:
        backend_name = "/".join(backend_modules)
        module_results: dict[str, dict[str, Any]] = {}
        backend_ok = False
        failure_reasons: list[str] = []
        for module in backend_modules:
            command = ["uv", "run", "--extra", requirement.skill, "python", "-c", f"import {module}"]
            completed = run_subprocess(
                command,
                cwd=str(workspace_root),
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=network_timeout_seconds,
            )
            ok = completed.returncode == 0
            module_results[module] = {"ok": ok, "returncode": completed.returncode}
            if ok:
                backend_ok = True
            else:
                failure_reasons.append((completed.stderr or completed.stdout).strip())
        checks["pdf_backends"][backend_name] = {"ok": backend_ok, "modules": module_results}
        if not backend_ok:
            unavailable.append(
                {
                    "kind": "missing_dependency",
                    "name": backend_name,
                    "reason": "\n".join(reason for reason in failure_reasons if reason).strip()[:500],
                }
            )

    for executable in requirement.executables:
        ok = shutil.which(executable) is not None
        checks["executables"][executable] = {"ok": ok}
        if not ok:
            unavailable.append({"kind": "missing_executable", "name": executable, "reason": f"{executable} not found in PATH"})

    for key in requirement.api_keys:
        ok = bool(str(environment.get(key) or "").strip())
        checks["api_keys"][key] = {"ok": ok}
        if not ok:
            unavailable.append({"kind": "missing_api_key", "name": key, "reason": f"{key} is not set"})

    for relative_path in requirement.data_files:
        ok = (workspace_root / relative_path).is_file()
        checks["data_files"][relative_path] = {"ok": ok}
        if not ok:
            unavailable.append({"kind": "missing_data_file", "name": relative_path, "reason": f"{relative_path} does not exist"})

    probe_timeout_seconds = requirement.network_probe_timeout_seconds or network_timeout_seconds
    for url in requirement.network_urls:
        command = [
            "uv",
            "run",
            "python",
            "-c",
            "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=float(sys.argv[2])).read(64)",
            url,
            str(probe_timeout_seconds),
        ]
        completed = run_subprocess(
            command,
            cwd=str(workspace_root),
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=probe_timeout_seconds + 2,
        )
        ok = completed.returncode == 0
        checks["network"][url] = {"ok": ok, "returncode": completed.returncode, "timeout_seconds": probe_timeout_seconds}
        if not ok:
            unavailable.append({"kind": "provider_failure", "name": url, "reason": (completed.stderr or completed.stdout).strip()[:500]})

    return {
        "skill": requirement.skill,
        "available": not unavailable,
        "checks": checks,
        "unavailable_reasons": unavailable,
    }


def check_all_skill_health(
    allowlist: Iterable[str],
    *,
    workspace_root: Path,
    env: dict[str, str] | None = None,
    run_subprocess: RunSubprocess = subprocess.run,
) -> dict[str, dict[str, Any]]:
    return {
        skill: check_skill_health(requirement, workspace_root=workspace_root, env=env, run_subprocess=run_subprocess)
        for skill, requirement in health_requirements_for_allowlist(allowlist).items()
    }


def summarize_skill_health(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    available = sorted(skill for skill, report in reports.items() if report.get("available") is True)
    unavailable = sorted(skill for skill, report in reports.items() if report.get("available") is not True)
    return {
        "available_skill_count": len(available),
        "unavailable_skill_count": len(unavailable),
        "available_skills": available,
        "unavailable_skills": unavailable,
    }
