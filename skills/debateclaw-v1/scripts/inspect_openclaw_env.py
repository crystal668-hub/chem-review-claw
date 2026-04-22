#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///

"""Inspect OpenClaw .env variable names without reading or printing secret values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openclaw_debate_common import FAMILY_SPECS, classify_names, parse_env_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect OpenClaw .env variable names without printing values.")
    parser.add_argument(
        "--env-file",
        default=str(Path.home() / ".openclaw" / ".env"),
        help="Path to the OpenClaw .env file.",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(FAMILY_SPECS.keys()),
        help="Inspect only the named provider/model family. Repeat for multiple families.",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero if any inspected family lacks both an API key name and a base URL name.",
    )
    return parser.parse_args()

def render_text(payload: dict[str, object]) -> str:
    lines = [
        f"Env file: {payload['env_file']}",
        f"Exists: {'yes' if payload['exists'] else 'no'}",
        "Note: values were not printed; this report only inspects variable names.",
        "",
    ]
    for family in payload["families"]:
        lines.append(f"[{family['family']}] status={family['status']}")
        lines.append("  aliases: " + ", ".join(family["aliases"]))
        lines.append("  api key names: " + (", ".join(family["api_key_names"]) or "none"))
        lines.append("  base url names: " + (", ".join(family["base_url_names"]) or "none"))
        lines.append("  suggested standard api key: " + family["standard_names"]["api_key"])
        lines.append("  suggested standard base url: " + family["standard_names"]["base_url"])
        if family["missing_standard_names"]:
            lines.append("  missing standard names: " + ", ".join(family["missing_standard_names"]))
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> int:
    args = parse_args()
    env_file = Path(args.env_file).expanduser().resolve()
    names = parse_env_names(env_file)

    family_names = args.family or list(FAMILY_SPECS.keys())
    families = [classify_names(names, FAMILY_SPECS[name]) for name in family_names]
    payload = {
        "env_file": str(env_file),
        "exists": env_file.is_file(),
        "inspected_variable_count": len(names),
        "families": families,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_text(payload))

    if args.require_complete:
        incomplete = [
            family["family"]
            for family in families
            if family["status"] != "complete"
        ]
        if incomplete:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
