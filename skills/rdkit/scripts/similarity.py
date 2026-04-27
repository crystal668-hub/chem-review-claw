from __future__ import annotations

from rdkit_capabilities import similarity
from rdkit_skill_common import run_named_capability


if __name__ == "__main__":
    raise SystemExit(run_named_capability("similarity", similarity))
