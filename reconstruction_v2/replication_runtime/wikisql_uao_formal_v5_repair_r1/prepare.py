"""Prepare the append-only repaired-v5 WikiSQL UAO deployment.

The content-addressed v5 deployment builder is loaded in isolated module
state and rebound to the repaired sibling root.  It continues to consume the
existing source-custody receipt and passed runtime-qualification template.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Sequence

from replication_runtime.wikisql_uao_formal_v5_repair_r1 import (
    runner as formal,
)


_PREPARE_SOURCE = (
    Path(__file__).parents[1] / "wikisql_uao_formal_v5/prepare.py"
)
_ISOLATED_PREPARE_NAME = (
    "replication_runtime.wikisql_uao_formal_v5_repair_r1."
    "_isolated_formal_v5_prepare"
)
_PREPARE_SPEC = importlib.util.spec_from_file_location(
    _ISOLATED_PREPARE_NAME,
    _PREPARE_SOURCE,
)
if _PREPARE_SPEC is None or _PREPARE_SPEC.loader is None:
    raise ImportError("frozen formal v5 deployment builder cannot be isolated")
_base_prepare = importlib.util.module_from_spec(_PREPARE_SPEC)
sys.modules[_ISOLATED_PREPARE_NAME] = _base_prepare
_PREPARE_SPEC.loader.exec_module(_base_prepare)

PREPARE_SCHEMA = "wikisql_uao_formal_v5_repair_r1_deployment_prepare_v1"
_base_prepare.formal = formal
_base_prepare.PREPARE_SCHEMA = PREPARE_SCHEMA

EXPECTED_DESIGN_SELF_SHA256 = _base_prepare.EXPECTED_DESIGN_SELF_SHA256
WikiSQLUAOFormalV5RepairPrepareError = (
    _base_prepare.WikiSQLUAOFormalV5PrepareError
)
build_config = _base_prepare.build_config
prepare = _base_prepare.prepare


def main(argv: Sequence[str] | None = None) -> int:
    parser = _base_prepare.argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-template",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--source-archive",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--source-custody",
        required=True,
        type=Path,
    )
    arguments = parser.parse_args(argv)
    receipt = prepare(
        arguments.runtime_template,
        arguments.source_archive,
        arguments.source_custody,
    )
    print(
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
