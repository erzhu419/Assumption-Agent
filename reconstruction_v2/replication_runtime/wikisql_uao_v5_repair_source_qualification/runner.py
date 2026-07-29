"""Qualify the real WikiSQL archive before the repaired-v5 effect freeze.

This command is deliberately outside the formal effect runtime.  It may be
iterated while adapting public source formats, but it cannot create a
selection secret, select a cohort, launch an action, or score an item.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v1 as base,
)
from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v5_repair as repair,
)
from assumption_agent.benchmarks import wikisql_uao_reality_v1 as reality


def qualify(
    archive_path: Path,
    *,
    expected_archive_sha256: str,
    output_path: Path,
    production_runtime: bool,
) -> dict[str, object]:
    config = (
        repair.CompilerConfig.production()
        if production_runtime
        else repair.CompilerConfig.synthetic_test(
            a_form_quota_per_family=64,
            a_hold_quota_per_family=24,
        )
    )
    receipt = dict(
        repair.qualify_archive(
            archive_path,
            expected_archive_sha256=expected_archive_sha256,
            config=config,
        )
    )
    if (
        receipt.get("secret_generation_count") != 0
        or receipt.get("HMAC_selection_count") != 0
        or receipt.get("cohort_selection_count") != 0
        or receipt.get("action_count") != 0
        or receipt.get("scorer_count") != 0
        or receipt.get("score_count") != 0
        or receipt.get("API_or_online_evaluation_count") != 0
    ):
        raise repair.WikiSQLSourceCompilerError(
            "source qualification crossed the non-effect boundary"
        )
    raw = reality.canonical_json_bytes(receipt)
    base._exclusive_write(output_path, raw, mode=0o600)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--production-runtime",
        action="store_true",
        help="require the frozen production archive identity and Babel 2.10.3",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = qualify(
        arguments.archive,
        expected_archive_sha256=arguments.sha256,
        output_path=arguments.output,
        production_runtime=arguments.production_runtime,
    )
    print(receipt["self_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
