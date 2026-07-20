"""Frozen qualifier entrypoint for the BRIGHT bridge-expansion source epoch.

The implementation reuses the previously tested aggregate-only BRIGHT
qualifier after replacing every source, manifest, demand, and output binding.
The imported base implementation is itself hash-pinned before formal access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_reasoning_retrieval_source_qualification_v1 as base,
)


SCHEMA = "bright_bridge_expansion_source_qualification_result_v1"
ATTEMPT_SCHEMA = "bright_bridge_expansion_source_qualification_attempt_v1"
FREEZE_SCHEMA = "bright_bridge_expansion_source_qualification_implementation_freeze_v1"
FAMILY_ORDER = ("PONY", "PSYCHOLOGY", "SUSTAINABLE_LIVING")
DEMANDS = {family: 68 for family in FAMILY_ORDER}

BASE_RELATIVE = Path(
    "assumption_agent/benchmarks/bright_reasoning_retrieval_source_qualification_v1.py"
)
BASE_SHA256 = "73b02635724222d6b378b54b6fec5af8a992786b35339ecf552e4d80517417f9"
CUSTODY_RELATIVE = Path("manifests/bright_bridge_expansion_source_custody_v1.json")
ACCESS_RELATIVE = Path("manifests/bright_bridge_expansion_source_access_v1.json")
DESIGN_RELATIVE = Path("manifests/bright_bridge_expansion_study_design_v1.json")
FREEZE_RELATIVE = Path(
    "manifests/bright_bridge_expansion_source_qualification_implementation_freeze_v1.json"
)
RESULT_RELATIVE = Path(
    "manifests/bright_bridge_expansion_source_qualification_result_v1.json"
)
QUALIFIER_RELATIVE = Path(
    "assumption_agent/benchmarks/bright_bridge_expansion_source_qualification_v1.py"
)
TEST_RELATIVE = Path(
    "tests/test_bright_bridge_expansion_source_qualification_v1.py"
)
ATTEMPT_ROOT_RELATIVE = Path(
    "artifacts/bright_bridge_expansion_source_qualification_v1"
)
SOURCE_ROOT_RELATIVE = Path("artifacts/bright_bridge_expansion_source_v1/dataset")

MANIFEST_BINDINGS = {
    CUSTODY_RELATIVE: {
        "file_sha256": "c4329ed93d9b7fd0ad84987804cbb7d0b30139fc25050edaac2ccc5dd701cdab",
        "self_field": "self_sha256",
        "self_sha256": "298c2195f82d1ae4e4d1184d68737fc00b8c9533ed64387deffbc53581b0ac12",
    },
    ACCESS_RELATIVE: {
        "file_sha256": "60f72017224f21e4daf71805cae6c1d30b0d845334d944c1f9c996aeee93fd9a",
        "self_field": "self_sha256",
        "self_sha256": "907e1bb96cc2136fba60e3cfef893739ddc9388ddc76cc3ecb0bbcec2dfb5d9d",
    },
    DESIGN_RELATIVE: {
        "file_sha256": "7ddfd37fc4a9e06e8e00e52db8e076f031ce457951712f1bf147d90e1109a8c3",
        "self_field": "self_sha256",
        "self_sha256": "e951ed1276efc4c28e42d7fa158889648213200888e2f5e00382ae8f5dddbc62",
    },
}

SOURCE_BINDINGS = {
    "PONY": {
        "documents": {
            "relative": Path("documents/pony-00000-of-00001.parquet"),
            "sha256": "afc23dc0a1db170b2e86364b85cbc0bc17a713b583d360668cdce0135a173fc3",
            "size": 1_125_040,
            "rows": 7_894,
        },
        "examples": {
            "relative": Path("examples/pony-00000-of-00001.parquet"),
            "sha256": "0c0718d3e0ef05da42f75b7c03e755d3e03d9fcdbd32b67dddd221768a8377d7",
            "size": 27_722,
            "rows": 112,
        },
    },
    "PSYCHOLOGY": {
        "documents": {
            "relative": Path("documents/psychology-00000-of-00001.parquet"),
            "sha256": "085d381739cb24b4227dfaf577f39d0adcad8b7b1ae74be028ac239d37be3c1d",
            "size": 11_430_533,
            "rows": 52_835,
        },
        "examples": {
            "relative": Path("examples/psychology-00000-of-00001.parquet"),
            "sha256": "404e7dff2a4528419df0bdc162541e92138e35b78918d82d3a04ade5b8f7876b",
            "size": 183_889,
            "rows": 101,
        },
    },
    "SUSTAINABLE_LIVING": {
        "documents": {
            "relative": Path("documents/sustainable_living-00000-of-00001.parquet"),
            "sha256": "474628623cf9de252bd80a7d1b667aa5070e21b87e1dd33f6723db4d24121fdf",
            "size": 11_720_059,
            "rows": 60_792,
        },
        "examples": {
            "relative": Path("examples/sustainable_living-00000-of-00001.parquet"),
            "sha256": "61f97837a16b47a0d9953039cf0b6a53d0fc5deae96a34f839b7cb5e798eb117",
            "size": 218_151,
            "rows": 108,
        },
    },
}


def _activate() -> None:
    base.SCHEMA = SCHEMA
    base.ATTEMPT_SCHEMA = ATTEMPT_SCHEMA
    base.FREEZE_SCHEMA = FREEZE_SCHEMA
    base.FAMILY_ORDER = FAMILY_ORDER
    base.DEMANDS = DEMANDS
    base.CUSTODY_RELATIVE = CUSTODY_RELATIVE
    base.ACCESS_RELATIVE = ACCESS_RELATIVE
    base.DESIGN_RELATIVE = DESIGN_RELATIVE
    base.FREEZE_RELATIVE = FREEZE_RELATIVE
    base.RESULT_RELATIVE = RESULT_RELATIVE
    base.QUALIFIER_RELATIVE = QUALIFIER_RELATIVE
    base.TEST_RELATIVE = TEST_RELATIVE
    base.ATTEMPT_ROOT_RELATIVE = ATTEMPT_ROOT_RELATIVE
    base.SOURCE_ROOT_RELATIVE = SOURCE_ROOT_RELATIVE
    base.MANIFEST_BINDINGS = MANIFEST_BINDINGS
    base.SOURCE_BINDINGS = SOURCE_BINDINGS


_activate()

BrightQualificationError = base.BrightQualificationError
OneShotRefusal = base.OneShotRefusal
canonical_json = base.canonical_json
file_sha256 = base.file_sha256
qualify_decoded_rows = base.qualify_decoded_rows


def _verify_base_binding(project_root: Path) -> None:
    path = project_root / BASE_RELATIVE
    if not path.is_file() or path.is_symlink() or file_sha256(path) != BASE_SHA256:
        raise BrightQualificationError("base qualifier implementation drifted")


def run_formal(project_root: Path) -> dict[str, Any]:
    _activate()
    root = project_root.resolve(strict=True)
    _verify_base_binding(root)
    return base.run_formal(root)


def main(argv: Iterable[str] | None = None) -> int:
    _activate()
    arguments = base.build_parser().parse_args(argv)
    receipt = run_formal(arguments.project_root)
    print(
        canonical_json(
            {
                "qualification_sha256": receipt["qualification_sha256"],
                "schema": SCHEMA,
                "status": receipt["status"],
            }
        ).decode("ascii")
    )
    return 0 if receipt["status"] == "qualified_source_capacity_no_selection" else 2


if __name__ == "__main__":
    raise SystemExit(main())
