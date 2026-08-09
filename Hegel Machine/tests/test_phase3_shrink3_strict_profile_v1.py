from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from hegel_machine.phase3_shrink3_golden_vectors_v1 import (
    STRICT_GOLDEN_VECTORS_V1,
    strict_golden_manifest_root_v1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = (
    PROJECT_ROOT / "src/hegel_machine/phase3_shrink3_capacity_entrypoint_v1.py"
)


def _direct_replay(mode: str) -> dict[str, object]:
    probe = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(ENTRYPOINT), mode],
        cwd=PROJECT_ROOT,
        env={"PATH": os.environ.get("PATH", "")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    return json.loads(probe.stdout)


def _direct_one(mode: str, payload: str) -> dict[str, object]:
    entrypoint = PROJECT_ROOT / "src/hegel_machine/phase3_shrink3_strict_entrypoint_v1.py"
    probe = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(entrypoint), mode, payload],
        cwd=PROJECT_ROOT,
        env={"PATH": os.environ.get("PATH", "")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    return json.loads(probe.stdout)


def test_sealed_strict_manifest_is_ordered_unique_and_exact() -> None:
    assert len(STRICT_GOLDEN_VECTORS_V1) == 36
    ids = tuple(vector.vector_id for vector in STRICT_GOLDEN_VECTORS_V1)
    assert len(set(ids)) == 36
    assert ids == (
        "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08",
        "A01", "A02", "A03", "A04",
        "P01", "P02", "P03", "P04", "P05", "P06",
        "F01", "F02", "F03",
        "Q01", "Q02", "Q03", "Q04", "Q05", "Q06",
        "H01", "H02", "H03", "H04", "H05", "H06",
        "R01", "R02", "R03",
    )
    assert strict_golden_manifest_root_v1() == (
        "sha256:e091e08f33be8bbfa579b6d333f618326b4ed2ebae6d2830d3adc0df7a6333b5"
    )


def test_direct_python_golden_replay_binds_manifest_and_all_outcomes() -> None:
    report = _direct_replay("--golden-replay")
    assert report["vector_count"] == report["passed_count"] == 36
    assert report["surviving_identity_checks"] == 8
    assert report["source_add_rejection_checks"] == 4
    assert report["source_priority_checks"] == 6
    assert report["formal_add_rejection_checks"] == 3
    assert report["formal_priority_checks"] == 6
    assert report["formal_shape_priority_checks"] == 6
    assert report["formal_alias_or_reserved_checks"] == 3
    assert report["golden_vector_manifest_root"] == (
        "sha256:e091e08f33be8bbfa579b6d333f618326b4ed2ebae6d2830d3adc0df7a6333b5"
    )
    assert report["golden_outcome_root"] == (
        "sha256:b37fcb96c78d53f7da3271513e0cae128ab7e2538288b8aa723254a0f98fde74"
    )
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots"] is None
    assert report["target_or_split_modules_loaded"] is False


def test_direct_python_survivor_replay_is_nonterminal() -> None:
    report = _direct_replay("--capacity-replay")
    assert report["accepted_source_count"] == 2_160
    assert report["accepted_unique_count"] == 2_160
    assert report["parent_identity_match_count"] == 2_160
    assert report["accepted_set_commitment"] == (
        "sha256:9045e4ebe6416dcbf699e7972f25468aef45c0f0aec0e58806061b7ce64d790e"
    )
    assert report["subset_status"] == "SURVIVOR_SUBSET_ONLY_NOT_COMPLETE"
    assert report["executed_closure_status"] == "NOT_RUN"
    assert report["complete_closure_enumerated"] is False
    assert report["formal_roots"] is None
    assert report["target_or_split_modules_loaded"] is False


def test_direct_single_vector_endpoint_is_target_free_and_exact() -> None:
    numeric_source = _direct_one(
        "--source-json", '[7,["scalar_const",1],["scalar_const",5]]'
    )
    assert numeric_source["status"] == "REJECTED"
    assert numeric_source["error_code"] == "REJECT_MALFORMED_SOURCE_AST"

    formal_id8 = _direct_one(
        "--formal-cbor-hex", "82018402088300000183000005"
    )
    assert formal_id8["status"] == "REJECTED"
    assert formal_id8["error_code"] == "REJECT_REGISTRY_INDEX_OUT_OF_RANGE"
    assert formal_id8["target_or_split_modules_loaded"] is False
