from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hegel_machine.phase3_m3_bounded_enumerator_v1 import program_mdl_length_q32
from hegel_machine.phase3_m3_shrink6_core_v1 import (
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    MAXIMUM_AST_DEPTH,
    MAXIMUM_AST_NODE_COUNT,
    MAX_TOP_LEVEL_CLAUSES,
    PARENT_DIAGNOSTIC_ARTIFACT_PATH,
    PARENT_DIAGNOSTIC_ARTIFACT_SHA256,
    PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID,
    PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS,
    PARENT_DIAGNOSTIC_RESULT_COMMIT,
    SHRINK_STEP_ID,
)
from hegel_machine.phase3_m3_shrink6_diagnostic_profile_v1 import (
    STRICT_QUALIFICATION_ARTIFACT_PATH,
    STRICT_QUALIFICATION_ARTIFACT_SHA256,
    STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH,
    STRICT_QUALIFICATION_EVIDENCE_COMMIT,
    STRICT_QUALIFICATION_SOURCE_COMMIT,
    STRICT_QUALIFICATION_STATUS,
    PREFIX_PRESERVATION_EXPECTATION_ID,
    PREFIX_PRESERVATION_EXPECTATION_STATUS,
)
from hegel_machine.phase3_shrink6_golden_vectors_v1 import (
    STRICT_GOLDEN_VECTORS_V1,
    strict_golden_manifest_root_v1,
)
from hegel_machine.phase3_shrink6_registry_v1 import (
    SHRINK6_STRUCTURAL_LIMITS,
    shrunk_dsl_surface_object,
    structural_limit_semantics_object,
)
from hegel_machine.strict_ast_shrink5_v1 import (
    canonicalize_shrink5_source_ast,
    decode_shrink5_canonical_ast,
)
from hegel_machine.strict_ast_shrink6_v1 import (
    canonicalize_shrink6_source_ast,
    decode_shrink6_canonical_ast,
    read_legacy_parent_program,
)
from hegel_machine.strict_ast_v1 import StrictAstError


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEPTH4_A = [
    "sign",
    [
        "absolute",
        [
            "difference",
            ["bit_to_scalar", ["bit_at", 0]],
            ["scalar_const", -1, 1],
        ],
    ],
]
_DEPTH4_NORMALIZES = [
    "sign",
    [
        "absolute",
        [
            "difference",
            ["bit_to_scalar", ["bit_at", 0]],
            ["scalar_const", 0, 1],
        ],
    ],
]


def _direct(entrypoint: str, *args: str) -> dict[str, object]:
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(_PROJECT_ROOT / "src/hegel_machine" / entrypoint),
            *args,
        ],
        cwd=_PROJECT_ROOT,
        env={"PATH": os.environ.get("PATH", "")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    return json.loads(probe.stdout)


def test_machine_ids_parent_evidence_and_not_run_state_are_exact() -> None:
    assert DSL_VERSION == "hegel-old-dsl-v1.6.0"
    assert FREEZE_VERSION == "hegel-freeze-p2b-p3-v1.6.0"
    assert HUMAN_AMENDMENT_ID == "hegel-freeze-p2b-p3-v1.6.0-shrink-step6"
    assert SHRINK_STEP_ID == "SHRINK_STEP_6_REDUCE_MAX_TOTAL_AST_DEPTH_4_TO_3"
    assert (MAXIMUM_AST_DEPTH, MAXIMUM_AST_NODE_COUNT, MAX_TOP_LEVEL_CLAUSES) == (
        3,
        6,
        2,
    )
    assert PARENT_DIAGNOSTIC_RESULT_COMMIT == (
        "5bfe8474ca63abbadb1d3484a51ce3012081dfb3"
    )
    assert PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS == (
        "a3c384b4cb0f95583af6a1eb1c1d256ef6e9128a"
    )
    assert PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID == (
        "phase3_shrink5_dual_complete_enumeration_diagnostic_"
        "f33b86f3fbab70acb7d8e61fa47f59568a0d56c884c4cf75dfef961cc73dd34b"
    )
    assert PARENT_DIAGNOSTIC_ARTIFACT_SHA256 == (
        "99a799e34876754a8f938f8e25f756992d0784b03bae398b1434e57320b80c82"
    )
    payload = (_PROJECT_ROOT.parent / PARENT_DIAGNOSTIC_ARTIFACT_PATH).read_bytes()
    assert sha256(payload).hexdigest() == PARENT_DIAGNOSTIC_ARTIFACT_SHA256
    assert STRICT_QUALIFICATION_STATUS == "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS"
    assert STRICT_QUALIFICATION_SOURCE_COMMIT == (
        "a69bf6d9746e302a07019f122047ac0bc74aa1c1"
    )
    assert STRICT_QUALIFICATION_EVIDENCE_COMMIT == (
        "f9218e28740953c9ac15a2ada70a8616e92c378b"
    )
    assert STRICT_QUALIFICATION_ARTIFACT_PATH == (
        "Hegel Machine/artifacts/phase3_m3_runtime/"
        "phase3_shrink6_sealed_dual_strict_qualification_v1.json"
    )
    strict_payload = (_PROJECT_ROOT.parent / STRICT_QUALIFICATION_ARTIFACT_PATH).read_bytes()
    assert sha256(strict_payload).hexdigest() == STRICT_QUALIFICATION_ARTIFACT_SHA256 == (
        "d5417639c651ea5d8dfbc224c79b0af56f1eb9d8705ee244f19dc9d95e6f2d08"
    )
    strict_report = json.loads(strict_payload)
    assert strict_report["status"] == STRICT_QUALIFICATION_STATUS
    assert strict_report["diagnostic_report_hash"] == (
        STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH
    )
    assert strict_report["repository_binding"]["qualification_basis_commit"] == (
        STRICT_QUALIFICATION_SOURCE_COMMIT
    )
    assert STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH == (
        "sha256:3d2a6f06daa47b34aa56ae0d318cc818ba211859063d7a6b81271bc6bf1f8287"
    )


def test_registry_freezes_only_depth_four_to_three() -> None:
    semantics = structural_limit_semantics_object()
    assert semantics["sole_changed_field"] == "maximum_ast_depth"
    assert semantics["changed_fields"] == {
        "maximum_ast_depth": {"parent": 4, "child": 3}
    }
    assert SHRINK6_STRUCTURAL_LIMITS.max_total_ast_depth == 3
    assert SHRINK6_STRUCTURAL_LIMITS.max_total_node_count == 6
    assert SHRINK6_STRUCTURAL_LIMITS.max_top_level_clauses == 2
    surface = shrunk_dsl_surface_object()
    assert surface["remaining_shrink_order"] == []
    assert surface["execution_state"] == "NOT_RUN"
    assert surface["formal_roots"] is None
    assert surface["formal_state_transition_allowed"] is False


def test_normalization_precedes_depth_limit_and_survivors_keep_identity() -> None:
    normalized = canonicalize_shrink6_source_ast(_DEPTH4_NORMALIZES)
    parent_normalized = canonicalize_shrink5_source_ast(_DEPTH4_NORMALIZES)
    assert normalized.metrics.depth == 3
    assert normalized.cbor_bytes == parent_normalized.cbor_bytes
    assert normalized.hash_id == parent_normalized.hash_id
    assert program_mdl_length_q32(normalized) == program_mdl_length_q32(
        parent_normalized
    )

    parent_only = canonicalize_shrink5_source_ast(_DEPTH4_A)
    assert (parent_only.metrics.depth, parent_only.metrics.node_count) == (4, 6)
    assert decode_shrink5_canonical_ast(parent_only.cbor_bytes) == parent_only
    with pytest.raises(StrictAstError) as source_error:
        canonicalize_shrink6_source_ast(_DEPTH4_A)
    assert source_error.value.code == "REJECT_STRUCTURAL_LIMIT"
    with pytest.raises(StrictAstError) as formal_error:
        decode_shrink6_canonical_ast(parent_only.cbor_bytes)
    assert formal_error.value.code == "REJECT_STRUCTURAL_LIMIT"
    legacy = read_legacy_parent_program(parent_only.cbor_bytes)
    assert legacy["legacy_program_status"] == "VALID_UNDER_PARENT_DSL_ONLY"
    assert legacy["automatic_rewrite_or_migration_performed"] is False


def test_sealed_25_vector_shape_and_exact_roots() -> None:
    assert [vector.vector_id for vector in STRICT_GOLDEN_VECTORS_V1] == [
        "S01", "S02", "S03", "N01", "N02", "L01", "L02", "L03",
        "P01", "P02", "P03", "P04", "P05", "F01", "F02", "F03",
        "F04", "F05", "F06", "F07", "F08", "F09", "F10", "F11",
        "F12",
    ]
    assert strict_golden_manifest_root_v1() == (
        "sha256:2690413926d15db52dbd5a502ebe3fdfb1dc74d5ee3c82b2ed868cd16ab34a42"
    )
    report = _direct("phase3_shrink6_capacity_entrypoint_v1.py", "--golden-replay")
    assert report["vector_count"] == report["passed_count"] == 25
    assert report["golden_outcome_root"] == (
        "sha256:e5fd0885f95669dc6d369d0d3274778425fabb7e8c6286a27237a1b2bc8d3960"
    )
    assert report["surviving_identity_checks"] == 3
    assert report["source_normalization_before_limit_checks"] == 2
    assert report["source_depth_limit_checks"] == 3
    assert report["source_priority_checks"] == 5
    assert report["formal_surviving_identity_checks"] == 1
    assert report["formal_depth_limit_checks"] == 3
    assert report["formal_priority_checks"] == 8
    assert report["maximum_ast_depth"] == 3
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots"] is None


def test_direct_strict_and_isolated_self_check_are_target_free() -> None:
    accepted = _direct(
        "phase3_shrink6_strict_entrypoint_v1.py",
        "--source-json",
        '["scalar_const",1]',
    )
    assert accepted["status"] == "ACCEPTED"
    assert accepted["maximum_ast_depth"] == 3
    rejected = _direct(
        "phase3_shrink6_strict_entrypoint_v1.py",
        "--source-json",
        json.dumps(_DEPTH4_A, separators=(",", ":")),
    )
    assert rejected["status"] == "REJECTED"
    assert rejected["error_code"] == "REJECT_STRUCTURAL_LIMIT"
    self_check = _direct(
        "phase3_m3_shrink6_isolated_entrypoint_v1.py",
        "--target-free-self-check",
    )
    assert self_check["maximum_ast_depth"] == 3
    assert self_check["strict_qualification_status"] == (
        "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS"
    )
    assert self_check["strict_qualification_source_commit"] == (
        "a69bf6d9746e302a07019f122047ac0bc74aa1c1"
    )
    assert self_check["prefix_preservation_expectation_id"] == (
        PREFIX_PRESERVATION_EXPECTATION_ID
    )
    assert self_check["prefix_preservation_expectation_status"] == (
        PREFIX_PRESERVATION_EXPECTATION_STATUS
    )
    assert self_check["formal_bucket_count"] == 120
    assert self_check["preregistered_shrink_order_total_steps"] == 6
    assert self_check["preregistered_shrink_order_consumed_through_step"] == 6
    assert self_check["next_preregistered_shrink_step_or_null"] is None
    assert self_check["budget_change_authorized"] is False
    assert self_check["additional_shrink_authorized"] is False
    assert self_check["new_dsl_version_authorized"] is False
    assert self_check["execution_state"] == "NOT_RUN"
    assert self_check["formal_roots"] is None
    assert self_check["target_or_split_modules_loaded"] is False
