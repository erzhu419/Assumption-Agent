from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hegel_machine.phase3_m3_bounded_enumerator_v1 import program_mdl_length_q32
from hegel_machine.phase3_m3_bounded_enumerator_shrink5_v1 import (
    DUAL_ENUMERATION_QUALIFIED,
    EnumerationBindingsV1,
    _Shrink5Enumerator,
    diagnostic_report_shrink5_v1,
    enumerate_bounded_closure_shrink5_v1,
)
from hegel_machine.phase3_m3_shrink5_core_v1 import (
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    MAXIMUM_AST_NODE_COUNT,
    MAX_TOP_LEVEL_CLAUSES,
    PARENT_DIAGNOSTIC_ARTIFACT_PATH,
    PARENT_DIAGNOSTIC_ARTIFACT_SHA256,
    PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID,
    PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS,
    PARENT_DIAGNOSTIC_RESULT_COMMIT,
    SHRINK_STEP_ID,
)
from hegel_machine.phase3_m3_shrink5_diagnostic_profile_v1 import (
    PARENT_DIAGNOSTIC_ARTIFACT_SHA256 as PROFILE_PARENT_ARTIFACT_SHA256,
    PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID as PROFILE_PARENT_RECORD_ID,
    PARENT_DIAGNOSTIC_RESULT_COMMIT as PROFILE_PARENT_RESULT_COMMIT,
    diagnostic_root_hex_v1,
)
from hegel_machine.phase3_shrink5_capacity_v1 import (
    EXPECTED_SHRINK5_BOUNDARY_SOURCE_COUNT,
    EXPECTED_SHRINK5_SOURCE_COUNT,
    EXPECTED_SHRINK5_SURVIVOR_SOURCE_COUNT,
    iter_shrink5_capacity_candidate_asts,
)
from hegel_machine.phase3_shrink5_golden_vectors_v1 import (
    STRICT_GOLDEN_VECTORS_V1,
    strict_golden_manifest_root_v1,
)
from hegel_machine.phase3_shrink5_registry_v1 import (
    SHRINK5_STRUCTURAL_LIMITS,
    SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID,
    STRUCTURAL_LIMIT_SEMANTICS_DIAGNOSTIC_ID,
    shrunk_dsl_surface_object,
    structural_limit_semantics_object,
)
from hegel_machine.strict_ast_shrink4_v1 import (
    canonicalize_shrink4_source_ast,
    decode_shrink4_canonical_ast,
)
from hegel_machine.strict_ast_shrink5_v1 import (
    canonicalize_shrink5_source_ast,
    decode_shrink5_canonical_ast,
    read_legacy_parent_program,
)
from hegel_machine.strict_ast_v1 import StrictAstError


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_A = ["context_flag", "c0"]
_AGGREGATE = [
    "aggregate",
    "sum_v1",
    "scope_all_observed_v1",
    "q0",
    [],
]
_NODE6_BOOL = [
    "less_equal",
    ["scalar_const", 1],
    ["absolute", _AGGREGATE],
]
_NODE6 = ["top_level_AND", _A, _NODE6_BOOL]
_NODE7 = [
    "top_level_AND",
    ["less_equal", ["scalar_const", 1], ["scalar_const", 3]],
    ["less_equal", ["scalar_const", 5], _AGGREGATE],
]


def _direct_replay(entrypoint: str, *args: str) -> dict[str, object]:
    path = _PROJECT_ROOT / "src/hegel_machine" / entrypoint
    probe = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(path), *args],
        cwd=_PROJECT_ROOT,
        env={"PATH": os.environ.get("PATH", "")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    return json.loads(probe.stdout)


def _bindings() -> EnumerationBindingsV1:
    return EnumerationBindingsV1(bytes(32), bytes((1,)) * 32, bytes((2,)) * 32)


def test_machine_ids_parent_evidence_and_artifact_are_exact() -> None:
    assert DSL_VERSION == "hegel-old-dsl-v1.5.0"
    assert FREEZE_VERSION == "hegel-freeze-p2b-p3-v1.5.0"
    assert HUMAN_AMENDMENT_ID == "hegel-freeze-p2b-p3-v1.5.0-shrink-step5"
    assert SHRINK_STEP_ID == "SHRINK_STEP_5_REDUCE_MAX_TOTAL_NODE_COUNT_7_TO_6"
    assert MAXIMUM_AST_NODE_COUNT == 6
    assert MAX_TOP_LEVEL_CLAUSES == 2
    assert PARENT_DIAGNOSTIC_RESULT_COMMIT == (
        "1bbdae8f3131625621c0bc1cfdfe5d7da6035e13"
    )
    assert PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS == (
        "103eb6ad2d8500024580193b895809784d894609"
    )
    assert PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID == (
        "phase3_shrink4_dual_complete_enumeration_diagnostic_"
        "5693b38315689969a1a525b75bec2917f95af1aa54951267797a0319afc60521"
    )
    assert PARENT_DIAGNOSTIC_ARTIFACT_SHA256 == (
        "2d653f667d8d43e0e8e68c54d6f0a939aab57bf6ba3add9b334809ca17745058"
    )
    assert PROFILE_PARENT_RESULT_COMMIT == PARENT_DIAGNOSTIC_RESULT_COMMIT
    assert PROFILE_PARENT_RECORD_ID == PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID
    assert PROFILE_PARENT_ARTIFACT_SHA256 == PARENT_DIAGNOSTIC_ARTIFACT_SHA256

    artifact_path = _PROJECT_ROOT.parent / PARENT_DIAGNOSTIC_ARTIFACT_PATH
    payload = artifact_path.read_bytes()
    assert sha256(payload).hexdigest() == PARENT_DIAGNOSTIC_ARTIFACT_SHA256
    artifact = json.loads(payload)
    assert artifact["evidence_record_id"] == PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID
    assert artifact["status"] == "DUAL_DSL_TOO_LARGE_HOST_REPLAY_PASS"
    assert artifact["claim_level"] == "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
    assert (
        artifact["repository_binding"]["source_q_commit"]
        == PARENT_DIAGNOSTIC_IMPLEMENTATION_BASIS
    )
    assert artifact["routing"] == {
        "authority": "ENGINEERING_ONLY",
        "formal_status_promotion_allowed": False,
        "from_max_total_node_count": 7,
        "maximum_top_level_clauses_remains": 2,
        "only_open_route": True,
        "operation": "reduce max_total_node_count from 7 to 6",
        "preregistered_shrink_order_step": 5,
        "to_max_total_node_count": 6,
    }


def test_registry_freezes_only_node_count_seven_to_six() -> None:
    semantics = structural_limit_semantics_object()
    assert semantics["sole_changed_field"] == "maximum_ast_node_count"
    assert semantics["changed_fields"] == {
        "maximum_ast_node_count": {"parent": 7, "child": 6}
    }
    assert semantics["maximum_top_level_clauses"] == 2
    assert semantics["canonical_seven_node_disposition"] == "REJECT_STRUCTURAL_LIMIT"
    assert SHRINK5_STRUCTURAL_LIMITS.max_total_node_count == 6
    assert SHRINK5_STRUCTURAL_LIMITS.max_top_level_clauses == 2
    assert STRUCTURAL_LIMIT_SEMANTICS_DIAGNOSTIC_ID == (
        "structural_limit_semantics_"
        "f844b7fd8a631f28dc6a2ef640ee79647a4dfe8f2dc00d12cd747c920dd82f93"
    )
    assert SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID == (
        "dsl_spec_530515cfbbc63506c46682c543cc325f84e5d441af159f23be53f0908b14a80a"
    )
    surface = shrunk_dsl_surface_object()
    assert surface["pre_registered_delta_only"] == (
        "reduce max_total_node_count from 7 to 6"
    )
    assert surface["engineering_trigger"]["artifact_sha256"] == (
        PARENT_DIAGNOSTIC_ARTIFACT_SHA256
    )
    assert surface["execution_state"] == "NOT_RUN"
    assert surface["formal_roots"] is None


def test_normalize_before_count_accepts_node6_and_rejects_node7() -> None:
    node6 = canonicalize_shrink5_source_ast(_NODE6)
    assert node6.metrics.node_count == 6
    assert node6.metrics.top_level_clause_count == 2
    parent6 = canonicalize_shrink4_source_ast(_NODE6)
    assert node6.cbor_bytes == parent6.cbor_bytes
    assert node6.hash_id == parent6.hash_id

    zero_difference = canonicalize_shrink5_source_ast(
        [
            "top_level_AND",
            _A,
            [
                "less_equal",
                ["difference", ["scalar_const", 1], ["scalar_const", 3]],
                ["absolute", _AGGREGATE],
            ],
        ]
    )
    nested_duplicate = canonicalize_shrink5_source_ast(
        ["top_level_AND", _A, ["top_level_AND", _A, _NODE6_BOOL]]
    )
    assert zero_difference.cbor_bytes == node6.cbor_bytes
    assert nested_duplicate.cbor_bytes == node6.cbor_bytes

    parent7 = canonicalize_shrink4_source_ast(_NODE7)
    assert parent7.metrics.node_count == 7
    assert decode_shrink4_canonical_ast(parent7.cbor_bytes) == parent7
    with pytest.raises(StrictAstError, match="REJECT_STRUCTURAL_LIMIT") as source_error:
        canonicalize_shrink5_source_ast(_NODE7)
    assert source_error.value.code == "REJECT_STRUCTURAL_LIMIT"
    with pytest.raises(StrictAstError, match="REJECT_STRUCTURAL_LIMIT") as formal_error:
        decode_shrink5_canonical_ast(parent7.cbor_bytes)
    assert formal_error.value.code == "REJECT_STRUCTURAL_LIMIT"
    legacy = read_legacy_parent_program(parent7.cbor_bytes)
    assert legacy["legacy_program_status"] == "VALID_UNDER_PARENT_DSL_ONLY"
    assert legacy["current_dsl_error_code"] == "REJECT_STRUCTURAL_LIMIT"


def test_all_22_golden_vectors_and_exact_wire_roots_pass() -> None:
    report = _direct_replay(
        "phase3_shrink5_capacity_entrypoint_v1.py", "--golden-replay"
    )
    assert len(STRICT_GOLDEN_VECTORS_V1) == 22
    assert report["vector_count"] == report["passed_count"] == 22
    assert strict_golden_manifest_root_v1() == (
        "sha256:156f7e20407437bb753b097a87932f469701d1de6d1d577b0fa1b7a98f47e52e"
    )
    assert report["golden_vector_manifest_root"] == strict_golden_manifest_root_v1()
    assert report["golden_outcome_root"] == (
        "sha256:8f82178c0f33d5295601d2e112b0b6e25ef18d73e5fc35d8d601024c1f0ddf94"
    )
    assert report["source_priority_checks"] == 5
    assert report["formal_priority_checks"] == 8
    assert report["maximum_ast_node_count"] == 6
    assert report["maximum_top_level_clauses"] == 2
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots"] is None
    assert len(report["loaded_hegel_modules"]) == 21
    assert len(report) == 34


def test_capacity_replay_binds_175_survivors_and_both_2160_boundaries() -> None:
    survivor_sources = tuple(iter_shrink5_capacity_candidate_asts())
    assert len(survivor_sources) == EXPECTED_SHRINK5_SURVIVOR_SOURCE_COUNT == 175
    assert EXPECTED_SHRINK5_BOUNDARY_SOURCE_COUNT == 2_160
    assert EXPECTED_SHRINK5_SOURCE_COUNT == 2_335
    parents = tuple(canonicalize_shrink4_source_ast(source) for source in survivor_sources)
    children = tuple(canonicalize_shrink5_source_ast(source) for source in survivor_sources)
    assert len({program.cbor_bytes for program in children}) == 175
    assert all(
        parent.cbor_bytes == child.cbor_bytes
        and parent.hash_id == child.hash_id
        and program_mdl_length_q32(parent) == program_mdl_length_q32(child)
        and decode_shrink5_canonical_ast(child.cbor_bytes) == child
        for parent, child in zip(parents, children)
    )

    report = _direct_replay(
        "phase3_shrink5_capacity_entrypoint_v1.py", "--capacity-replay"
    )
    assert report["survivor_source_candidate_count"] == 175
    assert report["survivor_accepted_count"] == 175
    assert report["survivor_unique_count"] == 175
    assert report["survivor_parent_identity_match_count"] == 175
    assert report["survivor_rejected_count"] == 0
    assert report["survivor_accepted_set_commitment"] == (
        "sha256:f5ab7f079ad943d65a74881eb59c7bb46385e1c437ca8ab036bb071dfa3874ac"
    )
    assert report["parent_only_source_candidate_count"] == 2_160
    assert report["parent_only_parent_accepted_count"] == 2_160
    assert report["parent_only_source_child_rejected_count"] == 2_160
    assert report["parent_only_formal_child_rejected_count"] == 2_160
    expected_errors = {"REJECT_STRUCTURAL_LIMIT": 2_160}
    assert report["parent_only_source_child_rejection_counts"] == expected_errors
    assert report["parent_only_formal_child_rejection_counts"] == expected_errors
    assert report["parent_only_set_commitment"] == (
        "sha256:7e0e8780149f03ce85723408f7e3eff2cd684e8938896125cf8e34be9ac70b5e"
    )
    assert report["parent_only_source_rejection_outcome_commitment"] == (
        "sha256:8617b56bdfa347f11f2c68b6a41f0992652f1e23e6d651017b17eb50169a9f39"
    )
    assert report["parent_only_formal_rejection_outcome_commitment"] == (
        "sha256:9a6b489ed90960008aebbecdbcf0bc5cf1595b7a8206d179bbe898540dabf617"
    )
    assert report["subset_status"] == (
        "FULL_175_SURVIVOR_AND_2160_PARENT_NODE7_BOUNDARY_SETS_ONLY_NOT_COMPLETE"
    )
    assert report["executed_closure_status"] == "NOT_RUN"
    assert report["formal_roots"] is None
    assert len(report["loaded_hegel_modules"]) == 21
    assert len(report) == 46


def test_generator_has_exact_six_node_bucket_lattice_and_and2_only() -> None:
    state = _Shrink5Enumerator(raw_cap=1_000)
    assert len(state.buckets) == 150
    assert max(key[2] for key in state.buckets) == 6
    state.leaves(1)
    before = state.raw_count
    state.conjunctions(depth=1, nodes=3)
    generated = state.groups[("Bool", 1, 3)]
    assert state.raw_count - before == len(generated) == 15
    assert all(
        program.expr.tag == 4
        and len(program.expr.children) == 2
        and program.ast.metrics.top_level_clause_count == 2
        for program in generated
    )


def test_bounded_prefix_and_report_remain_nonformal_with_150_buckets() -> None:
    result = enumerate_bounded_closure_shrink5_v1(
        _bindings(), canonical_budget=2_000, raw_application_cap=500_000
    )
    assert result.dsl_version == DSL_VERSION
    assert result.closure_status == "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED"
    assert result.canonical_program_count == 2_000
    assert len(result.bucket_accounting_records) == 150
    assert result.first_out_of_budget_program_hash is not None
    assert result.authoritative_claim_allowed is False
    assert DUAL_ENUMERATION_QUALIFIED is False
    report = diagnostic_report_shrink5_v1(
        result,
        _bindings(),
        canonical_budget=2_000,
        raw_application_cap=500_000,
    )
    assert report["maximum_ast_node_count"] == 6
    assert report["maximum_top_level_clauses"] == 2
    assert report["formal_bucket_count"] == 150
    assert report["and3_raw_operator_application_count"] == 0
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots"] is None


def test_profile_roots_and_isolated_self_check_are_exact() -> None:
    expected = {
        "child_dsl_spec_root": "3340b3278caf562b560cc30cd14d3cd5f1d628e222b43d29d9d1e41b379f5675",
        "operator_semantics_root": "5d2700884ae7125b9712a2bd06aa929feaf2fad1d4bfcd4fa5953c157a720ee1",
        "identifier_registry_root": "1b0c141126b278778009d3ebbbf49f5de231ad0166a88a8a9caf367b35bff8ef",
        "canonical_ast_schema_root": "828fdcc9f16ebd590702ff4297cac6f6ffa19b01299ea7a93753a4fced0961c5",
        "canonical_cbor_profile_root": "0ccbd740c0b1f6a39fb8151ea56e114561093ee4fccb228bf83a9294e0bae783",
    }
    assert diagnostic_root_hex_v1() == expected
    report = _direct_replay(
        "phase3_m3_shrink5_isolated_entrypoint_v1.py",
        "--target-free-self-check",
    )
    assert report["maximum_ast_node_count"] == 6
    assert report["maximum_top_level_clauses"] == 2
    assert report["sealed_dual_strict_outcome_replay_status"] == "NOT_RUN"
    assert report["parent_diagnostic_claim_level"] == (
        "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
    )
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots"] is None
    assert len(report["loaded_hegel_modules"]) == 20
    assert {key: report[key] for key in expected} == expected
    assert not any(key.startswith("strict_qualification_") for key in report)


def test_strict_accepted_and_rejected_reports_bind_both_maximums() -> None:
    accepted = _direct_replay(
        "phase3_shrink5_strict_entrypoint_v1.py",
        "--source-json",
        json.dumps(_NODE6, separators=(",", ":")),
    )
    assert accepted["status"] == "ACCEPTED"
    assert accepted["node_count"] == 6
    assert accepted["maximum_ast_node_count"] == 6
    assert accepted["maximum_top_level_clauses"] == 2
    assert len(accepted["loaded_hegel_modules"]) == 13

    rejected = _direct_replay(
        "phase3_shrink5_strict_entrypoint_v1.py",
        "--source-json",
        json.dumps(_NODE7, separators=(",", ":")),
    )
    assert rejected["status"] == "REJECTED"
    assert rejected["error_code"] == "REJECT_STRUCTURAL_LIMIT"
    assert rejected["maximum_ast_node_count"] == 6
    assert rejected["maximum_top_level_clauses"] == 2
    assert len(rejected["loaded_hegel_modules"]) == 13
    assert "max_total_node_count" not in accepted
    assert "max_total_node_count" not in rejected
