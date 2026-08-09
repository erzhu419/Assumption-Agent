from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hegel_machine.phase3_m3_bounded_enumerator_v1 import program_mdl_length_q32
from hegel_machine.phase3_m3_bounded_enumerator_shrink4_v1 import (
    DUAL_ENUMERATION_QUALIFIED,
    EnumerationBindingsV1,
    _Shrink4Enumerator,
    diagnostic_report_shrink4_v1,
    enumerate_bounded_closure_shrink4_v1,
)
from hegel_machine.phase3_m3_shrink4_core_v1 import (
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    MAX_TOP_LEVEL_CLAUSES,
    PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID,
    PARENT_DIAGNOSTIC_RESULT_COMMIT,
    SHRINK_STEP_ID,
)
from hegel_machine.phase3_m3_shrink4_diagnostic_profile_v1 import (
    diagnostic_root_hex_v1,
)
from hegel_machine.phase3_shrink4_capacity_v1 import (
    EXPECTED_SHRINK4_SOURCE_COUNT,
    iter_shrink4_capacity_candidate_asts,
)
from hegel_machine.phase3_shrink4_golden_vectors_v1 import (
    STRICT_GOLDEN_VECTORS_V1,
    strict_golden_manifest_root_v1,
)
from hegel_machine.phase3_shrink4_registry_v1 import (
    SHRINK4_STRUCTURAL_LIMITS,
    SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID,
    STRUCTURAL_LIMIT_SEMANTICS_DIAGNOSTIC_ID,
    shrunk_dsl_surface_object,
    structural_limit_semantics_object,
)
from hegel_machine.strict_ast_shrink3_v1 import (
    canonicalize_shrink3_source_ast,
    decode_shrink3_canonical_ast,
)
from hegel_machine.strict_ast_shrink4_v1 import (
    canonicalize_shrink4_source_ast,
    decode_shrink4_canonical_ast,
    read_legacy_parent_program,
)
from hegel_machine.strict_ast_v1 import StrictAstError


_A = ["context_flag", "c0"]
_B = ["context_flag", "c1"]
_C = ["context_flag", "c2"]


def _direct_replay(entrypoint: str, *args: str) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "src/hegel_machine" / entrypoint
    probe = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(path), *args],
        cwd=project_root,
        env={"PATH": os.environ.get("PATH", "")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    return json.loads(probe.stdout)


def _bindings() -> EnumerationBindingsV1:
    return EnumerationBindingsV1(bytes(32), bytes((1,)) * 32, bytes((2,)) * 32)


def test_machine_ids_and_parent_evidence_commit_are_exact() -> None:
    assert DSL_VERSION == "hegel-old-dsl-v1.4.0"
    assert FREEZE_VERSION == "hegel-freeze-p2b-p3-v1.4.0"
    assert HUMAN_AMENDMENT_ID == "hegel-freeze-p2b-p3-v1.4.0-shrink-step4"
    assert SHRINK_STEP_ID == "SHRINK_STEP_4_REDUCE_MAX_TOP_LEVEL_CLAUSES_3_TO_2"
    assert PARENT_DIAGNOSTIC_RESULT_COMMIT == (
        "c286732c140bd9adcfd3eef2b1788b3eac0eb3e9"
    )
    assert PARENT_DIAGNOSTIC_EVIDENCE_RECORD_ID == (
        "phase3_shrink3_dual_complete_enumeration_diagnostic_"
        "3030ad10f2cd4f767a8397597be1ab3ed6cac7cd71975d69f59cc5abec6a4f5a"
    )


def test_registry_freezes_one_normalized_structural_delta_only() -> None:
    semantics = structural_limit_semantics_object()
    assert semantics["sole_changed_field"] == "maximum_top_level_clauses"
    assert semantics["changed_fields"] == {
        "maximum_top_level_clauses": {"parent": 3, "child": 2}
    }
    assert semantics["normalization_before_limit"] is True
    assert semantics["canonical_and3_disposition"] == "REJECT_STRUCTURAL_LIMIT"
    assert SHRINK4_STRUCTURAL_LIMITS.max_top_level_clauses == 2
    assert STRUCTURAL_LIMIT_SEMANTICS_DIAGNOSTIC_ID == (
        "structural_limit_semantics_"
        "81b5772053bb4ee279213de791c50772987f5c4742071abe6b848ed56d6d3a9c"
    )
    assert SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID == (
        "dsl_spec_dc8e681c6a2c8e893992a7826ff383a45cfc2977ebd575a869a694e2f7341841"
    )
    surface = shrunk_dsl_surface_object()
    assert surface["engineering_trigger"]["result_commit"] == PARENT_DIAGNOSTIC_RESULT_COMMIT
    assert surface["execution_state"] == "NOT_RUN"
    assert surface["formal_roots"] is None


def test_source_normalization_precedes_two_clause_limit() -> None:
    and1 = canonicalize_shrink4_source_ast(["top_level_AND", _A])
    atom = canonicalize_shrink4_source_ast(_A)
    assert and1.cbor_bytes == atom.cbor_bytes
    duplicate = canonicalize_shrink4_source_ast(
        ["top_level_AND", _A, _A, _B]
    )
    and2 = canonicalize_shrink4_source_ast(["top_level_AND", _B, _A])
    assert duplicate.cbor_bytes == and2.cbor_bytes
    assert duplicate.hash_id == and2.hash_id
    assert duplicate.metrics.top_level_clause_count == MAX_TOP_LEVEL_CLAUSES

    for source in (
        ["top_level_AND", _A, _B, _C],
        ["top_level_AND", _A, ["top_level_AND", _B, _C]],
    ):
        with pytest.raises(StrictAstError) as raised:
            canonicalize_shrink4_source_ast(source)
        assert raised.value.code == "REJECT_STRUCTURAL_LIMIT"


def test_formal_and3_rejects_after_parent_canonical_acceptance() -> None:
    parent = canonicalize_shrink3_source_ast(["top_level_AND", _A, _B, _C])
    assert decode_shrink3_canonical_ast(parent.cbor_bytes) == parent
    with pytest.raises(StrictAstError) as raised:
        decode_shrink4_canonical_ast(parent.cbor_bytes)
    assert raised.value.code == "REJECT_STRUCTURAL_LIMIT"
    legacy = read_legacy_parent_program(parent.cbor_bytes)
    assert legacy["legacy_program_status"] == "VALID_UNDER_PARENT_DSL_ONLY"
    assert legacy["current_dsl_error_code"] == "REJECT_STRUCTURAL_LIMIT"


def test_inherited_priority_matrix_is_sealed_and_passes_all_22_vectors() -> None:
    report = _direct_replay(
        "phase3_shrink4_capacity_entrypoint_v1.py", "--golden-replay"
    )
    assert len(STRICT_GOLDEN_VECTORS_V1) == 22
    assert report["vector_count"] == report["passed_count"] == 22
    assert strict_golden_manifest_root_v1() == (
        "sha256:f84035e632bf5a655a9ebd636a0cafe7ab1097c45be87d4db944a0012f52aa90"
    )
    assert report["golden_vector_manifest_root"] == strict_golden_manifest_root_v1()
    assert report["golden_outcome_root"] == (
        "sha256:c19341f08ac5f5759c2cdcb3681a37d958de362b81d02c184f7e2413dca18d7c"
    )
    assert report["source_priority_checks"] == 5
    assert report["formal_priority_checks"] == 8
    assert report["source_structural_limit_checks"] == 2
    assert report["formal_structural_limit_checks"] == 1
    assert report["maximum_top_level_clauses"] == 2
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots_generated"] is False
    assert report["formal_roots"] is None
    assert "max_top_level_clauses" not in report
    assert len(report) == 33


def test_full_2160_and2_survivor_replay_preserves_cbor_hash_and_mdl() -> None:
    sources = tuple(iter_shrink4_capacity_candidate_asts())
    assert len(sources) == EXPECTED_SHRINK4_SOURCE_COUNT == 2_160
    parents = tuple(canonicalize_shrink3_source_ast(source) for source in sources)
    children = tuple(canonicalize_shrink4_source_ast(source) for source in sources)
    assert len({program.cbor_bytes for program in children}) == 2_160
    assert all(child.metrics.top_level_clause_count == 2 for child in children)
    assert all(
        parent.cbor_bytes == child.cbor_bytes
        and parent.hash_id == child.hash_id
        and program_mdl_length_q32(parent) == program_mdl_length_q32(child)
        and decode_shrink4_canonical_ast(child.cbor_bytes) == child
        for parent, child in zip(parents, children)
    )

    report = _direct_replay(
        "phase3_shrink4_capacity_entrypoint_v1.py", "--capacity-replay"
    )
    assert report["source_candidate_count"] == 2_160
    assert report["normalized_and2_count"] == 2_160
    assert report["accepted_unique_count"] == 2_160
    assert report["parent_identity_match_count"] == 2_160
    assert report["accepted_set_commitment"] == (
        "sha256:9045e4ebe6416dcbf699e7972f25468aef45c0f0aec0e58806061b7ce64d790e"
    )
    assert report["maximum_top_level_clauses"] == 2
    assert report["subset_status"] == "FULL_AND2_SURVIVOR_SET_ONLY_NOT_COMPLETE"
    assert report["executed_closure_status"] == "NOT_RUN"
    assert report["formal_roots"] is None
    assert report["target_or_split_modules_loaded"] is False
    assert "max_top_level_clauses" not in report
    assert len(report) == 37


def test_generator_constructs_only_and2_and_never_counts_and3_attempts() -> None:
    state = _Shrink4Enumerator(raw_cap=1_000)
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
    assert sum(
        bucket.structural_limit_rejections for bucket in state.buckets.values()
    ) == 0


def test_bounded_prefix_and_report_remain_nonformal() -> None:
    result = enumerate_bounded_closure_shrink4_v1(
        _bindings(), canonical_budget=2_000, raw_application_cap=500_000
    )
    assert result.dsl_version == DSL_VERSION
    assert result.closure_status == "DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED"
    assert result.canonical_program_count == 2_000
    assert result.first_out_of_budget_program_hash is not None
    assert result.authoritative_claim_allowed is False
    assert DUAL_ENUMERATION_QUALIFIED is False
    report = diagnostic_report_shrink4_v1(
        result,
        _bindings(),
        canonical_budget=2_000,
        raw_application_cap=500_000,
    )
    assert report["maximum_top_level_clauses"] == 2
    assert report["and3_raw_operator_application_count"] == 0
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots"] is None


def test_profile_roots_and_isolated_self_check_are_exact() -> None:
    expected = {
        "child_dsl_spec_root": "736c9cf98749d9a9d2d98596d15a5b09329e1d6eb74d4bee172837fdd34e876f",
        "operator_semantics_root": "45fe7c575759b6955eb6b52ad954a9ca6561083dbdb67155f9731e795c6fe050",
        "identifier_registry_root": "1f9b886480ace19440469267abd06f24c65ef61fb9c734c5ef5ff8ae7e981fd3",
        "canonical_ast_schema_root": "f9b02ddad69f04f1f9137501dccfdcefa111d0402570197b68b98c11ebcb4eda",
        "canonical_cbor_profile_root": "b7fd10722f31d780d53b2f490c92491872ffc749b4cb5cdfccc3eebd5f18837f",
    }
    assert diagnostic_root_hex_v1() == expected
    report = _direct_replay(
        "phase3_m3_shrink4_isolated_entrypoint_v1.py",
        "--target-free-self-check",
    )
    assert report["maximum_top_level_clauses"] == 2
    assert report["execution_state"] == "NOT_RUN"
    assert report["formal_roots"] is None
    assert report["target_or_split_modules_loaded"] is False
    assert {key: report[key] for key in expected} == expected


def test_strict_entrypoint_reports_exact_common_maximum_field() -> None:
    report = _direct_replay(
        "phase3_shrink4_strict_entrypoint_v1.py",
        "--source-json",
        json.dumps(["top_level_AND", _A, _B], separators=(",", ":")),
    )
    assert report["status"] == "ACCEPTED"
    assert report["maximum_top_level_clauses"] == 2
    assert "max_top_level_clauses" not in report
    assert report["target_or_split_modules_loaded"] is False
    rejected = _direct_replay(
        "phase3_shrink4_strict_entrypoint_v1.py",
        "--source-json",
        json.dumps(["top_level_AND", _A, _B, _C], separators=(",", ":")),
    )
    assert rejected["status"] == "REJECTED"
    assert rejected["error_code"] == "REJECT_STRUCTURAL_LIMIT"
    assert rejected["maximum_top_level_clauses"] == 2
