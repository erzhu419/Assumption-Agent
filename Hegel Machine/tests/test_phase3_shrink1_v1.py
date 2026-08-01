from __future__ import annotations

import json
from pathlib import Path

import pytest

from hegel_machine.phase3_shrink1_capacity_v1 import (
    EXPECTED_SHRINK1_SOURCE_COUNT,
    iter_shrink1_capacity_candidate_asts,
)
from hegel_machine.phase3_shrink1_publication_v1 import (
    M3_REQUIRED_GATES,
    binding_manifests_report,
    formal_root_state,
    m3_entry_contract_report,
    shrink1_publication_report,
    shrink_transition_report,
)
from hegel_machine.phase3_shrink1_registry_v1 import (
    ACTIVE_AGGREGATE_IDS,
    AGGREGATE_REGISTRY_DIAGNOSTIC_ID,
    DSL_VERSION,
    FREEZE_VERSION,
    NEXT_ALLOCATABLE_AGGREGATE_ID,
    REGISTRY_WIDTH,
    TOMBSTONED_AGGREGATE_IDS,
)
from hegel_machine.phase3_shrink1_replay_v1 import (
    DEFAULT_RUST_BINARY,
    PYTHON_CAPACITY_SOURCES,
    dual_shrink1_capacity_replay_report,
    dual_shrink1_strict_gate_report,
    python_shrink1_capacity_replay,
    python_shrink1_vector_report,
)
from hegel_machine.strict_ast_shrink1_v1 import (
    PROGRAM_SEMANTIC_IDENTITY_DOMAIN,
    ProgramAdmissionIdentityV1,
    ProgramSemanticIdentityV1,
    aggregate_id_is_active,
    canonicalize_shrink1_source_ast,
    decode_shrink1_canonical_ast,
    read_legacy_parent_program,
)
from hegel_machine.strict_ast_v1 import (
    StrictAstError,
    canonicalize_source_ast,
    decode_canonical_ast,
)
from hegel_machine.strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash_id,
)


ROOT = Path(__file__).resolve().parents[1]


def test_sparse_registry_freeze_is_exact_and_never_reuses_tombstones() -> None:
    assert DSL_VERSION == "hegel-old-dsl-v1.1.0"
    assert FREEZE_VERSION == "hegel-freeze-p2b-p3-v1.1.0"
    assert REGISTRY_WIDTH == 6
    assert ACTIVE_AGGREGATE_IDS == (0, 1, 5)
    assert TOMBSTONED_AGGREGATE_IDS == (2, 3, 4)
    assert NEXT_ALLOCATABLE_AGGREGATE_ID == 6
    assert AGGREGATE_REGISTRY_DIAGNOSTIC_ID.startswith("aggregate_registry_")
    assert aggregate_id_is_active(5) is True
    with pytest.raises(StrictAstError, match="REJECT_REMOVED_AGGREGATE_MAP"):
        aggregate_id_is_active(2)
    with pytest.raises(StrictAstError, match="REJECT_REGISTRY_INDEX_OUT_OF_RANGE"):
        aggregate_id_is_active(6)


@pytest.mark.parametrize("map_value", ["mean_v1", "min_v1", "max_v1", 2, 3, 4])
def test_removed_source_maps_use_exact_tombstone_error(map_value: object) -> None:
    with pytest.raises(StrictAstError) as raised:
        canonicalize_shrink1_source_ast(
            ["aggregate", map_value, "scope_all_observed_v1", "q0", []]
        )
    assert raised.value.code == "REJECT_REMOVED_AGGREGATE_MAP"


def test_future_source_map_is_out_of_range_not_tombstoned() -> None:
    with pytest.raises(StrictAstError) as raised:
        canonicalize_shrink1_source_ast(
            ["aggregate", 6, "scope_all_observed_v1", "q0", []]
        )
    assert raised.value.code == "REJECT_REGISTRY_INDEX_OUT_OF_RANGE"


def test_nested_and_atom_list_checks_the_first_real_ast_child() -> None:
    source = [
        "top_level_AND",
        [
            ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []],
            ["equal_exact", ["scalar_const", 0], ["scalar_const", 0]],
        ],
    ]
    with pytest.raises(StrictAstError) as raised:
        canonicalize_shrink1_source_ast(source)
    assert raised.value.code == "REJECT_REMOVED_AGGREGATE_MAP"


@pytest.mark.parametrize(
    ("source", "expected_code"),
    [
        (
            [
                "scalar_const",
                0,
                ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []],
            ],
            "REJECT_MALFORMED_SOURCE_AST",
        ),
        (
            [
                "unknown_outer",
                ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []],
            ],
            "REJECT_UNKNOWN_EXPRESSION",
        ),
        (
            [
                "top_level_AND",
                [[], ["aggregate", 2, "scope_all_observed_v1", "q0", []]],
            ],
            "REJECT_MALFORMED_SOURCE_AST",
        ),
    ],
)
def test_source_tombstone_precheck_does_not_scan_non_ast_payloads(
    source: object, expected_code: str
) -> None:
    with pytest.raises(StrictAstError) as parent_raised:
        canonicalize_source_ast(source)
    with pytest.raises(StrictAstError) as child_raised:
        canonicalize_shrink1_source_ast(source)
    assert parent_raised.value.code == expected_code
    assert child_raised.value.code == expected_code


@pytest.mark.parametrize("map_name", ["sum_v1", "count_nonzero_v1", "signed_balance_v1"])
def test_surviving_source_bytes_and_hashes_are_stable(map_name: str) -> None:
    source = ["aggregate", map_name, "scope_all_observed_v1", "q0", []]
    parent = canonicalize_source_ast(source)
    child = canonicalize_shrink1_source_ast(source)
    assert child.cbor_bytes == parent.cbor_bytes
    assert child.hash_id == parent.hash_id


@pytest.mark.parametrize("map_name", ["mean_v1", "min_v1", "max_v1"])
def test_removed_formal_bytes_remain_generic_cbor_and_parent_history(map_name: str) -> None:
    parent = canonicalize_source_ast(
        ["aggregate", map_name, "scope_all_observed_v1", "q0", []]
    )
    assert canonical_cbor_decode(parent.cbor_bytes)
    assert decode_canonical_ast(parent.cbor_bytes).hash_id == parent.hash_id
    with pytest.raises(StrictAstError) as raised:
        decode_shrink1_canonical_ast(parent.cbor_bytes)
    assert raised.value.code == "REJECT_REMOVED_AGGREGATE_MAP"
    legacy = read_legacy_parent_program(parent.cbor_bytes)
    assert legacy["legacy_program_status"] == "VALID_UNDER_PARENT_DSL_ONLY"
    assert legacy["admitted_under_current_dsl"] is False
    assert legacy["automatic_map_migration_performed"] is False


@pytest.mark.parametrize(
    "formal_value",
    [
        (2, (0, 3, 2, 0, 0, ())),
        (0, 3, 2, 0, 0, ()),
    ],
)
def test_formal_tombstone_precheck_requires_the_v1_ast_envelope(
    formal_value: object,
) -> None:
    payload = canonical_cbor_encode(formal_value)
    with pytest.raises(StrictAstError) as parent_raised:
        decode_canonical_ast(payload)
    with pytest.raises(StrictAstError) as child_raised:
        decode_shrink1_canonical_ast(payload)
    assert child_raised.value.code == parent_raised.value.code
    assert child_raised.value.code != "REJECT_REMOVED_AGGREGATE_MAP"


def test_formal_real_child_tombstone_precedes_noncanonical_and_arity() -> None:
    payload = canonical_cbor_encode((1, (4, ((0, 3, 2, 0, 0, ()),))))
    with pytest.raises(StrictAstError) as parent_raised:
        decode_canonical_ast(payload)
    assert parent_raised.value.code == "REJECT_NONCANONICAL_AST"
    with pytest.raises(StrictAstError) as child_raised:
        decode_shrink1_canonical_ast(payload)
    assert child_raised.value.code == "REJECT_REMOVED_AGGREGATE_MAP"


def test_cross_dsl_semantic_identity_requires_version_binding_roots() -> None:
    program = canonicalize_shrink1_source_ast(
        ["aggregate", "sum_v1", "scope_all_observed_v1", "q0", []]
    )
    a = "sha256:" + "11" * 32
    b = "sha256:" + "22" * 32
    c = "sha256:" + "33" * 32
    admission = ProgramAdmissionIdentityV1(program.hash_id, a, b)
    semantic = ProgramSemanticIdentityV1(program.hash_id, a, c, b)
    expected_admission_value = (
        bytes.fromhex(program.hash_id.removeprefix("sha256:")),
        bytes.fromhex(a.removeprefix("sha256:")),
        bytes.fromhex(b.removeprefix("sha256:")),
    )
    assert admission.canonical_value == expected_admission_value
    expected_semantic_value = (
        expected_admission_value[0],
        expected_admission_value[1],
        bytes.fromhex(c.removeprefix("sha256:")),
        expected_admission_value[2],
    )
    assert semantic.content_id == content_hash_id(
        PROGRAM_SEMANTIC_IDENTITY_DOMAIN,
        expected_semantic_value,
    )
    with pytest.raises(ValueError, match="sha256 root"):
        ProgramAdmissionIdentityV1(program.hash_id, "diagnostic_id", b).canonical_value


def test_pure_subset_generator_has_exact_source_count_and_no_report_cycle() -> None:
    assert sum(1 for _ in iter_shrink1_capacity_candidate_asts()) == 25_872
    relative = {path.relative_to(ROOT).as_posix() for path in PYTHON_CAPACITY_SOURCES}
    assert "src/hegel_machine/phase3_shrink1_capacity_v1.py" in relative
    generator = (
        ROOT / "src" / "hegel_machine" / "phase3_shrink1_capacity_v1.py"
    ).read_text(encoding="utf-8")
    assert "phase3_shrink1_dual_strict_gate_" not in generator
    assert "phase3_shrink1_dual_capacity_replay_" not in generator


def test_python_vector_and_subset_replay_match_frozen_baseline() -> None:
    vectors = python_shrink1_vector_report()
    assert vectors["vector_count"] == 23
    assert vectors["passed_count"] == 23
    assert vectors["all_expectations_match"] is True
    capacity = python_shrink1_capacity_replay()
    assert capacity["source_candidate_count"] == EXPECTED_SHRINK1_SOURCE_COUNT
    assert capacity["accepted_unique_count"] == EXPECTED_SHRINK1_SOURCE_COUNT
    assert capacity["accepted_set_commitment"] == (
        "sha256:653fcb9428684cfed11c3f2345ac95ed98ded6e31564c9eeabf97c57ee71a7e9"
    )
    assert capacity["first_out_of_budget_ast_hash"] is None
    assert capacity["executed_closure_status"] == "NOT_RUN"
    assert capacity["complete_closure_enumerated"] is False


def test_publication_is_fail_closed_on_missing_external_seed_evidence() -> None:
    bindings = binding_manifests_report()
    assert bindings["m3_commitment_gate_satisfied"] is False
    assert bindings["old_validation_disposition"] == (
        "HISTORICAL_PRECOMMITMENT_ONLY_SEALED"
    )
    assert bindings["old_validation_still_sealed"] is True
    assert bindings["split_binding_manifest"]["split_seed_commitment"] is None
    assert bindings["custodian_binding_manifest"]["reuse_authorized"] is False
    roots = formal_root_state()
    assert roots["formal_roots"] is None
    assert all(value is None for value in roots["binding_roots"].values())
    assert all(value is None for value in roots["run_output_roots"].values())


@pytest.mark.skipif(
    not DEFAULT_RUST_BINARY.is_file(),
    reason="compiled Rust release replay is an execution artifact, not checked in",
)
def test_dual_shrink1_artifacts_and_m3_gate_remain_bounded() -> None:
    gate = dual_shrink1_strict_gate_report()
    capacity = dual_shrink1_capacity_replay_report()
    assert gate["status"] == "VERIFIED"
    assert gate["cross_language_vector_identity_equal"] is True
    assert capacity["status"] == "VERIFIED_WITHIN_BUDGET"
    assert capacity["dual_replay_equal"] is True
    assert capacity["first_out_of_budget_witness"] is None
    assert capacity["subset_is_complete_closure"] is False
    assert capacity["complete_claim_allowed"] is False
    assert capacity["executed_closure_status"] == "NOT_RUN"
    m3 = m3_entry_contract_report()
    assert m3["required_gate_count"] == len(M3_REQUIRED_GATES) == 24
    assert m3["satisfied_gate_count"] == 14
    assert m3["m3_entry_allowed"] is False


def test_checked_in_shrink1_artifacts_keep_formal_claims_closed() -> None:
    gate = json.loads(
        (ROOT / "artifacts" / "phase3_shrink1_dual_strict_gate_v1.json").read_text()
    )
    capacity = json.loads(
        (ROOT / "artifacts" / "phase3_shrink1_dual_capacity_replay_v1.json").read_text()
    )
    publication = json.loads(
        (ROOT / "artifacts" / "phase3_shrink1_publication_v1.json").read_text()
    )
    transition = json.loads(
        (ROOT / "artifacts" / "phase3_dsl_shrink_transition_v1.json").read_text()
    )
    assert gate["formal_roots"] is None
    assert capacity["complete_closure_enumerated"] is False
    assert capacity["closure_cardinality"] is None
    assert publication["formal_roots"] is None
    assert publication["m3_entry"]["m3_entry_allowed"] is False
    assert transition["parent_status"] == "DSL_TOO_LARGE"
    assert transition["child_initial_state"] == "NOT_RUN"


@pytest.mark.skipif(
    not DEFAULT_RUST_BINARY.is_file(),
    reason="compiled Rust release replay is an execution artifact, not checked in",
)
def test_checked_in_shrink1_artifacts_exactly_replay() -> None:
    expected_gate = json.loads(
        (ROOT / "artifacts" / "phase3_shrink1_dual_strict_gate_v1.json").read_text()
    )
    expected_capacity = json.loads(
        (ROOT / "artifacts" / "phase3_shrink1_dual_capacity_replay_v1.json").read_text()
    )
    expected_publication = json.loads(
        (ROOT / "artifacts" / "phase3_shrink1_publication_v1.json").read_text()
    )
    expected_transition = json.loads(
        (ROOT / "artifacts" / "phase3_dsl_shrink_transition_v1.json").read_text()
    )
    assert dual_shrink1_strict_gate_report() == expected_gate
    assert dual_shrink1_capacity_replay_report() == expected_capacity
    assert shrink1_publication_report() == expected_publication
    assert shrink_transition_report() == expected_transition
