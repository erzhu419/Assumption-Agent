from hegel_machine.phase3_closure_preflight import (
    CAPACITY_PROOF,
    DSL_TOO_LARGE_STATUS,
    FIRST_OUT_OF_BUDGET_AST_HASH,
    STRICT_CAPACITY_SET_COMMITMENT,
    phase3_closure_capacity_preflight_report,
    replay_constructive_subset,
)
import json
from pathlib import Path

from hegel_machine.phase3_dsl_v1 import OLD_DSL_V1


def test_constructive_subset_exceeds_frozen_budget_with_valid_shape():
    proof = CAPACITY_PROOF
    assert proof.constant_only_atom_count == 77
    assert proof.one_aggregate_atom_count == 840
    assert proof.witness_candidate_ast_count == 77 * 840 == 64_680
    assert proof.canonical_program_budget == 50_000
    assert proof.first_out_of_budget_ordinal == 50_001
    assert proof.capacity_status == DSL_TOO_LARGE_STATUS
    assert proof.witness_ast_depth <= OLD_DSL_V1.structural_limits.max_total_ast_depth
    assert proof.witness_node_count <= OLD_DSL_V1.structural_limits.max_total_node_count
    assert (
        proof.witness_max_scalar_parameter_occurrences
        <= OLD_DSL_V1.structural_limits.max_fitted_scalar_parameters
    )
    assert proof.witness_aggregate_leaf_count == 1


def test_constructive_generator_has_exact_candidate_ast_count():
    replay = replay_constructive_subset()
    assert (
        replay["observed_unique_candidate_ast_count"]
        == CAPACITY_PROOF.witness_candidate_ast_count
    )


def test_diagnostic_replay_is_explicitly_not_rust_or_formal_cbor():
    replay = replay_constructive_subset()
    assert replay["diagnostic_subset_python_materialization_complete"] is True
    assert replay["observed_unique_candidate_ast_count"] == 64_680
    assert replay["candidate_ast_count_agreement"] is True
    assert replay["typing_and_ast_limits_recomputed_for_every_candidate_ast"] is True
    assert replay["formal_canonical_cbor_archive"] is False
    assert replay["strict_canonicalizer_acceptance_verified"] is False
    assert replay["rust_replay_complete"] is False


def test_preflight_discharges_capacity_condition_but_keeps_formal_gates_closed():
    report = phase3_closure_capacity_preflight_report()
    assert report["status"] == DSL_TOO_LARGE_STATUS
    assert report["executed_closure_status"] == DSL_TOO_LARGE_STATUS
    assert report["capacity_condition_discharged"] is True
    assert report["freeze_version"] == "hegel-freeze-p2b-p3-v1.0.2"
    assert report["constructive_candidate_ast_count"] == 64_680
    assert report["strict_acceptance_specification_complete"] is True
    assert report["strict_acceptance_implementation_verified"] is True
    assert report["strict_rewrite_application_pending"] is False
    assert report["formal_root_generation_allowed"] is False
    assert report["formal_roots"] is None
    assert report["dsl_too_large_claim_allowed"] is True
    assert report["strict_capacity_replay"] == {
        "source_candidate_count": 64_680,
        "type_rejected_count": 0,
        "limit_rejected_count": 0,
        "other_rejected_count": 0,
        "rewrite_collapsed_count": 0,
        "accepted_strict_canonical_count": 64_680,
        "first_accepted_out_of_budget_ordinal": 50_001,
        "first_accepted_out_of_budget_ast_hash": FIRST_OUT_OF_BUDGET_AST_HASH,
        "first_accepted_out_of_budget_cbor_hex": (
            "820182048284020383000002830000048402038600030000008083000000"
        ),
        "python_accepted_set_commitment": STRICT_CAPACITY_SET_COMMITMENT,
        "rust_accepted_set_commitment": STRICT_CAPACITY_SET_COMMITMENT,
        "accepted_set_commitment_is_formal_root": False,
        "dual_replay_equal": True,
    }
    assert report["complete_closure_enumerated"] is False
    assert report["outside_certificate_allowed"] is False
    assert report["outside_frozen_closure_certificate_issued"] is False
    assert report["unbounded_outside_language_claim_issued"] is False
    assert report["target_synthesis_allowed"] is False
    assert report["hidden_sink_formal_verdict_allowed"] is False
    assert report["mdl_certificate_allowed"] is False
    assert report["active_promotion_allowed"] is False
    assert report["phase2b_formal_exit"] is False
    assert report["required_next_action"] == {
        "freeze_strict_canonical_ast_schema_and_acceptance_rules": False,
        "implement_python_strict_acceptance": False,
        "implement_rust_strict_acceptance": False,
        "verify_cross_language_golden_vectors": False,
        "replay_64680_with_strict_canonical_cbor": False,
        "action": "PUBLISH_SHRUNK_OLD_DSL_VERSION_USING_FROZEN_STEP_1",
        "frozen_shrink_step": "remove mean_v1, min_v1, max_v1",
        "regenerate_target_commitments_after_version_change": True,
    }


def test_checked_in_capacity_preflight_artifact_matches_executable_replay():
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "phase3_closure_capacity_preflight_v1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact == phase3_closure_capacity_preflight_report(
        replay_subset=True
    )
