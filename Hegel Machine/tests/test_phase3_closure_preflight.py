from hegel_machine.phase3_closure_preflight import (
    CAPACITY_PROOF,
    CONDITIONAL_CAPACITY_STATUS,
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
    assert proof.capacity_status == CONDITIONAL_CAPACITY_STATUS
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


def test_preflight_fails_closed_and_conditions_any_new_dsl_version():
    report = phase3_closure_capacity_preflight_report()
    assert report["status"] == CONDITIONAL_CAPACITY_STATUS
    assert report["executed_closure_status"] == "NOT_RUN"
    assert report["constructive_candidate_ast_count"] == 64_680
    assert report["complete_closure_enumerated"] is False
    assert report["outside_frozen_closure_certificate_issued"] is False
    assert report["unbounded_outside_language_claim_issued"] is False
    assert report["required_next_action"] == {
        "freeze_strict_canonical_ast_schema_and_acceptance_rules": True,
        "replay_with_formal_canonical_cbor": True,
        "publish_new_dsl_version_if_witnesses_are_accepted": True,
        "conditional_first_frozen_shrink_step": "remove mean_v1, min_v1, max_v1",
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
