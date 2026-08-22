from __future__ import annotations

import ast
from dataclasses import fields, replace
import hashlib
import inspect
from itertools import combinations
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from hegel_machine import (
    phase2b_c1_owner_actual_unsealed_960_authority_v1 as owner_authority,
    phase2b_actual_unsealed_960_generator_construction_profile_v1 as profile,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / (
    "src/hegel_machine/"
    "phase2b_actual_unsealed_960_generator_construction_profile_v1.py"
)
CONFIG_PATH = ROOT / (
    "config/phase2b_actual_unsealed_960_generator_construction_profile_v1.json"
)
DOC_PATH = ROOT / (
    "docs/"
    "Hegel_Machine_Phase2B_Actual_Unsealed_960_Generator_Construction_Profile_v1.md"
)
OWNER_CONFIG_PATH = ROOT / (
    "config/phase2b_c1_owner_actual_unsealed_960_authorization_v1.json"
)
OWNER_SOURCE_PATH = ROOT / (
    "src/hegel_machine/phase2b_c1_owner_actual_unsealed_960_authority_v1.py"
)
WIRE_SOURCE_PATH = ROOT / "src/hegel_machine/phase2b_formal_trusted_wire_v3.py"
WIRE_TEST_PATH = ROOT / "tests/test_phase2b_formal_trusted_wire_v3.py"
WIRE_CONFIG_PATH = ROOT / "config/phase2b_formal_trusted_wire_v3.json"
WIRE_DOC_PATH = ROOT / "docs/Hegel_Machine_Phase2B_Formal_Trusted_Wire_V3.md"
CORE_SOURCE_PATH = ROOT / "src/hegel_machine/phase2b_exact_transform_core_v1.py"
CORE_TEST_PATH = ROOT / "tests/test_phase2b_exact_transform_core_v1.py"
RFC8785_SOURCE_PATH = ROOT / "src/hegel_machine/rfc8785_v1.py"

PROFILE_DOMAIN = (
    b"HEGEL/PHASE2B/ACTUAL_UNSEALED_960/GENERATOR_CONSTRUCTION_PROFILE/V1\x00"
)
PROFILE_PREFIX = (
    "phase2b_actual_unsealed_960_generator_construction_profile_v1_"
)

PROFILE_ID = (
    "phase2b_actual_unsealed_960_generator_construction_profile_v1_"
    "3d5175cbfb3749ce7f225a55e3d3d045156adf53ef7aea726ea592d4aa82d531"
)
CONFIG_SHA256 = (
    "8d14629f1e449d9e31bdc9a65b38375e3a994cd6c4a0b8e80006248d0d32b4aa"
)


FAMILIES = (
    "F01_symmetry_equivariance",
    "F02_monotonicity_order",
    "F03_conservation_balance",
    "F04_additivity_complementarity",
    "F05_locality_composition",
    "F06_negative_feedback_stability",
)
CASE_MATRIX = (
    ("unique_scale_answerable", 11, 8, 0, 0, 19),
    ("admissible_scale_set_answerable", 0, 0, 0, 1, 1),
    ("wrong_family_hard_negative", 3, 2, 3, 0, 8),
    ("binding_counterfactual", 3, 2, 3, 0, 8),
    ("scale_counterfactual", 2, 3, 3, 0, 8),
    ("sign_or_invariant_break", 2, 3, 3, 0, 8),
    ("insufficient_or_nonidentifiable", 0, 0, 0, 8, 8),
)
PRESERVATION_TOTALS = (
    ("entity_alpha_renaming", 72, 12),
    ("observation_reorder", 48, 8),
    ("irrelevant_entity_augmentation", 72, 12),
    ("unit_conversion", 96, 12),
    ("coordinate_affine_transform", 64, 8),
    ("equivalent_aggregation_split_merge", 48, 6),
    ("nontrivial_scale_map", 60, 12),
    ("sign_convention_reparameterization", 36, 6),
)
NAMESPACES = (
    "bundle_id",
    "observation_id",
    "entity_id",
    "role_candidate_id",
    "quantity_id",
    "context_id",
    "task_id",
    "scale_id",
    "aggregate_map_id",
    "transform_id",
)


def _independent_canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _wrapper() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_bytes())


def _semantic_body() -> dict[str, object]:
    return profile._expected_profile_body_v1()


def _semantic_wrapper() -> dict[str, object]:
    return {"profile_body": _semantic_body(), "profile_id": ""}


def _reidentified_bytes(wrapper: dict[str, object]) -> bytes:
    body = wrapper["profile_body"]
    wrapper["profile_id"] = PROFILE_PREFIX + hashlib.sha256(
        PROFILE_DOMAIN + _independent_canonical(body)
    ).hexdigest()
    return _independent_canonical(wrapper)


def test_independent_quota_and_preservation_arithmetic_kat() -> None:
    assert tuple(sum(row[index] for row in CASE_MATRIX) for index in range(1, 5)) == (
        21,
        18,
        12,
        9,
    )
    assert tuple(row[-1] for row in CASE_MATRIX) == (19, 1, 8, 8, 8, 8, 8)
    assert sum(row[-1] for row in CASE_MATRIX) == 60
    assert 6 * 2 * 60 == 720
    assert 6 * 2 * (10 + 10) == 240
    assert sum(row[1] for row in PRESERVATION_TOTALS) == 496
    assert sum(row[2] for row in PRESERVATION_TOTALS) == 76
    assert 496 + 76 == 572
    assert 720 + 240 == 960
    assert 960 <= 1024 and 572 <= 1024 and 960 + 572 > 1024


def test_descriptor_slots_and_preservation_permutation_kat() -> None:
    body = _wrapper()["profile_body"]
    descriptor = body["semantic_descriptor"]
    token_rows = [descriptor["neutral_token_bank_in_order"]]
    token_rows.extend(
        row["tokens_in_order"] for row in descriptor["family_lexicons_in_order"]
    )
    for row in token_rows:
        for token in row:
            assert token.isascii() and token.isalpha() and token.islower()
            padded = token + (" " * (24 - len(token.encode("utf-8"))))
            assert len(padded.encode("utf-8")) == 24
            assert padded.rstrip(" ") == token

    base = body["preservation_manifest"]["base_permutation"]
    assert base["formula"] == "(17*k+11*family_index+7*scale_index) MOD 60"
    assert base["maximum_consumed_slots_by_family_in_order"] == [45, 45, 52, 45, 54, 45]
    for family_index in range(6):
        for scale_index in range(2):
            permutation = tuple(
                (17 * k + 11 * family_index + 7 * scale_index) % 60
                for k in range(60)
            )
            assert len(set(permutation)) == 60
            consumed = base["maximum_consumed_slots_by_family_in_order"][family_index]
            assert len(set(permutation[:consumed])) == consumed


def test_uncertainty_and_scale_regret_literals_are_closed() -> None:
    body = _wrapper()["profile_body"]
    oracles = body["oracle_contracts"]
    assert oracles["uncertainty"] == {
        "allowed_kinds": ["absolute_bound"],
        "standard_error_encounter_policy": (
            "ABORT_WHOLE_PROFILE_BUILD_NO_PARTIAL_OUTPUT"
        ),
        "standard_error_status": "STANDARD_ERROR_UNSUPPORTED",
    }
    assert oracles["scale_oracle"]["point_estimate_maximum"] == "0.05"
    assert (
        oracles["scale_oracle"]["one_sided_bootstrap_upper_bound_maximum"]
        == "0.08"
    )
    assert body["dag_contract"]["r05_parallel_successors_in_order"] == [
        "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
        "R07_BASELINE_REGISTRATIONS_FROZEN",
    ]
    assert body["dag_contract"]["r08_successor_dependencies_in_order"] == [
        "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
        "R07_BASELINE_REGISTRATIONS_FROZEN",
    ]


def test_owner_authorization_and_exact_five_fresh_secret_contract() -> None:
    expected_id = (
        "phase2b_c1_owner_actual_unsealed_960_authorization_v1_"
        "dd2cd6a98b429fe01f172daf16ea70404c91356e40e9e42daed4cb5416a5c50d"
    )
    owner_wrapper = json.loads(OWNER_CONFIG_PATH.read_bytes())
    contract = _semantic_body()["authorization_and_fresh_secret_contract"]
    binding = contract["authorization_binding"]
    assert owner_wrapper["authorization_id"] == expected_id
    assert owner_authority.PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_ID == expected_id
    assert binding["authorization_id"] == expected_id
    assert hashlib.sha256(OWNER_CONFIG_PATH.read_bytes()).hexdigest() == binding[
        "authorization_config_sha256"
    ]
    assert hashlib.sha256(OWNER_SOURCE_PATH.read_bytes()).hexdigest() == binding[
        "authorization_source_sha256"
    ]
    assert binding["live_active_and_not_revoked_verified_by_profile"] is False

    secrets = contract["fresh_secret_contract"]
    assert secrets["count"] == 5
    assert secrets["raw_bytes_each"] == 32
    assert secrets["purposes_in_order"] == [
        "latent_generator_seed",
        "answer_seed",
        "K_shuffle",
        "K_id",
        "K_padding",
    ]
    assert secrets["pairwise_distinct_raw_values_required"] is True
    assert secrets["commitments_pairwise_distinct_required"] is True
    assert secrets["cross_purpose_same_raw_value_rejected_before_commitment"] is True
    assert secrets["cross_purpose_same_commitment_rejected_before_store"] is True
    assert (
        secrets["custody_attestation_must_bind_pairwise_raw_value_distinctness"]
        is True
    )
    assert secrets["independent_source_attestation_required_per_purpose"] is True
    assert secrets["custody_receipt_required_per_purpose"] is True
    assert secrets["commitment_rows_fields_in_order"] == [
        "purpose",
        "commitment",
        "independent_source_attestation_root",
        "custody_receipt_root",
    ]
    assert secrets["statistical_independence_inferred_from_distinctness"] is False
    exposure = contract["custody_and_exposure"]
    assert all(
        exposure[name] is False
        for name in (
            "answer_manifest_plaintext_allowed_for_baselines",
            "answer_manifest_plaintext_allowed_for_recognizer",
            "answer_manifest_content_allowed_for_baselines",
            "answer_manifest_content_allowed_for_recognizer",
            "generator_source_archive_or_bytes_allowed_for_baselines",
            "generator_source_archive_or_bytes_allowed_for_recognizer",
            "generator_source_manifest_allowed_for_baselines",
            "generator_source_manifest_allowed_for_recognizer",
            "seed_manifest_content_allowed_for_baselines",
            "seed_manifest_content_allowed_for_recognizer",
            "seed_manifest_allowed_for_baselines",
            "seed_manifest_allowed_for_recognizer",
        )
    )


def test_three_uuid_domains_are_globally_and_per_namespace_disjoint() -> None:
    contract = _semantic_body()["uuid_domain_contract"]
    assert contract["namespaces_in_order"] == list(NAMESPACES)
    assert contract["per_namespace_and_global_disjointness_both_required"] is True
    assert contract["pairwise_empty_intersections_in_order"] == [
        "PRIVATE_SOURCE_INTERSECT_PRIMARY_PUBLIC_TARGET",
        "PRIVATE_SOURCE_INTERSECT_VARIANT_PUBLIC_TARGET",
        "PRIMARY_PUBLIC_TARGET_INTERSECT_VARIANT_PUBLIC_TARGET",
    ]
    assert contract["primary_or_variant_target_may_reuse_any_private_or_public_uuid"] is False
    assert all(contract["global_rules"].values())


def test_nontrivial_scale_map_has_exact_5_5_and_3_3_2_2_schedule() -> None:
    schedule = _semantic_body()["preservation_manifest"][
        "nontrivial_scale_map_schedule"
    ]
    rows = schedule["ordinal_rows_in_order"]
    assert [row["ordinal"] for row in rows] == list(range(10))
    assert [row["base_scale"] for row in rows].count("S0_FINE_LOCAL") == 5
    assert [row["base_scale"] for row in rows].count("S1_COARSE_AGGREGATE") == 5
    assert {row["target_scale"] for row in rows[:5]} == {"T0_FOR_S0"}
    assert {row["target_scale"] for row in rows[5:]} == {"T1_FOR_S1"}
    operation_counts = {
        operation: sum(row["operation"] == operation for row in rows)
        for operation in (
            "temporal_aggregation",
            "spatial_aggregation",
            "sampling_resolution",
            "coarse_graining",
        )
    }
    assert operation_counts == {
        "temporal_aggregation": 3,
        "spatial_aggregation": 3,
        "sampling_resolution": 2,
        "coarse_graining": 2,
    }
    assert len({(row["operation"], row["parameter_profile"]) for row in rows}) == 10


def test_challenge_statistics_are_owned_by_exact_future_r08_binding() -> None:
    challenge = _semantic_body()["challenge_descriptor"]
    encoded = json.dumps(challenge, sort_keys=True)
    assert "minimum" not in encoded
    assert "point_estimate" not in encoded
    binding = challenge["r08_evaluation_binding"]
    assert binding["evaluation_owner_node"] == "R08_STATISTICAL_VALIDATION_PASSED"
    assert binding["generator_profile_defines_thresholds"] is False
    assert binding["generator_profile_evaluates_wilson"] is False
    assert binding["generator_profile_evaluates_bootstrap"] is False
    assert binding["required_content_addresses_in_order"] == [
        "r08_validation_version_receipt_root",
        "challenge_threshold_table_root",
        "one_sided_wilson_contract_root",
        "paired_cluster_bootstrap_contract_root",
    ]


def test_r06_synthetic_dev_qualification_contract_is_non_generating_and_bounded() -> None:
    contract = _semantic_body()["r06_qualification_contract"]
    assert contract["input_class"] == "SYNTHETIC_DEV_ONLY"
    assert contract["actual_corpus_input_allowed"] is False
    assert contract["issuance_status"] == "FUTURE_REQUIRED_NOT_ISSUED"
    assert contract["target_node"] == "R06_GENERATOR_IMPLEMENTATION_QUALIFIED"
    assert contract["required_artifact_bindings_in_order"] == [
        "source_commit_sha",
        "source_archive_root",
        "dependency_lock_sha256",
        "runtime_image_digest",
        "synthetic_fixture_root",
    ]
    assert contract["fixture_contracts_in_order"][-2:] == [
        ["SYNTHETIC_WIRE_CAPACITY", 1024, "ACCEPT"],
        ["SYNTHETIC_WIRE_CAPACITY_PLUS_ONE", 1025, "ATOMIC_REJECT"],
    ]
    assert contract["deterministic_replay"] == {
        "independent_fresh_process_replays": 2,
        "replay_outputs_required_byte_identical": True,
        "replay_roots_required_equal": True,
        "synthetic_public_seed_manifest_only": True,
    }
    assert contract["resource_limits"] == {
        "maximum_logical_cpus": 16,
        "maximum_output_bytes": 134217728,
        "maximum_rss_bytes": 12884901888,
        "maximum_wall_seconds": 1800,
    }
    cache = contract["qualification_cache_schema"]
    assert set(cache["body_fields_in_order"]) == set(cache["body_field_types"])
    assert cache["wrapper_fields_in_order"] == ["cache_body", "cache_id"]
    assert cache["required_literals"]["actual_input_count"] == 0
    assert cache["required_literals"]["tests_failed"] == 0
    assert cache["required_literals"]["evidence_class"] == "NON_EVIDENCE_CACHE_ONLY"
    assert cache["body_field_types"]["source_archive_root"] == "CONTENT_ADDRESS"
    assert cache["self_asserted_all_pass_is_admission_evidence"] is False
    policy = contract["cache_evidence_policy"]
    assert policy["cache_receipt_is_admission_evidence"] is False
    assert policy["cache_receipt_may_satisfy_r06"] is False
    assert policy["fresh_admission_rerun_required"] is True
    assert policy["forged_all_pass_cache_disposition"] == "IGNORE_AND_FRESH_RERUN"
    rerun = contract["fresh_admission_rerun_contract"]
    assert rerun["admission_controller_must_not_trust_cache_values"] is True
    assert rerun["recompute_all_kat_groups"] is True
    assert rerun["fresh_process_replays"] == 2
    assert rerun["fresh_rerun_occurs_inside_admission_decision_not_from_prior_cache"] is True
    assert rerun["trusted_qualifier_identity_must_be_pinned_before_rerun"] is True
    assert rerun["pinned_execution_bindings_in_order"][0] == "qualifier_identity_id"
    forged_all_pass_cache = {
        "tests_collected": 100,
        "tests_passed": 100,
        "tests_failed": 0,
        "atomic_success": True,
    }
    assert forged_all_pass_cache["tests_passed"] == forged_all_pass_cache["tests_collected"]
    assert policy["all_pass_fields_self_authenticate"] is False
    assert contract["r06_admission_can_consume_cache_without_fresh_rerun"] is False


def test_future_generation_receipt_is_closed_attempt_bound_and_not_issued() -> None:
    receipt = _semantic_body()["generation_receipt_schema"]
    assert receipt["issuance_status"] == "FUTURE_REQUIRED_NOT_ISSUED"
    assert receipt["issued_by_profile_verifier"] is False
    assert receipt["wrapper_fields_in_order"] == ["receipt_body", "receipt_id"]
    assert set(receipt["body_fields_in_order"]) == set(receipt["body_field_types"])
    root_index = receipt["body_fields_in_order"].index("body_signed_root")
    assert receipt["body_signed_root_covered_fields_in_order"] == receipt[
        "body_fields_in_order"
    ][:root_index]
    assert receipt["body_fields_in_order"][root_index:] == [
        "body_signed_root",
        "durable_store_transaction",
        "final_receipt_root",
        "purpose_signatures",
    ]
    assert receipt["transport"] == {
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "floats_or_nonfinite_allowed": False,
        "integers": "EXACT_0_THROUGH_2_POW_53_MINUS_1",
        "maximum_array_items": 1024,
        "maximum_bytes": 262144,
        "maximum_depth": 32,
        "maximum_nodes": 20000,
        "maximum_string_utf8_bytes": 4096,
        "unknown_or_duplicate_fields_allowed": False,
        "campaign_attempt_id_type": "FROZEN_PREFIX_PLUS_64_LOWER_HEX",
    }
    nested = receipt["closed_nested_schemas"]
    assert nested["readiness_evidence_roots"]["exact_count"] == 13
    for node in (
        "R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED",
        "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
        "R07_BASELINE_REGISTRATIONS_FROZEN",
        "R08_STATISTICAL_VALIDATION_PASSED",
        "R11_EXECUTION_FREEZE_COMMITMENTS_BOUND",
        "R12_INDEPENDENT_LAUNCH_OVERLAY_VERIFIED",
        "R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY",
    ):
        assert node in nested["readiness_evidence_roots"]["nodes_in_order"]
    assert nested["fresh_secret_commitments"]["exact_count"] == 5
    assert nested["fresh_secret_commitments"]["raw_secret_or_seed_allowed"] is False
    assert nested["fresh_secret_commitments"]["commitments_pairwise_distinct_required"] is True
    assert (
        nested["fresh_secret_commitments"][
            "cross_purpose_same_commitment_rejected_before_store"
        ]
        is True
    )
    assert (
        nested["fresh_secret_commitments"][
            "cross_purpose_same_raw_value_rejected_before_commitment"
        ]
        is True
    )
    assert nested["fresh_secret_commitments"]["fields_per_row_in_order"] == [
        "purpose",
        "commitment",
        "independent_source_attestation_root",
        "custody_receipt_root",
    ]
    assert nested["private_partition"]["required_counts"] == [720, 240, 960]
    assert nested["public_partition"]["required_counts"] == [720, 240, 960, 572]
    store = nested["durable_store_transaction"]
    assert set(store["fields_in_order"]) == set(store["field_types"])
    attempt_cas = store["attempt_state_transition"]
    purpose_cas = store["per_purpose_commitment_consumptions"]
    commitment_set_cas = store["full_commitment_set_consumption"]
    assert attempt_cas["required_expected_generation"] == 0
    assert attempt_cas["required_new_generation"] == 1
    assert attempt_cas["new_unsigned_receipt_root_exactly_equals"] == "BODY_SIGNED_ROOT"
    assert attempt_cas["attempt_state_key_must_recompute_from_top_level_scope"] is True
    assert store["store_operation"] == "ATOMIC_SEVEN_KEY_COMPARE_AND_SWAP_ALL_OR_NONE"
    assert store["exact_key_count"] == 7
    assert store["seven_state_keys_pairwise_distinct_required"] is True
    assert store["all_seven_preconditions_checked_before_any_mutation"] is True
    assert store["any_used_key_rejects_entire_transaction"] is True
    assert store["winner_binding_rows_count"] == 6
    assert store["winner_rows_bind_purpose_campaign_attempt_body_and_final_root"] is True
    assert store["winning_final_root_is_derived_and_omitted_only_from_request_root"] is True
    assert (
        store["store_must_recompute_and_verify_all_six_winner_final_roots_before_commit"]
        is True
    )
    assert store["derived_final_root_inserted_before_atomic_seven_key_commit"] is True
    assert store["fresh_seven_key_readback_required_before_purpose_signatures"] is True
    assert store["atomic_store_request_root_normalization"] == (
        "OMIT_ONLY_WINNING_FINAL_RECEIPT_ROOT_FROM_EXACT_FIVE_"
        "PER_PURPOSE_ROWS_AND_ONE_FULL_SET_ROW"
    )
    assert store["atomic_store_request_normalized_object_fields_in_order"] == [
        "schema_version",
        "attempt_state_transition",
        "per_purpose_commitment_consumptions_without_winning_final_receipt_root",
        "full_commitment_set_consumption_without_winning_final_receipt_root",
    ]
    assert store["computation_and_commit_order_in_order"][-5:] == [
        "INSERT_FINAL_ROOT_IN_SIX_WINNER_ROWS",
        "VERIFY_ALL_SEVEN_CAS_PRECONDITIONS",
        "ATOMIC_COMMIT_ALL_SEVEN",
        "FRESH_READBACK_ALL_SEVEN",
        "PURPOSE_SIGNATURES",
    ]
    assert "exact_key_count" in store["atomic_store_receipt_root_formula"]
    assert "ATOMIC_STORE_RECEIPT_ROOT_BYTES" in store["transaction_root_formula"]
    assert purpose_cas["exact_count"] == 5
    assert purpose_cas["purposes_in_order"] == [
        "latent_generator_seed",
        "answer_seed",
        "K_shuffle",
        "K_id",
        "K_padding",
    ]
    assert purpose_cas["overlap_cardinalities_rejected_in_order"] == [1, 2, 3, 4, 5]
    assert purpose_cas["any_used_or_consumed_key_rejects_entire_transaction"] is True
    assert "PURPOSE_UTF8" in purpose_cas["commitment_state_key_formula"]
    assert "COMMITMENT_DIGEST_32_BYTES" in purpose_cas["commitment_state_key_formula"]
    assert (
        purpose_cas[
            "state_key_excludes_authorization_profile_overlay_campaign_attempt"
        ]
        is True
    )
    assert set(purpose_cas["fields_per_row_in_order"]) == set(
        purpose_cas["row_field_types"]
    )
    for winner_field in (
        "purpose",
        "winning_campaign_id",
        "winning_attempt_id",
        "winning_body_signed_root",
        "winning_final_receipt_root",
    ):
        assert winner_field in purpose_cas["fields_per_row_in_order"]
        assert winner_field in commitment_set_cas["fields_in_order"]
    assert commitment_set_cas["state_key_must_differ_from_attempt_state_key"] is True
    assert commitment_set_cas["required_purpose"] == "FULL_FIVE_COMMITMENT_SET"
    assert (
        commitment_set_cas[
            "commitment_set_id_must_recompute_from_top_level_fresh_secret_commitments"
        ]
        is True
    )
    assert (
        commitment_set_cas["two_fresh_attempts_same_commitments_exactly_one_may_win"]
        is True
    )
    assert "CAMPAIGN_ID" not in commitment_set_cas["commitment_set_state_key_formula"]
    assert "ATTEMPT_ID" not in commitment_set_cas["commitment_set_state_key_formula"]
    assert "AUTHORIZATION_ID" not in commitment_set_cas["commitment_set_state_key_formula"]
    assert "PROFILE_ID" not in commitment_set_cas["commitment_set_state_key_formula"]
    assert "CAMPAIGN_ID" in attempt_cas["attempt_state_key_formula"]
    assert "ATTEMPT_ID" in attempt_cas["attempt_state_key_formula"]
    assert nested["logical_time_chain"]["strictly_contiguous_from_zero"] is True
    registry = nested["signature_key_registry"]
    signatures = nested["purpose_signatures"]
    assert set(registry["fields_in_order"]) == set(registry["field_types"])
    assert set(registry["row_fields_in_order"]) == set(registry["row_field_types"])
    assert set(signatures["fields_per_row_in_order"]) == set(
        signatures["row_field_types"]
    )
    assert len(signatures["purposes_in_order"]) == 4
    assert registry["row_purposes_in_order"] == signatures["purposes_in_order"]
    assert registry["row_key_ids_pairwise_distinct"] is True
    assert registry["row_public_keys_pairwise_distinct"] is True
    assert registry["registry_root_must_be_freshly_recomputed"] is True
    assert (
        registry["launch_overlay_must_pin_epoch_registry_and_revocation_snapshot_roots"]
        is True
    )
    assert registry["status_change_after_overlay_invalidates_attempt"] is True
    assert "PURPOSE_UTF8" in signatures["signed_preimage_formula"]
    assert "KEY_ID_UTF8" in signatures["signed_preimage_formula"]
    assert "REGISTRY_ROOT_BYTES" in signatures["signed_preimage_formula"]
    assert "FINAL_RECEIPT_ROOT_BYTES" in signatures["signed_preimage_formula"]
    assert signatures["signatures_cover_store_transcript_via_final_receipt_root"] is True
    assert signatures["all_four_signatures_required_and_must_verify"] is True
    assert "DURABLE_STORE_TRANSACTION_ROOT_BYTES" in receipt["final_receipt_root_formula"]
    assert receipt["signature_or_store_side_effect_performed_by_profile_verifier"] is False
    assert receipt["cross_attempt_and_replay_rejection"][
        "fresh_secret_overlap_cardinalities_rejected_in_order"
    ] == [1, 2, 3, 4, 5]
    assert receipt["cross_attempt_and_replay_rejection"][
        "store_transaction_requires_exact_seven_key_atomicity"
    ] is True
    assert all(receipt["cross_attempt_and_replay_rejection"].values())


def test_cas_splice_commitment_reuse_registry_relabel_and_cache_promotion_attacks_reject() -> None:
    attacks: list[dict[str, object]] = []

    cas_splice = _semantic_wrapper()
    cas_splice["profile_body"]["generation_receipt_schema"]["final_receipt_root_formula"] = (
        "SHA256(BODY_SIGNED_ROOT_BYTES)"
    )
    attacks.append(cas_splice)

    cas_wrong_body_root = _semantic_wrapper()
    cas_wrong_body_root["profile_body"]["generation_receipt_schema"][
        "closed_nested_schemas"
    ]["durable_store_transaction"]["attempt_state_transition"][
        "new_unsigned_receipt_root_exactly_equals"
    ] = "ARBITRARY_ROOT"
    attacks.append(cas_wrong_body_root)

    six_key_store = _semantic_wrapper()
    six_key_store["profile_body"]["generation_receipt_schema"][
        "closed_nested_schemas"
    ]["durable_store_transaction"]["exact_key_count"] = 6
    attacks.append(six_key_store)

    used_key_allowed = _semantic_wrapper()
    used_key_allowed["profile_body"]["generation_receipt_schema"][
        "closed_nested_schemas"
    ]["durable_store_transaction"]["any_used_key_rejects_entire_transaction"] = False
    attacks.append(used_key_allowed)

    unbound_winner_final_root = _semantic_wrapper()
    unbound_winner_final_root["profile_body"]["generation_receipt_schema"][
        "closed_nested_schemas"
    ]["durable_store_transaction"][
        "winner_rows_bind_purpose_campaign_attempt_body_and_final_root"
    ] = False
    attacks.append(unbound_winner_final_root)

    same_commitments_two_attempts = _semantic_wrapper()
    same_commitments_two_attempts["profile_body"]["generation_receipt_schema"][
        "closed_nested_schemas"
    ]["durable_store_transaction"]["full_commitment_set_consumption"][
        "two_fresh_attempts_same_commitments_exactly_one_may_win"
    ] = False
    attacks.append(same_commitments_two_attempts)

    attempt_scoped_commitment_key = _semantic_wrapper()
    attempt_scoped_commitment_key["profile_body"]["generation_receipt_schema"][
        "closed_nested_schemas"
    ]["durable_store_transaction"]["full_commitment_set_consumption"][
        "commitment_set_state_key_formula"
    ] += "||ATTEMPT_ID_UTF8"
    attacks.append(attempt_scoped_commitment_key)

    for overlap_cardinality in range(1, 5):
        partial_reuse_allowed = _semantic_wrapper()
        rejected = partial_reuse_allowed["profile_body"]["generation_receipt_schema"][
            "closed_nested_schemas"
        ]["durable_store_transaction"]["per_purpose_commitment_consumptions"][
            "overlap_cardinalities_rejected_in_order"
        ]
        rejected.remove(overlap_cardinality)
        attacks.append(partial_reuse_allowed)

    same_commitment_cross_purpose_allowed = _semantic_wrapper()
    same_commitment_cross_purpose_allowed["profile_body"][
        "authorization_and_fresh_secret_contract"
    ]["fresh_secret_contract"][
        "cross_purpose_same_commitment_rejected_before_store"
    ] = False
    attacks.append(same_commitment_cross_purpose_allowed)

    receipt_same_commitment_cross_purpose_allowed = _semantic_wrapper()
    receipt_same_commitment_cross_purpose_allowed["profile_body"][
        "generation_receipt_schema"
    ]["closed_nested_schemas"]["fresh_secret_commitments"][
        "cross_purpose_same_commitment_rejected_before_store"
    ] = False
    attacks.append(receipt_same_commitment_cross_purpose_allowed)

    same_raw_value_cross_purpose_allowed = _semantic_wrapper()
    same_raw_value_cross_purpose_allowed["profile_body"][
        "authorization_and_fresh_secret_contract"
    ]["fresh_secret_contract"][
        "cross_purpose_same_raw_value_rejected_before_commitment"
    ] = False
    attacks.append(same_raw_value_cross_purpose_allowed)

    rogue_registry = _semantic_wrapper()
    rogue_registry["profile_body"]["generation_receipt_schema"][
        "closed_nested_schemas"
    ]["signature_key_registry"]["row_public_keys_pairwise_distinct"] = False
    attacks.append(rogue_registry)

    unpinned_registry = _semantic_wrapper()
    unpinned_registry["profile_body"]["generation_receipt_schema"][
        "closed_nested_schemas"
    ]["signature_key_registry"][
        "launch_overlay_must_pin_epoch_registry_and_revocation_snapshot_roots"
    ] = False
    attacks.append(unpinned_registry)

    purpose_relabel = _semantic_wrapper()
    purpose_relabel["profile_body"]["generation_receipt_schema"][
        "closed_nested_schemas"
    ]["purpose_signatures"]["purpose_relabel_or_key_substitution_rejected"] = False
    attacks.append(purpose_relabel)

    swapped_purposes = _semantic_wrapper()
    signature_purposes = swapped_purposes["profile_body"]["generation_receipt_schema"][
        "closed_nested_schemas"
    ]["purpose_signatures"]["purposes_in_order"]
    signature_purposes[0], signature_purposes[1] = signature_purposes[1], signature_purposes[0]
    attacks.append(swapped_purposes)

    forged_all_pass_cache = _semantic_wrapper()
    forged_all_pass_cache["profile_body"]["r06_qualification_contract"][
        "cache_evidence_policy"
    ]["cache_receipt_is_admission_evidence"] = True
    attacks.append(forged_all_pass_cache)

    for attack in attacks:
        with pytest.raises(profile.GeneratorConstructionProfileErrorV1):
            profile.verify_phase2b_actual_unsealed_960_generator_construction_profile_v1(
                _reidentified_bytes(attack)
            )


def test_seven_key_cas_rejects_every_partial_overlap_and_cross_purpose_reuse() -> None:
    purposes = (
        "latent_generator_seed",
        "answer_seed",
        "K_shuffle",
        "K_id",
        "K_padding",
    )

    def purpose_state_keys(commitments: tuple[str, ...]) -> frozenset[tuple[str, str]]:
        return frozenset(zip(purposes, commitments, strict=True))

    first_commitments = tuple(f"commitment-a-{index}" for index in range(5))
    first_keys = purpose_state_keys(first_commitments)
    assert len(first_keys) == 5
    for overlap_cardinality in range(1, 5):
        for reused_indices in combinations(range(5), overlap_cardinality):
            reused = frozenset(reused_indices)
            second_commitments = tuple(
                first_commitments[index]
                if index in reused
                else f"commitment-b-{overlap_cardinality}-{index}"
                for index in range(5)
            )
            second_keys = purpose_state_keys(second_commitments)
            assert len(first_keys & second_keys) == overlap_cardinality
            assert not first_keys.isdisjoint(second_keys)
            committed_keys = set(first_keys)
            before = committed_keys.copy()
            transaction_admitted = committed_keys.isdisjoint(second_keys)
            if transaction_admitted:
                committed_keys.update(second_keys)
            assert transaction_admitted is False
            assert committed_keys == before

    same_commitment_cross_purpose = (
        "same-commitment",
        "same-commitment",
        "commitment-2",
        "commitment-3",
        "commitment-4",
    )
    assert len(set(same_commitment_cross_purpose)) != 5
    assert len(purpose_state_keys(same_commitment_cross_purpose)) == 5
    same_commitment_set_admitted = len(set(same_commitment_cross_purpose)) == 5
    assert same_commitment_set_admitted is False
    assert (
        _semantic_body()["authorization_and_fresh_secret_contract"][
            "fresh_secret_contract"
        ]["cross_purpose_same_commitment_rejected_before_store"]
        is True
    )

    same_raw_value_cross_purpose = (
        b"same-raw-value",
        b"same-raw-value",
        b"raw-2",
        b"raw-3",
        b"raw-4",
    )
    assert len(set(same_raw_value_cross_purpose)) != 5
    same_raw_value_set_admitted = len(set(same_raw_value_cross_purpose)) == 5
    assert same_raw_value_set_admitted is False
    assert (
        _semantic_body()["authorization_and_fresh_secret_contract"][
            "fresh_secret_contract"
        ]["cross_purpose_same_raw_value_rejected_before_commitment"]
        is True
    )


def test_section_level_provenance_is_total_and_unambiguous() -> None:
    body = _semantic_body()
    rows = body["decision_provenance"]["section_provenance_in_order"]
    provenance = dict(rows)
    expected_sections = set(body) - {
        "decision_provenance",
        "document_id",
        "schema_version",
        "status",
    }
    assert set(provenance) == expected_sections
    assert provenance["wire_binding"] == (
        "FINAL_WIRE_DEPENDENCY_PINS_VERIFIED_STATIC_GO_REPORTED_NOT_R03_R04_R05_R06"
    )
    assert all(
        "EXISTING" in status
        or "NEW" in status
        or status == "OWNER_AUTHORIZATION_DERIVED_PLUS_NEW_EXECUTION_COMPANION"
        or status
        == "FINAL_WIRE_DEPENDENCY_PINS_VERIFIED_STATIC_GO_REPORTED_NOT_R03_R04_R05_R06"
        for status in provenance.values()
    )


def test_public_result_constructor_and_resource_rejections_do_not_need_config() -> None:
    false_claims = _semantic_body()["claim_boundary"]["false_claims"]
    for name in (
        "r03_satisfied",
        "r04_satisfied",
        "r06_qualification_executed",
        "r06_qualification_cache_issued",
        "qualification_cache_admission_evidence",
        "r06_qualification_receipt_issued",
        "wire_replay_executed",
        "custody_verified",
        "technical_role_independence_verified",
        "generator_seed_commitment_issued",
        "answer_seed_commitment_issued",
        "commitment_set_consumed",
        "per_purpose_commitments_consumed",
        "durable_store_transaction_executed",
        "final_receipt_root_issued",
        "signature_key_registry_verified",
        "public_variant_uuid_disjointness_verified",
    ):
        assert false_claims[name] is False
    result = profile.GeneratorConstructionProfileVerificationV1(
        schema_version=profile.PROFILE_SCHEMA_VERSION_V1,
        profile_id="cooperative-forgeable-dto",
        disposition=profile.GeneratorConstructionProfileDispositionV1(
            "PROFILE_VERIFIED_NOT_GENERATOR_NOT_GENERATED"
        ),
        claim_level=profile.PROFILE_CLAIM_LEVEL_V1,
        canonical_profile_sha256="0" * 64,
        profile=MappingProxyType({}),
        contract=profile._contract_v1(),
    )
    assert result.result_object_unforgeable is False
    assert result.is_execution_evidence is False
    assert result.r03_satisfied is False
    assert result.r04_satisfied is False
    assert result.wire_replay_executed is False
    with pytest.raises(profile.GeneratorConstructionProfileErrorV1) as oversized:
        profile.verify_phase2b_actual_unsealed_960_generator_construction_profile_v1(
            b"x" * 262145
        )
    assert oversized.value.code == "PROFILE_SIZE_INVALID"
    with pytest.raises(profile.GeneratorConstructionProfileErrorV1) as too_many_nodes:
        profile._validate_json_shape([0] * 100001)
    assert too_many_nodes.value.code == "PROFILE_JSON_INVALID"


def test_internal_contract_is_immutable_and_exact() -> None:
    contract = profile._contract_v1()
    assert contract.families == FAMILIES
    assert contract.private_scale_classes == ("S0_FINE_LOCAL", "S1_COARSE_AGGREGATE")
    assert contract.private_uuid_namespaces == NAMESPACES
    assert tuple(
        (
            row.case_type,
            row.clear_interior,
            row.moderate,
            row.near_boundary_identifiable,
            row.nonunique_or_insufficient,
            row.total_per_family_scale_cell,
        )
        for row in contract.case_margin_rows
    ) == CASE_MATRIX
    assert tuple(
        (row.transform, row.legal_total, row.invalid_total)
        for row in contract.preservation_rows
    ) == PRESERVATION_TOTALS
    assert contract.generation_receipt_status == "FUTURE_REQUIRED_NOT_ISSUED"
    with pytest.raises((AttributeError, TypeError)):
        contract.main_case_count = 0  # type: ignore[misc]


def test_result_dto_is_publicly_constructible_and_explicitly_not_evidence() -> None:
    result = profile.verify_phase2b_actual_unsealed_960_generator_construction_profile_v1(
        CONFIG_PATH.read_bytes()
    )
    clone = profile.GeneratorConstructionProfileVerificationV1(
        **{item.name: getattr(result, item.name) for item in fields(result)}
    )
    assert clone == result
    assert clone.result_object_unforgeable is False
    assert clone.is_execution_evidence is False
    assert clone.fresh_original_bytes_verification_required is True
    for name in (
        "generator_implemented",
        "generator_qualified",
        "actual_rows_generated",
        "actual_bytes_materialized",
        "actual_corpus_audited",
        "authority_verified",
        "origin_authenticated",
        "r03_satisfied",
        "r04_satisfied",
        "r05_verified",
        "r06_qualification_executed",
        "r06_qualification_cache_issued",
        "qualification_cache_admission_evidence",
        "r06_qualification_receipt_issued",
        "r06_satisfied",
        "r07_satisfied",
        "r08_satisfied",
        "r14_satisfied",
        "r15_satisfied",
        "r16_satisfied",
        "entropy_used",
        "identifiers_allocated",
        "filesystem_accessed",
        "network_accessed",
        "docker_invoked",
        "keys_generated",
        "signatures_generated",
        "signature_key_registry_verified",
        "commitment_issued",
        "commitment_set_consumed",
        "per_purpose_commitments_consumed",
        "generator_seed_commitment_issued",
        "answer_seed_commitment_issued",
        "generation_receipt_issued",
        "final_receipt_root_issued",
        "durable_store_transaction_executed",
        "custody_verified",
        "technical_role_independence_verified",
        "deterministic_replay_executed",
        "public_variant_uuid_disjointness_verified",
        "wire_replay_executed",
        "scoring_performed",
    ):
        assert getattr(clone, name) is False
    with pytest.raises(ValueError):
        replace(clone, r06_satisfied=True)


def test_source_ast_has_one_callable_api_and_no_effect_capability() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.split(".", 1)[0])
    assert not imports.intersection(
        {
            "asyncio",
            "datetime",
            "docker",
            "http",
            "importlib",
            "multiprocessing",
            "os",
            "pathlib",
            "random",
            "requests",
            "secrets",
            "shutil",
            "socket",
            "subprocess",
            "tempfile",
            "time",
            "urllib",
            "uuid",
        }
    )
    forbidden_calls = {
        "open",
        "exec",
        "eval",
        "compile",
        "system",
        "popen",
        "run",
        "check_call",
        "check_output",
        "urlopen",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            literal_keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and type(key.value) is str
            ]
            assert len(literal_keys) == len(set(literal_keys))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls
    public_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    assert public_functions == {
        "verify_phase2b_actual_unsealed_960_generator_construction_profile_v1"
    }
    assert profile.__all__ == [
        "verify_phase2b_actual_unsealed_960_generator_construction_profile_v1"
    ]
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(token in node.name.lower() for token in ("generate", "materialize", "issue"))
        for node in ast.walk(tree)
    )


def test_public_verifier_signature_accepts_only_exact_profile_bytes() -> None:
    function = profile.verify_phase2b_actual_unsealed_960_generator_construction_profile_v1
    assert tuple(inspect.signature(function).parameters) == ("profile_bytes",)
    with pytest.raises(profile.GeneratorConstructionProfileErrorV1) as caught:
        function(bytearray(CONFIG_PATH.read_bytes()))  # type: ignore[arg-type]
    assert caught.value.code == "WRONG_INPUT_TYPE"


def test_config_is_exact_rfc8785_without_trailing_lf_and_independent_id_kat() -> None:
    payload = CONFIG_PATH.read_bytes()
    wrapper = _wrapper()
    assert payload == _independent_canonical(wrapper)
    assert not payload.endswith(b"\n")
    assert wrapper["profile_id"] == PROFILE_ID
    assert wrapper["profile_id"] == PROFILE_PREFIX + hashlib.sha256(
        PROFILE_DOMAIN + _independent_canonical(wrapper["profile_body"])
    ).hexdigest()
    assert hashlib.sha256(payload).hexdigest() == CONFIG_SHA256
    result = profile.verify_phase2b_actual_unsealed_960_generator_construction_profile_v1(
        payload
    )
    assert result.profile_id == PROFILE_ID
    assert result.canonical_profile_sha256 == CONFIG_SHA256
    assert result.disposition.value == "PROFILE_VERIFIED_NOT_GENERATOR_NOT_GENERATED"
    assert profile.PROFILE_ID_V1 == PROFILE_ID
    assert profile.PROFILE_CANONICAL_SHA256_V1 == CONFIG_SHA256
    with pytest.raises(TypeError):
        result.profile["status"] = "GENERATED"  # type: ignore[index]


def test_final_wire_dependency_pins_are_literal_complete_and_claim_bounded() -> None:
    wire = _wrapper()["profile_body"]["wire_binding"]
    assert wire["binding_status"] == (
        "FINAL_WIRE_DEPENDENCY_PINS_FRESHLY_VERIFIED"
    )
    assert wire["dependency_disposition"] == "FINAL_STATIC_GO_REPORTED_P0_0_P1_0"
    assert wire["wire_stop_executed_by_profile_verifier"] is False
    assert wire["formal_trusted_wire_schema_version"] == (
        "hegel-machine-phase2b-formal-trusted-wire/3"
    )
    assert wire["graph_completeness_policy_id"] == (
        "phase2b_formal_case_graph_completeness_policy_v3_"
        "7585a9f9e58823d766906296905efcd24b1e350330ca5c3f64d75361e9ad90e5"
    )
    assert wire["field_classification_manifest_id"] == (
        "phase2b_formal_wire_field_manifest_v3_"
        "12916eefe10770f54c3d6c71f4b8b2c14ebbd671642e96ff4082dd298e5088dc"
    )
    assert wire["wire_source_sha256"] == (
        "c8120920f0e31b959043f63a0f672471fcbea50d05ca2cd8eaf7fd0c29089765"
    )
    assert wire["wire_test_sha256"] == (
        "253001d11d7603456b80cf43837ed71de567bfd2cd6613cc78b21ca23f7db0f0"
    )
    assert wire["wire_config_sha256"] == (
        "4f82ab060e2d53912cd17c8966a681a53d11e96ce3c6bb02ea3a42c5d2e85784"
    )
    assert wire["wire_doc_sha256"] == (
        "704a937967441ebbb9b2a77a865f83890e241d37031e2b0a87f1da2877b3456a"
    )
    assert wire["wire_test_stop"] == {
        "core_and_rfc_tests_passed": 45,
        "wire_tests_include_rich_1024_and_1025_atomic_reject": True,
        "wire_tests_passed": 73,
    }
    for path, field in (
        (WIRE_SOURCE_PATH, "wire_source_sha256"),
        (WIRE_TEST_PATH, "wire_test_sha256"),
        (WIRE_CONFIG_PATH, "wire_config_sha256"),
        (WIRE_DOC_PATH, "wire_doc_sha256"),
        (CORE_SOURCE_PATH, "exact_transform_core_source_sha256"),
        (CORE_TEST_PATH, "exact_transform_core_test_sha256"),
        (RFC8785_SOURCE_PATH, "rfc8785_source_sha256"),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == wire[field]
    assert tuple(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in (WIRE_SOURCE_PATH, WIRE_TEST_PATH, WIRE_CONFIG_PATH, WIRE_DOC_PATH)
    ) == (6941, 4833, 262, 405)
    wire_config = json.loads(WIRE_CONFIG_PATH.read_bytes())
    assert wire_config["closed_public_schema"]["graph_completeness_policy_id"] == wire[
        "graph_completeness_policy_id"
    ]
    assert wire_config["closed_public_schema"]["field_classification_manifest_id"] == wire[
        "field_classification_manifest_id"
    ]


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"\xef\xbb\xbf{}",
        b"{}\n",
        b'{"profile_body":{},"profile_body":{},"profile_id":"x"}',
        b'{"profile_body":{"x":NaN},"profile_id":"x"}',
        b'{"profile_body":{"x":Infinity},"profile_id":"x"}',
        b'{"profile_body":{"x":1.0},"profile_id":"x"}',
        b"\xff",
        (b'{"x":' * 66) + b"0" + (b"}" * 66),
    ),
)
def test_strict_json_and_canonical_transport_attacks_reject(payload: bytes) -> None:
    with pytest.raises(profile.GeneratorConstructionProfileErrorV1):
        profile.verify_phase2b_actual_unsealed_960_generator_construction_profile_v1(
            payload
        )


def test_unknown_bool_as_int_quota_matrix_and_reorder_attacks_reject() -> None:
    attacks: list[dict[str, object]] = []

    unknown = _wrapper()
    unknown["profile_body"]["unknown"] = 0
    attacks.append(unknown)

    bool_as_int = _wrapper()
    bool_as_int["profile_body"]["corpus_contract"]["main_case_count"] = True
    attacks.append(bool_as_int)

    quota = _wrapper()
    quota["profile_body"]["corpus_contract"]["main_case_count"] = 719
    attacks.append(quota)

    matrix = _wrapper()
    matrix["profile_body"]["corpus_contract"]["case_margin_rows"][0][
        "clear_interior"
    ] = 10
    attacks.append(matrix)

    reordered = _wrapper()
    rows = reordered["profile_body"]["preservation_manifest"]["transform_rows"]
    rows[0], rows[1] = rows[1], rows[0]
    attacks.append(reordered)

    padding = _wrapper()
    padding["profile_body"]["semantic_descriptor"]["padding_codepoint"] = "U+00A0"
    attacks.append(padding)

    for attack in attacks:
        with pytest.raises(profile.GeneratorConstructionProfileErrorV1):
            profile.verify_phase2b_actual_unsealed_960_generator_construction_profile_v1(
                _reidentified_bytes(attack)
            )


def test_wire_binding_and_claim_promotion_attacks_reject() -> None:
    wire = _wrapper()
    wire["profile_body"]["wire_binding"]["maximum_batch_cases"] = 1023
    claims = _wrapper()
    claims["profile_body"]["claim_boundary"]["false_claims"]["r06_satisfied"] = True
    for attack in (wire, claims):
        with pytest.raises(profile.GeneratorConstructionProfileErrorV1):
            profile.verify_phase2b_actual_unsealed_960_generator_construction_profile_v1(
                _reidentified_bytes(attack)
            )


def test_docs_preserve_non_generating_boundary() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "PROFILE_VERIFIED_NOT_GENERATOR_NOT_GENERATED" in text
    assert "FUTURE_REQUIRED_NOT_ISSUED" in text
    assert "does not satisfy R06" in text
    assert "Actual cohort materialization is permitted only at R14" in text
