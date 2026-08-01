from dataclasses import replace

import pytest

from hegel_machine.phase2b_freeze_v1 import (
    BOOTSTRAP_DERIVED_UINT32_SEED,
    BOOTSTRAP_MASTER_SEED,
    BOOTSTRAP_SEED,
    BOOTSTRAP_UINT32_DERIVATION_ID,
    CanonicalFamilyId,
    FormalUncertaintyKind,
    PHASE2B_EXACT_FREEZE_VERSION,
    PreservationTransform,
    canonical_family_id,
    derive_bootstrap_uint32_seed,
    frozen_phase2b_exact_freeze,
)
from hegel_machine.schema import LawKind


def test_exact_freeze_binds_canonical_family_ids_and_stays_nonformal():
    freeze = frozen_phase2b_exact_freeze()
    assert (
        freeze.freeze_version
        == PHASE2B_EXACT_FREEZE_VERSION
        == "hegel-freeze-p2b-p3-v1.0.1"
    )
    assert dict(freeze.family_mapping) == {
        LawKind.SYMMETRY: CanonicalFamilyId.F01,
        LawKind.MONOTONICITY: CanonicalFamilyId.F02,
        LawKind.CONSERVATION: CanonicalFamilyId.F03,
        LawKind.COMPLEMENTARITY: CanonicalFamilyId.F04,
        LawKind.LOCALITY: CanonicalFamilyId.F05,
        LawKind.NEGATIVE_FEEDBACK: CanonicalFamilyId.F06,
    }
    assert freeze.freeze_id.startswith("phase2b_exact_freeze_")
    assert freeze.formal_holdout_generation_authorized is False
    assert freeze.formal_holdout_generated is False
    assert freeze.formal_holdout_consumed is False
    assert freeze.shadow_only is True
    assert freeze.implementation_blockers
    assert canonical_family_id(LawKind.LOCALITY) is CanonicalFamilyId.F05
    with pytest.raises(TypeError, match="LawKind"):
        canonical_family_id("locality_markov")


def test_720_table_is_exactly_19_plus_1_and_21_18_12_9_per_cell():
    freeze = frozen_phase2b_exact_freeze()
    assert freeze.holdout.cell_count == 12
    assert freeze.holdout.independent_latent_case_count == 720
    assert dict(freeze.holdout.case_quota_per_cell) == {
        "unique_scale_answerable": 19,
        "admissible_scale_set_answerable": 1,
        "wrong_family_hard_negative": 8,
        "binding_counterfactual": 8,
        "scale_counterfactual": 8,
        "sign_or_invariant_break": 8,
        "insufficient_or_nonidentifiable": 8,
    }
    assert dict(freeze.holdout.margin_quota_per_cell) == {
        "clear_interior": 21,
        "moderate": 18,
        "near_boundary_identifiable": 12,
        "nonunique_or_insufficient": 9,
    }
    assert freeze.holdout.margin_case_joint_quota_per_cell == (
        (
            "nonunique_or_insufficient",
            "insufficient_or_nonidentifiable",
            8,
        ),
        (
            "nonunique_or_insufficient",
            "admissible_scale_set_answerable",
            1,
        ),
    )
    assert dict(freeze.case_type_totals) == {
        "unique_scale_answerable": 228,
        "admissible_scale_set_answerable": 12,
        "wrong_family_hard_negative": 96,
        "binding_counterfactual": 96,
        "scale_counterfactual": 96,
        "sign_or_invariant_break": 96,
        "insufficient_or_nonidentifiable": 96,
    }
    assert dict(freeze.margin_stratum_totals) == {
        "clear_interior": 252,
        "moderate": 216,
        "near_boundary_identifiable": 144,
        "nonunique_or_insufficient": 108,
    }


def test_metric_denominators_include_set_valued_answers_only_where_frozen():
    holdout = frozen_phase2b_exact_freeze().holdout
    metrics = {item.metric: item for item in holdout.metric_denominators}
    assert metrics["answerable_count"].expected_count == 240
    assert metrics["family_exact_accuracy"].expected_count == 240
    assert metrics["binding_exact_accuracy"].expected_count == 240
    assert metrics["scale_set_accuracy"].expected_count == 240
    assert metrics["joint_exact_accuracy"].expected_count == 240
    assert metrics["unique_scale_accuracy"].expected_count == 228
    assert metrics["abstention_specificity"].expected_count == 228
    assert metrics["nonidentifiability_abstention_accuracy"].expected_count == 96
    assert metrics["set_valued_answer_accuracy"].expected_count == 12
    assert metrics["set_valued_answer_accuracy"].separately_reported is True
    assert "admissible_scale_set_answerable" in metrics[
        "joint_exact_accuracy"
    ].included_case_types
    assert "admissible_scale_set_answerable" not in metrics[
        "unique_scale_accuracy"
    ].included_case_types
    assert holdout.set_valued_joint_rule == (
        "family_exact_and_binding_exact_and_scale_set_exact_and_ANSWER_SET"
    )


def test_preservation_matrix_derives_496_plus_76_without_entering_720():
    freeze = frozen_phase2b_exact_freeze()
    legal = {
        rule.transform: rule.legal_pair_count for rule in freeze.preservation_rules
    }
    invalid = {
        rule.transform: rule.invalid_control_count
        for rule in freeze.preservation_rules
    }
    assert legal == {
        PreservationTransform.ENTITY_ALPHA_RENAMING: 72,
        PreservationTransform.OBSERVATION_REORDER: 48,
        PreservationTransform.IRRELEVANT_ENTITY_AUGMENTATION: 72,
        PreservationTransform.UNIT_CONVERSION: 96,
        PreservationTransform.COORDINATE_AFFINE_TRANSFORM: 64,
        PreservationTransform.EQUIVALENT_AGGREGATION_SPLIT_MERGE: 48,
        PreservationTransform.NONTRIVIAL_SCALE_MAP: 60,
        PreservationTransform.SIGN_CONVENTION_REPARAMETERIZATION: 36,
    }
    assert sum(invalid.values()) == 76
    assert freeze.legal_preservation_pair_count == 496
    assert freeze.invalid_transform_control_count == 76
    assert freeze.total_preservation_sensitivity_pair_count == 572
    assert freeze.holdout.independent_latent_case_count == 720


def test_every_preservation_transform_pair_quota_is_fail_closed():
    freeze = frozen_phase2b_exact_freeze()
    for index, rule in enumerate(freeze.preservation_rules):
        field = (
            "legal_pairs_per_family_scale"
            if rule.legal_pairs_per_family_scale
            else "legal_pairs_per_family"
        )
        changed_rules = list(freeze.preservation_rules)
        changed_rules[index] = replace(
            rule,
            **{field: getattr(rule, field) + 1},
        )
        with pytest.raises(ValueError):
            replace(freeze, preservation_rules=tuple(changed_rules))


def test_baselines_bootstrap_challenge_and_footprint_are_exactly_frozen():
    freeze = frozen_phase2b_exact_freeze()
    baselines = {item.baseline_id: item for item in freeze.baselines}
    assert baselines["embedding_nearest_prototype"].implementation == (
        "sentence-transformers/all-mpnet-base-v2"
    )
    assert baselines["embedding_nearest_prototype"].revision_policy == (
        "exact_40_hex_commit_required"
    )
    assert baselines["frozen_llm_semantic_only"].implementation == (
        "Qwen/Qwen2.5-7B-Instruct"
    )
    assert "Do not execute equations" in baselines[
        "frozen_llm_semantic_only"
    ].prompt
    assert baselines["flat_learned_typed"].output_heads == (
        "family",
        "binding",
        "scale_set_class",
        "answer_vs_abstain",
    )
    assert freeze.bootstrap.replicates == 10_000
    assert BOOTSTRAP_MASTER_SEED == BOOTSTRAP_SEED == 411876909552964556
    assert freeze.bootstrap.seed == BOOTSTRAP_MASTER_SEED
    assert (
        freeze.bootstrap.seed_derivation_id
        == BOOTSTRAP_UINT32_DERIVATION_ID
        == "sha256_domain_separated_uint64_be_first32_v1"
    )
    assert freeze.bootstrap.derived_uint32_seed == BOOTSTRAP_DERIVED_UINT32_SEED
    assert BOOTSTRAP_DERIVED_UINT32_SEED == 2611585425
    assert BOOTSTRAP_DERIVED_UINT32_SEED == derive_bootstrap_uint32_seed(
        BOOTSTRAP_MASTER_SEED
    )
    assert 0 <= BOOTSTRAP_DERIVED_UINT32_SEED < 2**32
    with pytest.raises(TypeError, match="integer"):
        derive_bootstrap_uint32_seed(True)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        derive_bootstrap_uint32_seed(1 << 64)
    assert freeze.bootstrap.resampling_unit == "latent_base_case"
    assert freeze.semantic_conflict.case_count == 240
    assert freeze.semantic_conflict.included_in_main_accuracy_denominator is False
    assert freeze.semantic_conflict.threshold_tuning_allowed is False
    assert freeze.footprint_audit.minimum_classes_per_family_scale_cell == 3
    assert freeze.footprint_audit.grouped_permutation_replicates == 1_000
    assert freeze.footprint_audit.maximum_single_group_share == 0.50
    assert (
        freeze.footprint_audit.maximum_best_single_measurement_balanced_accuracy
        == 0.50
    )


def test_flat_baseline_really_initializes_with_the_frozen_uint32_seed():
    from sklearn.ensemble import HistGradientBoostingClassifier

    baseline = next(
        item
        for item in frozen_phase2b_exact_freeze().baselines
        if item.baseline_id == "flat_learned_typed"
    )
    parameters = dict(baseline.parameters)
    assert parameters.pop("holdout_adjustment_allowed") is False
    classifier = HistGradientBoostingClassifier(**parameters)
    features = [
        [float(index % 5), float((index // 5) % 5)] for index in range(40)
    ]
    labels = [index % 2 for index in range(40)]
    classifier.fit(features, labels)
    assert classifier.get_params()["random_state"] == BOOTSTRAP_DERIVED_UINT32_SEED


def test_phase2b_exact_freeze_rejects_compensating_and_nested_field_drift():
    freeze = frozen_phase2b_exact_freeze()

    preservation = list(freeze.preservation_rules)
    preservation[0] = replace(
        preservation[0],
        legal_pairs_per_family_scale=5,
    )
    preservation[2] = replace(
        preservation[2],
        legal_pairs_per_family_scale=7,
    )
    with pytest.raises(ValueError, match="preservation_rules"):
        replace(freeze, preservation_rules=tuple(preservation))

    with pytest.raises(ValueError, match="semantic_conflict"):
        replace(
            freeze,
            semantic_conflict=replace(
                freeze.semantic_conflict,
                low_overlap_structural_positives_per_cell=9,
                high_overlap_structural_negatives_per_cell=11,
            ),
        )

    changed_baseline = replace(
        freeze.baselines[0],
        parameters=freeze.baselines[0].parameters
        + (("unexpected_parameter", True),),
    )
    with pytest.raises(ValueError, match="baselines"):
        replace(freeze, baselines=(changed_baseline, *freeze.baselines[1:]))

    with pytest.raises(ValueError, match="bootstrap"):
        replace(
            freeze,
            bootstrap=replace(
                freeze.bootstrap,
                cluster_members=("original_case", "all_preservation_variants"),
            ),
        )

    with pytest.raises(ValueError, match="rerun_policy"):
        replace(
            freeze,
            rerun_policy=replace(
                freeze.rerun_policy,
                allowed_reexecution_reasons=(
                    *freeze.rerun_policy.allowed_reexecution_reasons,
                    "UNFROZEN_REASON",
                ),
            ),
        )

    with pytest.raises(ValueError, match="covert_channel_audit"):
        replace(
            freeze,
            covert_channel_audit=replace(
                freeze.covert_channel_audit,
                channel_targets=(
                    *freeze.covert_channel_audit.channel_targets[:-1],
                    "changed_joint_target",
                ),
            ),
        )

    with pytest.raises(ValueError, match="implementation_blockers"):
        replace(freeze, implementation_blockers=("unfrozen_blocker",))


def test_margin_case_joint_row_rejects_a_different_nine_case_split():
    freeze = frozen_phase2b_exact_freeze()
    with pytest.raises(ValueError, match=r"exactly 8\+1"):
        replace(
            freeze.holdout,
            margin_case_joint_quota_per_cell=(
                (
                    "nonunique_or_insufficient",
                    "insufficient_or_nonidentifiable",
                    7,
                ),
                (
                    "nonunique_or_insufficient",
                    "admissible_scale_set_answerable",
                    2,
                ),
            ),
        )


def test_rerun_and_validation_versions_fail_closed():
    freeze = frozen_phase2b_exact_freeze()
    rerun = freeze.rerun_policy
    assert rerun.maximum_reexecutions == 2
    assert rerun.permits_reexecution(
        "CONTAINER_START_FAILURE",
        any_valid_prediction_byte_produced=False,
    )
    assert not rerun.permits_reexecution(
        "CONTAINER_START_FAILURE",
        any_valid_prediction_byte_produced=True,
    )
    assert rerun.retry_action(
        "CONTAINER_START_FAILURE",
        any_valid_prediction_byte_produced=False,
    ) == "REEXECUTE"
    assert rerun.retry_action(
        rerun.upload_only_retry_reason,
        any_valid_prediction_byte_produced=True,
    ) == "REUPLOAD_COMMITTED_OUTPUT"
    assert rerun.retry_action(
        "MODEL_EXCEPTION",
        any_valid_prediction_byte_produced=False,
    ) == "FORBIDDEN"
    assert not rerun.permits_reexecution(
        rerun.upload_only_retry_reason,
        any_valid_prediction_byte_produced=False,
    )
    assert not rerun.permits_reexecution(
        "MODEL_EXCEPTION",
        any_valid_prediction_byte_produced=False,
    )
    with pytest.raises(TypeError, match="boolean"):
        rerun.permits_reexecution(
            "CONTAINER_START_FAILURE",
            any_valid_prediction_byte_produced=1,
        )
    validation = freeze.validation_policy
    assert validation.attempts_per_version == 2
    assert validation.maximum_validation_versions_before_no_go == 2
    assert validation.version_fields_required_after_change == (
        "protocol_version",
        "selector_version",
        "validation_version",
    )
    assert validation.sealed_holdout_only_after_validation_pass is True


def test_covert_channel_contract_freezes_keys_tests_and_invariances():
    audit = frozen_phase2b_exact_freeze().covert_channel_audit
    assert audit.independent_secret_keys == ("K_shuffle", "K_id", "K_padding")
    assert audit.keys_separate_from == ("latent_generator_seed", "answer_seed")
    assert audit.id_assignment_after_global_shuffle is True
    assert audit.id_bits == 128
    assert audit.collision_max_retries == 10
    assert audit.label_permutations == 10_000
    assert audit.multiple_testing == "Holm-Bonferroni"
    assert audit.family_wise_alpha == 0.01
    assert audit.maximum_normalized_mutual_information == 0.02
    assert audit.maximum_balanced_accuracy_advantage == 0.05
    assert audit.global_consistent_renamings == 32
    assert audit.global_case_order_permutations == 32
    assert audit.within_case_observation_order_permutations == 16
    assert audit.envelope_bytes == 65_536
    assert len(audit.renaming_namespaces) == 10
    assert audit.per_case_unused_transform_list_allowed is False


def test_formal_selector_is_absolute_bound_only_until_semantics_exist():
    policy = frozen_phase2b_exact_freeze().uncertainty_policy
    assert policy.allowed_kinds == (FormalUncertaintyKind.ABSOLUTE_BOUND,)
    assert policy.standard_error_status == "STANDARD_ERROR_UNSUPPORTED"
    assert policy.simultaneous_interval_family_wise_coverage == 0.99
    assert policy.endpoint_rounding == "outward_to_frozen_RationalValue_grid"
    with pytest.raises(ValueError, match="absolute-bound only"):
        replace(
            policy,
            allowed_kinds=(
                FormalUncertaintyKind.ABSOLUTE_BOUND,
                FormalUncertaintyKind.STANDARD_ERROR,
            ),
        )
