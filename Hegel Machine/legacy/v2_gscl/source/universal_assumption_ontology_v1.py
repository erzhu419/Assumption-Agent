"""Frozen Universal Assumption Ontology v1.

The catalog is an experimental search-space type system, not a list of truths.
It separates ontology parents from epistemic roles and keeps all diagnostics
TRAIN-only.  A template can only reach the existing runtime after a
problem-specific claim has been probed and compiled.
"""

from __future__ import annotations

from .meta_assumption import (
    AssumptionRole,
    CompilerTarget,
    DiagnosticProbePlan,
    LegacyAssumptionAlias,
    MetaAssumptionTemplate,
    OntologyRoot,
    UniversalAssumptionOntology,
)


VERSION = "universal_assumption_ontology_v1"

R1 = "uao.v1.root.compression_sufficiency"
R2 = "uao.v1.root.decomposition_reuse"
R3 = "uao.v1.root.geometry_dynamics"
R4 = "uao.v1.root.invariance_constraints"
R5 = "uao.v1.root.exceptions_uncertainty"
R6 = "uao.v1.root.epistemic_governance"

T01 = "uao.v1.t01_mdl_compressibility"
T02 = "uao.v1.t02_sparsity"
T03 = "uao.v1.t03_low_rank_separability"
T04 = "uao.v1.t04_minimal_sufficient_representation"
T05 = "uao.v1.t05_low_order_interaction"
T06 = "uao.v1.t06_nonredundant_predictions"
T07 = "uao.v1.t07_modular_independent_mechanisms"
T08 = "uao.v1.t08_locality_markov_blanket"
T09 = "uao.v1.t09_compositional_reuse"
T10 = "uao.v1.t10_low_frequency_smoothness"
T11 = "uao.v1.t11_piecewise_stability"
T12 = "uao.v1.t12_scale_separation"
T13 = "uao.v1.t13_stability_contraction"
T14 = "uao.v1.t14_symmetry_equivariance"
T15 = "uao.v1.t15_conservation_balance"
T16 = "uao.v1.t16_topological_persistence"
T17 = "uao.v1.t17_order_shape_constraint"
T18 = "uao.v1.t18_sparse_contamination"
T19 = "uao.v1.t19_minimum_commitment"
T20 = "uao.v1.t20_falsifiability"
T21 = "uao.v1.t21_evidence_triangulation"
T22 = "uao.v1.t22_decision_relevance"


def _probe(
    ordinal: int,
    name: str,
    observables: tuple[str, ...],
    support_rule: str,
    counter_rule: str,
) -> DiagnosticProbePlan:
    return DiagnosticProbePlan(
        probe_id=f"uao.v1.probe.t{ordinal:02d}.{name}",
        observable_ids=observables,
        support_rule_id=support_rule,
        counter_rule_id=counter_rule,
        max_evaluations=1,
        train_only=True,
    )


def _template(
    *,
    template_id: str,
    primary_parent_id: str,
    parent_ids: tuple[str, ...],
    roles: tuple[AssumptionRole, ...],
    claim_schema: str,
    admissible_variable_types: tuple[str, ...],
    support_signatures: tuple[str, ...],
    counter_signatures: tuple[str, ...],
    probe_plan: DiagnosticProbePlan,
    compiler_targets: tuple[CompilerTarget, ...],
    not_applicable_conditions: tuple[str, ...],
    invariances: tuple[str, ...] = (),
) -> MetaAssumptionTemplate:
    return MetaAssumptionTemplate(
        template_id=template_id,
        primary_parent_id=primary_parent_id,
        parent_ids=parent_ids,
        roles=roles,
        claim_schema=claim_schema,
        admissible_variable_types=admissible_variable_types,
        support_signatures=support_signatures,
        counter_signatures=counter_signatures,
        probe_plan=probe_plan,
        compiler_targets=compiler_targets,
        not_applicable_conditions=not_applicable_conditions,
        invariances=invariances,
    )


def _roots() -> tuple[OntologyRoot, ...]:
    return (
        OntologyRoot(
            root_id=R1,
            title="Compression and sufficient representation",
            description="Short, sparse, low-dimensional, decision-sufficient descriptions.",
        ),
        OntologyRoot(
            root_id=R2,
            title="Decomposition and mechanism reuse",
            description="Low-order, local, modular, distinguishable, reusable mechanisms.",
        ),
        OntologyRoot(
            root_id=R3,
            title="Geometry and dynamics",
            description="Smooth, piecewise, multi-scale, or stable behavior in a declared geometry.",
        ),
        OntologyRoot(
            root_id=R4,
            title="Invariance and physical structure",
            description="Certified transformations, balances, topology, and shape restrictions.",
        ),
        OntologyRoot(
            root_id=R5,
            title="Exceptions and uncertainty",
            description="Sparse exceptions and principled preservation of unresolved uncertainty.",
        ),
        OntologyRoot(
            root_id=R6,
            title="Epistemic and decision governance",
            description="Falsification, independent evidence, and action-relevant claim management.",
        ),
    )


def _templates() -> tuple[MetaAssumptionTemplate, ...]:
    W = AssumptionRole.WORLD_CLAIM
    P = AssumptionRole.REPRESENTATION_PRIOR
    R = AssumptionRole.REGULARIZER
    G = AssumptionRole.GOVERNANCE_RULE
    D = AssumptionRole.DECISION_RULE
    TP = CompilerTarget.TASK_PROGRAM
    PP = CompilerTarget.POLICY_PROGRAM
    EV = CompilerTarget.EVALUATOR_ARTIFACT
    IC = CompilerTarget.IMPLEMENTATION_CONTRACT
    ND = CompilerTarget.NO_DIRECT_TREATMENT

    return (
        _template(
            template_id=T01,
            primary_parent_id=R1,
            parent_ids=(R1, R6),
            roles=(R,),
            claim_schema=(
                "Among claims with equivalent TRAIN predictive or decision utility, "
                "the canonically shorter claim transfers at least as well."
            ),
            admissible_variable_types=("canonical_program", "loss", "decision_utility"),
            support_signatures=(
                "shorter_description_with_equivalent_crossfit_utility",
                "complexity_order_stable_across_train_folds",
            ),
            counter_signatures=(
                "shorter_description_has_material_crossfit_harm",
                "description_order_changes_under_semantics_preserving_serialization",
            ),
            probe_plan=_probe(
                1,
                "paired_mdl_crossfit",
                ("canonical_code_length", "crossfit_loss", "crossfit_decision_utility"),
                "prefer_shorter_only_with_equivalent_train_crossfit_utility",
                "reject_if_shorter_is_harmful_or_encoding_unstable",
            ),
            compiler_targets=(ND,),
            not_applicable_conditions=(
                "canonical_encoding_is_undefined",
                "fewer_than_two_competing_claims",
                "train_only_utility_is_unavailable",
            ),
            invariances=("canonical_semantics_preserving_serialization",),
        ),
        _template(
            template_id=T02,
            primary_parent_id=R1,
            parent_ids=(R1, R2),
            roles=(W, R),
            claim_schema=(
                "A stable minority of candidate factors or actions explains most "
                "TRAIN incremental utility; the remainder should be inactive."
            ),
            admissible_variable_types=("candidate_factor_set", "typed_action_set", "incremental_utility"),
            support_signatures=(
                "small_active_set_captures_most_incremental_utility",
                "active_set_is_stable_across_train_folds",
            ),
            counter_signatures=(
                "incremental_utility_is_diffuse",
                "selected_active_set_is_fold_unstable",
            ),
            probe_plan=_probe(
                2,
                "stability_selection",
                ("active_set_size", "captured_utility_fraction", "fold_selection_frequency"),
                "support_if_small_stable_set_captures_train_utility",
                "counter_if_contributions_are_diffuse_or_unstable",
            ),
            compiler_targets=(PP, EV),
            not_applicable_conditions=(
                "candidate_factors_are_not_decomposable",
                "no_inactive_alternative_can_be_defined",
            ),
            invariances=("candidate_identifier_renaming",),
        ),
        _template(
            template_id=T03,
            primary_parent_id=R1,
            parent_ids=(R1, R2),
            roles=(W, P),
            claim_schema=(
                "Observed interactions are generated by a fixed small number of "
                "latent factors and admit held-fold low-rank reconstruction."
            ),
            admissible_variable_types=("matrix", "tensor", "bilinear_interaction"),
            support_signatures=(
                "rapid_spectral_decay",
                "fixed_rank_reconstruction_transfers_across_train_folds",
            ),
            counter_signatures=("no_spectral_gap", "fixed_rank_reconstruction_is_unstable_or_harmful"),
            probe_plan=_probe(
                3,
                "heldfold_low_rank_reconstruction",
                ("singular_spectrum", "heldfold_reconstruction_error", "rank_stability"),
                "support_if_frozen_low_rank_reconstructs_held_train_fold",
                "counter_if_spectrum_or_reconstruction_rejects_low_rank",
            ),
            compiler_targets=(EV,),
            not_applicable_conditions=(
                "no_matrix_tensor_or_bilinear_observable",
                "rank_is_not_identifiable_from_train",
            ),
            invariances=("latent_factor_orthogonal_rotation",),
        ),
        _template(
            template_id=T04,
            primary_parent_id=R1,
            parent_ids=(R1, R4, R6),
            roles=(P,),
            claim_schema=(
                "A declared quotient representation preserves TRAIN decision "
                "information while raw surface variables add no stable increment."
            ),
            admissible_variable_types=("raw_state", "typed_representation", "decision_target"),
            support_signatures=(
                "raw_features_add_no_conditional_crossfit_gain",
                "quotient_representation_preserves_action_value",
            ),
            counter_signatures=(
                "raw_features_add_stable_conditional_gain",
                "quotient_merges_decision_distinct_states",
            ),
            probe_plan=_probe(
                4,
                "conditional_predictive_sufficiency",
                ("representation_score", "raw_incremental_score", "action_value_preservation"),
                "support_if_raw_increment_is_absent_and_action_value_is_preserved",
                "counter_if_raw_increment_or_decision_aliasing_is_present",
            ),
            compiler_targets=(TP, PP, EV),
            not_applicable_conditions=(
                "candidate_representation_is_undefined",
                "train_decision_target_is_unavailable",
            ),
            invariances=("declared_quotient_equivalence",),
        ),
        _template(
            template_id=T05,
            primary_parent_id=R2,
            parent_ids=(R2,),
            roles=(W,),
            claim_schema=(
                "Joint utility is adequately represented by frozen unary and "
                "low-order interactions rather than unrestricted high-order terms."
            ),
            admissible_variable_types=("component_set", "set_utility", "interaction_term"),
            support_signatures=(
                "unary_pair_terms_explain_train_utility",
                "low_order_effects_transfer_across_train_folds",
            ),
            counter_signatures=(
                "irreducible_higher_order_synergy",
                "low_order_effects_fail_held_combination_test",
            ),
            probe_plan=_probe(
                5,
                "functional_anova_interaction",
                ("unary_effect", "pair_effect", "higher_order_residual", "held_combination_error"),
                "support_if_low_order_terms_explain_held_train_combinations",
                "counter_if_irreducible_high_order_residual_remains",
            ),
            compiler_targets=(PP, EV),
            not_applicable_conditions=(
                "fewer_than_two_composable_components",
                "joint_action_or_set_utility_is_undefined",
            ),
            invariances=("component_identifier_renaming",),
        ),
        _template(
            template_id=T06,
            primary_parent_id=R2,
            parent_ids=(R2, R6),
            roles=(R, G),
            claim_schema=(
                "Retained claims explain residual increments and make predictions "
                "that are distinguishable on at least one admissible TRAIN probe."
            ),
            admissible_variable_types=("claim_set", "prediction_signature", "residual_vector"),
            support_signatures=(
                "positive_residualized_incremental_effect",
                "prediction_disagreement_on_admissible_probe",
            ),
            counter_signatures=(
                "prediction_signatures_are_observationally_equivalent",
                "no_residualized_incremental_effect",
            ),
            probe_plan=_probe(
                6,
                "prediction_distinctness",
                ("residualized_effect", "prediction_signature", "probe_disagreement"),
                "support_if_claim_adds_residual_effect_and_distinct_prediction",
                "counter_if_claim_is_redundant_on_all_admissible_probes",
            ),
            compiler_targets=(ND,),
            not_applicable_conditions=(
                "fewer_than_two_competing_claims",
                "no_common_admissible_probe_space",
            ),
            invariances=("claim_identifier_renaming",),
        ),
        _template(
            template_id=T07,
            primary_parent_id=R2,
            parent_ids=(R2,),
            roles=(W,),
            claim_schema=(
                "A declared mechanism can be intervened on locally and reused "
                "without changing unrelated module outputs."
            ),
            admissible_variable_types=("module_graph", "typed_intervention", "module_output"),
            support_signatures=(
                "intervention_effect_is_local_to_declared_module",
                "module_effect_transfers_across_train_contexts",
            ),
            counter_signatures=(
                "intervention_causes_unbounded_cross_module_change",
                "module_effect_is_context_specific",
            ),
            probe_plan=_probe(
                7,
                "intervention_locality",
                ("local_output_delta", "nonlocal_output_delta", "held_context_module_effect"),
                "support_if_intervention_is_local_and_reusable",
                "counter_if_effect_is_globally_coupled_or_nonportable",
            ),
            compiler_targets=(PP, EV),
            not_applicable_conditions=(
                "no_declared_module_boundary",
                "no_typed_intervention_is_available",
            ),
            invariances=("module_identifier_renaming",),
        ),
        _template(
            template_id=T08,
            primary_parent_id=R2,
            parent_ids=(R2, R3),
            roles=(W, P),
            claim_schema=(
                "Influence is conditionally concentrated in a declared graph, "
                "temporal, spatial, or dependency neighborhood."
            ),
            admissible_variable_types=("adjacency_graph", "distance", "ordered_context", "conditional_variable_set"),
            support_signatures=(
                "effect_decays_with_declared_distance",
                "declared_neighborhood_is_conditionally_sufficient",
            ),
            counter_signatures=(
                "distant_variables_have_equal_or_greater_stable_effect",
                "declared_neighborhood_omits_stable_predictive_parent",
            ),
            probe_plan=_probe(
                8,
                "distance_conditioned_ablation",
                ("distance_bin", "ablated_effect", "conditional_increment"),
                "support_if_neighborhood_contains_stable_effect",
                "counter_if_remote_effect_or_omitted_parent_remains",
            ),
            compiler_targets=(PP, EV),
            not_applicable_conditions=(
                "locality_variant_is_unbound",
                "no_declared_graph_order_distance_or_conditioning_set",
            ),
            invariances=("distance_preserving_relabeling",),
        ),
        _template(
            template_id=T09,
            primary_parent_id=R2,
            parent_ids=(R2,),
            roles=(W,),
            claim_schema=(
                "Typed modules learned on observed combinations can be recombined "
                "through frozen interfaces on held TRAIN combinations."
            ),
            admissible_variable_types=("typed_module", "composition_dag", "module_interface"),
            support_signatures=(
                "held_combination_transfer",
                "module_interface_is_context_invariant",
            ),
            counter_signatures=(
                "memorized_combination_only",
                "unmodeled_cross_interface_interaction",
            ),
            probe_plan=_probe(
                9,
                "held_combination",
                ("seen_combination_score", "held_combination_score", "interface_violation"),
                "support_if_modules_transfer_to_held_combinations",
                "counter_if_recombination_fails_or_interface_drifts",
            ),
            compiler_targets=(TP, PP, EV),
            not_applicable_conditions=(
                "no_repeated_module_types",
                "held_combinations_cannot_be_constructed_train_only",
            ),
            invariances=("module_instance_renaming",),
        ),
        _template(
            template_id=T10,
            primary_parent_id=R3,
            parent_ids=(R3,),
            roles=(W, P, R),
            claim_schema=(
                "Response varies smoothly in a declared effective metric and "
                "concentrates energy in low graph or spectral modes."
            ),
            admissible_variable_types=("metric_space", "graph_signal", "response"),
            support_signatures=(
                "low_mode_energy_concentration",
                "bounded_neighbor_response_difference",
            ),
            counter_signatures=(
                "unexplained_local_sign_flips",
                "high_frequency_energy_is_decision_relevant",
            ),
            probe_plan=_probe(
                10,
                "spectral_smoothness",
                ("low_mode_energy_fraction", "neighbor_response_delta", "local_sign_flip_rate"),
                "support_if_response_is_low_frequency_in_declared_geometry",
                "counter_if_local_discontinuity_or_high_frequency_signal_dominates",
            ),
            compiler_targets=(EV,),
            not_applicable_conditions=(
                "no_semantically_valid_metric_or_graph",
                "target_is_declared_discontinuous",
            ),
            invariances=("isometry_of_declared_metric",),
        ),
        _template(
            template_id=T11,
            primary_parent_id=R3,
            parent_ids=(R3, R5),
            roles=(W,),
            claim_schema=(
                "An ordered process is stable within a small fixed number of "
                "segments and changes at sparse boundaries."
            ),
            admissible_variable_types=("ordered_index", "regime_label", "segment_response"),
            support_signatures=(
                "within_segment_stability",
                "small_stable_change_point_set",
            ),
            counter_signatures=(
                "continuous_drift",
                "change_points_are_dense_or_fold_unstable",
            ),
            probe_plan=_probe(
                11,
                "piecewise_change_point",
                ("change_point_count", "within_segment_error", "fold_boundary_stability"),
                "support_if_sparse_boundaries_yield_stable_segments",
                "counter_if_drift_or_unstable_dense_boundaries_remain",
            ),
            compiler_targets=(PP, EV),
            not_applicable_conditions=(
                "no_declared_order_axis",
                "insufficient_train_observations_per_candidate_segment",
            ),
            invariances=("order_preserving_reindexing",),
        ),
        _template(
            template_id=T12,
            primary_parent_id=R3,
            parent_ids=(R3, R5),
            roles=(W, P),
            claim_schema=(
                "Slow structural variables remain predictive across aggregation "
                "while fast fluctuations average into a residual component."
            ),
            admissible_variable_types=("multi_scale_index", "slow_state", "fast_residual"),
            support_signatures=(
                "stable_timescale_gap",
                "fast_component_averages_without_decision_loss",
            ),
            counter_signatures=(
                "no_timescale_gap",
                "fast_component_changes_decision_utility",
            ),
            probe_plan=_probe(
                12,
                "scale_aggregation",
                ("timescale_spectrum", "aggregation_error", "fast_component_decision_increment"),
                "support_if_slow_structure_survives_aggregation",
                "counter_if_no_gap_or_fast_component_is_decision_relevant",
            ),
            compiler_targets=(TP, EV),
            not_applicable_conditions=(
                "no_declared_scale_or_time_axis",
                "no_repeated_fine_scale_measurement",
            ),
            invariances=("within_scale_sampling_refinement",),
        ),
        _template(
            template_id=T13,
            primary_parent_id=R3,
            parent_ids=(R3,),
            roles=(W, R),
            claim_schema=(
                "Admissible perturbations or local updates have bounded influence "
                "and do not amplify unrelated state differences."
            ),
            admissible_variable_types=("metric_state", "perturbation", "update_operator"),
            support_signatures=(
                "empirical_lipschitz_bound_is_stable",
                "local_update_has_bounded_nonlocal_spillover",
            ),
            counter_signatures=(
                "small_perturbation_is_amplified",
                "local_update_causes_unbounded_nonlocal_change",
            ),
            probe_plan=_probe(
                13,
                "paired_perturbation_stability",
                ("input_distance", "output_distance", "nonlocal_spillover"),
                "support_if_update_sensitivity_is_bounded",
                "counter_if_perturbations_are_amplified",
            ),
            compiler_targets=(EV, IC),
            not_applicable_conditions=(
                "no_declared_metric",
                "no_admissible_perturbation_or_update_operator",
            ),
            invariances=("metric_preserving_relabeling",),
        ),
        _template(
            template_id=T14,
            primary_parent_id=R4,
            parent_ids=(R4, R1),
            roles=(W, P, R),
            claim_schema=(
                "A certified semantics-preserving transformation leaves the "
                "decision invariant or transforms the output equivariantly."
            ),
            admissible_variable_types=("group_action", "input_object", "output_representation"),
            support_signatures=(
                "metamorphic_prediction_consistency",
                "equivariance_residual_within_declared_tolerance",
            ),
            counter_signatures=(
                "certified_transform_causes_systematic_effect",
                "equivariance_map_is_inconsistent",
            ),
            probe_plan=_probe(
                14,
                "metamorphic_transform",
                ("transformed_prediction_delta", "equivariance_residual"),
                "support_if_certified_transforms_preserve_or_equivariantly_map_output",
                "counter_if_transform_causes_systematic_violation",
            ),
            compiler_targets=(TP, PP, EV, IC),
            not_applicable_conditions=(
                "no_certified_semantics_preserving_transform",
                "group_or_equivariance_action_is_unbound",
            ),
            invariances=("transformation_composition",),
        ),
        _template(
            template_id=T15,
            primary_parent_id=R4,
            parent_ids=(R4,),
            roles=(W, R),
            claim_schema=(
                "A declared quantity satisfies closed accounting after explicit "
                "sources, sinks, and boundary flows are included."
            ),
            admissible_variable_types=("conserved_quantity", "transition", "boundary_flow"),
            support_signatures=(
                "balance_residual_matches_measurement_uncertainty",
                "balance_holds_across_train_transitions",
            ),
            counter_signatures=(
                "systematic_unaccounted_source_or_sink",
                "balance_violation_exceeds_measurement_uncertainty",
            ),
            probe_plan=_probe(
                15,
                "closed_balance",
                ("storage_delta", "inflow", "outflow", "source_sink", "balance_residual"),
                "support_if_closed_accounting_holds",
                "counter_if_unaccounted_mass_or_flow_remains",
            ),
            compiler_targets=(PP, EV, IC),
            not_applicable_conditions=(
                "no_declared_conserved_quantity",
                "boundary_sources_or_sinks_are_unobserved",
            ),
            invariances=("accounting_partition_refinement",),
        ),
        _template(
            template_id=T16,
            primary_parent_id=R4,
            parent_ids=(R4, R3),
            roles=(W, P),
            claim_schema=(
                "Declared connectivity, cycle, branch, or reachability features "
                "persist under admissible metric perturbations and predict utility."
            ),
            admissible_variable_types=("metric_graph", "filtration", "topological_feature"),
            support_signatures=(
                "feature_persists_across_admissible_perturbations",
                "persistent_feature_has_train_decision_increment",
            ),
            counter_signatures=(
                "feature_is_filtration_unstable",
                "persistent_feature_is_decision_irrelevant",
            ),
            probe_plan=_probe(
                16,
                "filtration_persistence",
                ("persistence_lifetime", "perturbation_stability", "decision_increment"),
                "support_if_stable_topology_adds_train_decision_signal",
                "counter_if_topology_is_unstable_or_irrelevant",
            ),
            compiler_targets=(EV,),
            not_applicable_conditions=(
                "no_metric_graph_or_filtration",
                "topological_feature_cannot_be_estimated_train_only",
            ),
            invariances=("graph_isomorphism",),
        ),
        _template(
            template_id=T17,
            primary_parent_id=R4,
            parent_ids=(R4, R6),
            roles=(W, R),
            claim_schema=(
                "A bound shape variant imposes a stable directional restriction: "
                "monotonicity, convexity, concavity, lattice submodularity, "
                "DR-submodularity, or stochastic dominance."
            ),
            admissible_variable_types=("ordered_variable", "partial_order", "lattice", "shape_variant"),
            support_signatures=(
                "variant_specific_shape_inequality_holds_across_train_folds",
                "shape_constrained_fit_has_no_material_train_crossfit_harm",
            ),
            counter_signatures=(
                "repeated_variant_specific_shape_reversal",
                "shape_constraint_causes_material_crossfit_harm",
            ),
            probe_plan=_probe(
                17,
                "variant_specific_shape",
                ("shape_variant", "inequality_violation_rate", "constrained_crossfit_delta"),
                "support_if_bound_shape_inequality_is_stable",
                "counter_if_variant_specific_reversals_are_stable",
            ),
            compiler_targets=(PP, EV, IC),
            not_applicable_conditions=(
                "shape_variant_is_unbound",
                "required_order_or_lattice_structure_is_absent",
            ),
            invariances=("order_preserving_reparameterization",),
        ),
        _template(
            template_id=T18,
            primary_parent_id=R5,
            parent_ids=(R5, R3),
            roles=(W, R),
            claim_schema=(
                "A bounded minority of units exerts disproportionate harmful "
                "influence while the regular mechanism remains stable."
            ),
            admissible_variable_types=("observation_unit", "influence_score", "robust_utility"),
            support_signatures=(
                "small_influence_set_drives_train_harm",
                "robust_estimate_is_stable_under_fixed_bounded_perturbation",
            ),
            counter_signatures=(
                "harm_is_diffuse",
                "bounded_leave_set_out_does_not_improve_robust_fit",
            ),
            probe_plan=_probe(
                18,
                "bounded_influence",
                ("unit_influence", "bounded_leave_set_out_delta", "robust_fold_stability"),
                "support_if_bounded_minority_drives_harm",
                "counter_if_harm_is_diffuse_or_robust_fit_does_not_stabilize",
            ),
            compiler_targets=(PP, EV),
            not_applicable_conditions=(
                "observation_units_are_not_separable",
                "bounded_influence_cannot_be_measured_train_only",
            ),
            invariances=("observation_unit_permutation",),
        ),
        _template(
            template_id=T19,
            primary_parent_id=R5,
            parent_ids=(R5, R6),
            roles=(G, D),
            claim_schema=(
                "Unsupported structure remains unresolved; when admissible claims "
                "are indistinguishable the policy preserves no-op or abstains."
            ),
            admissible_variable_types=("constraint_set", "ambiguity_set", "decision_action"),
            support_signatures=(
                "multiple_claims_remain_probe_indistinguishable",
                "no_op_has_lower_worst_case_train_regret",
            ),
            counter_signatures=(
                "one_claim_is_decisively_separated_by_frozen_probe",
                "no_op_has_materially_higher_train_regret",
            ),
            probe_plan=_probe(
                19,
                "ambiguity_and_noop",
                ("claim_support_margin", "calibration_error", "noop_regret"),
                "support_if_uncertainty_remains_and_noop_is_conservative",
                "counter_if_claim_is_separated_or_noop_is_harmful",
            ),
            compiler_targets=(PP, EV, ND),
            not_applicable_conditions=(
                "only_one_admissible_deterministic_action",
                "constraint_or_ambiguity_set_is_undefined",
            ),
            invariances=("claim_identifier_renaming",),
        ),
        _template(
            template_id=T20,
            primary_parent_id=R6,
            parent_ids=(R6,),
            roles=(G,),
            claim_schema=(
                "Every retained claim has a bounded TRAIN-only probe on which an "
                "observable outcome could falsify it relative to a competitor."
            ),
            admissible_variable_types=("claim", "counter_prediction", "admissible_probe"),
            support_signatures=(
                "claim_and_competitor_disagree_on_admissible_probe",
                "counter_outcome_is_declared_before_probe",
            ),
            counter_signatures=(
                "claim_is_observationally_equivalent_on_all_admissible_probes",
                "counter_outcome_is_added_after_observation",
            ),
            probe_plan=_probe(
                20,
                "expected_elimination",
                ("prediction_disagreement", "counter_outcome_commitment", "probe_cost"),
                "support_if_preregistered_probe_can_eliminate_claim",
                "counter_if_claim_has_no_admissible_falsifier",
            ),
            compiler_targets=(ND,),
            not_applicable_conditions=(
                "no_hypothesis_claim_is_being_considered",
            ),
            invariances=("claim_identifier_renaming",),
        ),
        _template(
            template_id=T21,
            primary_parent_id=R6,
            parent_ids=(R6,),
            roles=(G,),
            claim_schema=(
                "Claim support repeats across predeclared evidence channels whose "
                "custody and failure modes are separately recorded."
            ),
            admissible_variable_types=("evidence_channel", "evidence_receipt", "claim"),
            support_signatures=(
                "effect_repeats_across_predeclared_channels",
                "leave_channel_out_support_remains_positive",
            ),
            counter_signatures=(
                "support_depends_on_single_channel",
                "channels_share_unrecorded_custody_or_failure_mode",
            ),
            probe_plan=_probe(
                21,
                "leave_channel_out",
                ("channel_support", "leave_channel_out_delta", "custody_overlap"),
                "support_if_independent_channels_repeat_effect",
                "counter_if_single_or_correlated_channel_drives_support",
            ),
            compiler_targets=(ND,),
            not_applicable_conditions=(
                "fewer_than_two_predeclared_evidence_channels",
            ),
            invariances=("evidence_channel_order",),
        ),
        _template(
            template_id=T22,
            primary_parent_id=R6,
            parent_ids=(R6, R1),
            roles=(G, D),
            claim_schema=(
                "A retained claim changes a declared action, confidence boundary, "
                "or counterfactual regret under the frozen decision rule."
            ),
            admissible_variable_types=("claim", "decision_rule", "action", "utility"),
            support_signatures=(
                "claim_changes_optimal_action_or_boundary",
                "claim_reduces_train_counterfactual_regret",
            ),
            counter_signatures=(
                "claim_never_changes_decision",
                "claim_increases_counterfactual_regret",
            ),
            probe_plan=_probe(
                22,
                "counterfactual_decision_value",
                ("action_change", "decision_margin", "counterfactual_regret"),
                "support_if_claim_changes_decision_with_lower_regret",
                "counter_if_claim_is_decision_irrelevant_or_harmful",
            ),
            compiler_targets=(PP, EV),
            not_applicable_conditions=(
                "no_declared_action_space",
                "utility_or_decision_boundary_is_undefined",
            ),
            invariances=("utility_preserving_action_renaming",),
        ),
    )


def _legacy_aliases() -> tuple[LegacyAssumptionAlias, ...]:
    """Map the earlier 13-item catalog into v1; split mappings are deliberate."""

    return (
        LegacyAssumptionAlias("legacy13.v0.01_symmetry_equivariance", (T14,)),
        LegacyAssumptionAlias("legacy13.v0.02_locality_markov", (T08,)),
        LegacyAssumptionAlias("legacy13.v0.03_manifold_intrinsic_dimension", (T04, T10)),
        LegacyAssumptionAlias("legacy13.v0.04_low_rank_separability", (T03,)),
        LegacyAssumptionAlias("legacy13.v0.05_monotonic_shape", (T17,)),
        LegacyAssumptionAlias("legacy13.v0.06_diminishing_returns_submodularity", (T17,)),
        LegacyAssumptionAlias("legacy13.v0.07_conservation_balance", (T15,)),
        LegacyAssumptionAlias("legacy13.v0.08_stability_contractivity", (T13,)),
        LegacyAssumptionAlias("legacy13.v0.09_exchangeability_hierarchical_bayes", (T14, T07)),
        LegacyAssumptionAlias("legacy13.v0.10_maximum_entropy", (T19,)),
        LegacyAssumptionAlias("legacy13.v0.11_mdl_occam", (T01,)),
        LegacyAssumptionAlias("legacy13.v0.12_information_bottleneck", (T04,)),
        LegacyAssumptionAlias("legacy13.v0.13_pac_bayes", (T01, T21)),
    )


def build_universal_assumption_ontology_v1() -> UniversalAssumptionOntology:
    """Return the immutable, deterministically ordered v1 catalog."""

    return UniversalAssumptionOntology(
        version=VERSION,
        roots=_roots(),
        templates=_templates(),
        legacy_aliases=_legacy_aliases(),
    )


__all__ = [
    "VERSION",
    "T01",
    "T02",
    "T03",
    "T04",
    "T05",
    "T06",
    "T07",
    "T08",
    "T09",
    "T10",
    "T11",
    "T12",
    "T13",
    "T14",
    "T15",
    "T16",
    "T17",
    "T18",
    "T19",
    "T20",
    "T21",
    "T22",
    "build_universal_assumption_ontology_v1",
]
