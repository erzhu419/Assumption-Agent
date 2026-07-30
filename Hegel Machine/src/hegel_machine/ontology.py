"""Frozen v0.1 ontology: six roots, twenty-two leaves, six active laws."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import LawKind, RelationLaw, ViolationFunctionalSpec


@dataclass(frozen=True, slots=True)
class AssumptionTemplate:
    template_id: str
    root: str
    plane: str
    name: str
    compiler_target: str


UNIVERSAL_ASSUMPTIONS: tuple[AssumptionTemplate, ...] = (
    AssumptionTemplate("T01", "compression", "representation", "MDL/compressibility", "model_selector"),
    AssumptionTemplate("T02", "compression", "representation", "sparsity", "sparse_selector"),
    AssumptionTemplate("T03", "compression", "representation", "low_rank/separability", "factorizer"),
    AssumptionTemplate("T04", "compression", "representation", "minimal_sufficient_quotient", "quotient_builder"),
    AssumptionTemplate("T05", "decomposition", "world", "additivity/low_order_interaction", "interaction_verifier"),
    AssumptionTemplate("T06", "decomposition", "representation", "orthogonality/nonredundancy", "probe_selector"),
    AssumptionTemplate("T07", "decomposition", "world", "modularity/independent_mechanisms", "module_binder"),
    AssumptionTemplate("T08", "decomposition", "world", "locality/Markov_blanket", "locality_verifier"),
    AssumptionTemplate("T09", "decomposition", "world", "composition/mechanism_reuse", "path_verifier"),
    AssumptionTemplate("T10", "dynamics", "representation", "low_frequency/smoothness", "spectral_probe"),
    AssumptionTemplate("T11", "dynamics", "representation", "piecewise_smooth/sparse_change", "change_point_probe"),
    AssumptionTemplate("T12", "dynamics", "representation", "scale_separation/slow_variables", "scale_selector"),
    AssumptionTemplate("T13", "dynamics", "world", "stability/contraction/dissipation", "feedback_verifier"),
    AssumptionTemplate("T14", "invariance", "world", "symmetry/invariance/equivariance", "symmetry_verifier"),
    AssumptionTemplate("T15", "invariance", "world", "conservation/balance/flow", "balance_verifier"),
    AssumptionTemplate("T16", "invariance", "world", "topological_persistence", "topology_probe"),
    AssumptionTemplate("T17", "shape", "world", "monotonicity/order/diminishing_returns", "order_verifier"),
    AssumptionTemplate("T18", "uncertainty", "world", "typical_rule/sparse_exceptions", "contamination_model"),
    AssumptionTemplate("T19", "uncertainty", "governance", "maximum_entropy/minimum_commitment", "abstention_policy"),
    AssumptionTemplate("T20", "governance", "governance", "falsifiability/active_discrimination", "falsifier"),
    AssumptionTemplate("T21", "governance", "governance", "evidence_triangulation", "evidence_ledger"),
    AssumptionTemplate("T22", "governance", "governance", "decision_relevance/boundary_sufficiency", "task_geometry"),
)


ACTIVE_FUNCTIONALS: tuple[ViolationFunctionalSpec, ...] = (
    ViolationFunctionalSpec(
        "vf_symmetry_v1",
        LawKind.SYMMETRY,
        ("forward", "transformed", "common_codomains"),
        "normalized maximum equivariance residual",
        0.01,
    ),
    ViolationFunctionalSpec(
        "vf_monotonicity_v1",
        LawKind.MONOTONICITY,
        ("x_low", "x_high", "y_low", "y_high", "direction"),
        "normalized order violation",
        0.01,
    ),
    ViolationFunctionalSpec(
        "vf_conservation_v1",
        LawKind.CONSERVATION,
        (
            "storage_delta",
            "inflows",
            "outflows",
            "sources",
            "sinks",
            "boundary_observed",
        ),
        "normalized signed balance residual",
        0.01,
    ),
    ViolationFunctionalSpec(
        "vf_complementarity_v1",
        LawKind.COMPLEMENTARITY,
        (
            "u_empty",
            "u_a",
            "u_b",
            "u_ab",
            "expected_interaction",
            "interaction_margin",
        ),
        "preregistered pair-interaction sign/margin violation",
        0.01,
    ),
    ViolationFunctionalSpec(
        "vf_negative_feedback_v1",
        LawKind.NEGATIVE_FEEDBACK,
        (
            "disturbance_delta",
            "response_delta",
            "deviation_before_response",
            "deviation_after_response",
            "controlled_quantity_observed",
            "disturbance_precedes_response",
            "system_induced_response",
            "same_controlled_quantity",
            "local_stability_window_observed",
            "response_margin",
            "mitigation_margin",
        ),
        "strict sign-opposition and mitigation-margin violation",
        0.01,
    ),
    ViolationFunctionalSpec(
        "vf_locality_v1",
        LawKind.LOCALITY,
        (
            "conditional_a",
            "conditional_b",
            "blanket_observed",
            "same_blanket_state",
        ),
        "conditional total variation outside a fixed Markov blanket",
        0.01,
    ),
)


ACTIVE_LAWS: tuple[RelationLaw, ...] = (
    RelationLaw(
        "law_symmetry_v1",
        LawKind.SYMMETRY,
        "T_g ∘ f = f ∘ T_g",
        2,
        ("source", "transformed_source"),
        "hegel_machine.laws.evaluate_symmetry",
        "vf_symmetry_v1",
        ("declared_common_codomains",),
        ("phase2_default",),
        ACTIVE_FUNCTIONALS[0].required_observables,
        (
            ("source", ("forward",)),
            ("transformed_source", ("transformed", "common_codomains")),
        ),
    ),
    RelationLaw(
        "law_monotonicity_v1",
        LawKind.MONOTONICITY,
        "x₁ ≼ x₂ ⇒ f(x₁) ≼ f(x₂)",
        2,
        ("lower", "upper"),
        "hegel_machine.laws.evaluate_monotonicity",
        "vf_monotonicity_v1",
        ("declared_partial_order",),
        ("phase2_default",),
        ACTIVE_FUNCTIONALS[1].required_observables,
        (
            ("lower", ("x_low", "y_low", "direction")),
            ("upper", ("x_high", "y_high", "direction")),
        ),
    ),
    RelationLaw(
        "law_conservation_v1",
        LawKind.CONSERVATION,
        "Δstorage + out − in − source + sink = 0",
        3,
        ("system", "source", "sink"),
        "hegel_machine.laws.evaluate_conservation",
        "vf_conservation_v1",
        ("closed_observed_boundary", "declared_time_window"),
        ("phase2_default",),
        ACTIVE_FUNCTIONALS[2].required_observables,
        (
            ("system", ("storage_delta", "boundary_observed")),
            ("source", ("inflows", "sources")),
            ("sink", ("outflows", "sinks")),
        ),
    ),
    RelationLaw(
        "law_complementarity_v1",
        LawKind.COMPLEMENTARITY,
        "I(a,b)=U(ab)−U(a)−U(b)+U(∅)",
        2,
        ("intervention_a", "intervention_b"),
        "hegel_machine.laws.evaluate_complementarity",
        "vf_complementarity_v1",
        ("declared_utility_direction", "fixed_baseline"),
        ("phase2_default",),
        ACTIVE_FUNCTIONALS[3].required_observables,
        (
            ("intervention_a", ("u_empty", "u_a", "u_ab")),
            ("intervention_b", ("u_empty", "u_b", "u_ab")),
        ),
    ),
    RelationLaw(
        "law_negative_feedback_v1",
        LawKind.NEGATIVE_FEEDBACK,
        "-disturbance · response ≥ response_margin and "
        "|before|−|after| ≥ mitigation_margin",
        3,
        ("disturbance", "response", "controlled_quantity"),
        "hegel_machine.laws.evaluate_negative_feedback",
        "vf_negative_feedback_v1",
        ("observable_temporal_order", "local_stability_window"),
        ("phase2_default",),
        ACTIVE_FUNCTIONALS[4].required_observables,
        (
            (
                "disturbance",
                ("disturbance_delta", "disturbance_precedes_response"),
            ),
            (
                "response",
                (
                    "response_delta",
                    "system_induced_response",
                    "response_margin",
                ),
            ),
            (
                "controlled_quantity",
                (
                    "deviation_before_response",
                    "deviation_after_response",
                    "controlled_quantity_observed",
                    "same_controlled_quantity",
                    "local_stability_window_observed",
                    "mitigation_margin",
                ),
            ),
        ),
    ),
    RelationLaw(
        "law_locality_v1",
        LawKind.LOCALITY,
        "P(Y|blanket, outside₁)=P(Y|blanket, outside₂)",
        3,
        ("target", "markov_blanket", "outside_context"),
        "hegel_machine.laws.evaluate_locality",
        "vf_locality_v1",
        ("fixed_blanket_state",),
        ("phase2_default",),
        ACTIVE_FUNCTIONALS[5].required_observables,
        (
            ("target", ("conditional_a", "conditional_b")),
            (
                "markov_blanket",
                ("blanket_observed", "same_blanket_state"),
            ),
            ("outside_context", ("conditional_a", "conditional_b")),
        ),
    ),
)


def validate_registry() -> None:
    if len(UNIVERSAL_ASSUMPTIONS) != 22:
        raise AssertionError("the universal ontology must contain exactly 22 leaves")
    if len({item.template_id for item in UNIVERSAL_ASSUMPTIONS}) != 22:
        raise AssertionError("universal ontology identifiers are not unique")
    if {law.kind for law in ACTIVE_LAWS} != set(LawKind):
        raise AssertionError("the active Phase-2 law library is incomplete")
    functional_ids = {item.functional_id for item in ACTIVE_FUNCTIONALS}
    if any(law.violation_functional_id not in functional_ids for law in ACTIVE_LAWS):
        raise AssertionError("an active law lacks its violation functional")


validate_registry()
