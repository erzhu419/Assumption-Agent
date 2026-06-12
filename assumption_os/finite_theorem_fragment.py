"""Finite theorem fragment for category-inspired assumption certificates.

This module deliberately implements a bounded, machine-checkable fragment:
finite categories, functors, naturality squares, finite poset
limits/colimits, finite adjunctions, strict monoidal one-object examples,
finite stochastic kernels, exact finite Blackwell witnesses for small square
kernels, Fisher-Rao measurement checks, and a small deterministic
natural-language-to-diagram extractor.

It is not a replacement for Lean/Coq/mathlib.  The purpose is to make the
formal layer stronger and more systematic while preserving honest claim gates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .finite_theorem_lean_verifier import build_finite_theorem_lean_verifier_payload


DEFAULT_OUT = PAPER_DIR / "finite_theorem_fragment_20260612.json"


@dataclass(frozen=True)
class FiniteMorphism:
    morphism_id: str
    source: str
    target: str
    kind: str = "ordinary"


@dataclass(frozen=True)
class ExplicitFiniteCategory:
    category_id: str
    objects: tuple[str, ...]
    morphisms: tuple[FiniteMorphism, ...]
    composition_table: dict[str, str]

    @property
    def morphism_by_id(self) -> dict[str, FiniteMorphism]:
        return {morphism.morphism_id: morphism for morphism in self.morphisms}

    def compose(self, left: str, right: str) -> str | None:
        return self.composition_table.get(_composition_key(left, right))

    def identity_for(self, obj: str) -> str | None:
        identities = [
            morphism.morphism_id
            for morphism in self.morphisms
            if morphism.kind == "identity" and morphism.source == obj and morphism.target == obj
        ]
        return identities[0] if len(identities) == 1 else None

    def validate(self) -> dict[str, Any]:
        object_set = set(self.objects)
        morphism_by_id = self.morphism_by_id
        issues: list[dict[str, str]] = []
        if len(object_set) != len(self.objects) or not self.objects:
            issues.append({"issue": "objects_not_unique_or_empty"})

        for morphism in self.morphisms:
            if morphism.source not in object_set or morphism.target not in object_set:
                issues.append({"issue": "morphism_endpoint_not_object", "morphism_id": morphism.morphism_id})

        for obj in self.objects:
            if self.identity_for(obj) is None:
                issues.append({"issue": "identity_missing_or_ambiguous", "object": obj})

        for left in self.morphisms:
            for right in self.morphisms:
                if left.target != right.source:
                    continue
                out_id = self.compose(left.morphism_id, right.morphism_id)
                if out_id is None:
                    issues.append(
                        {
                            "issue": "composition_missing_for_composable_pair",
                            "left": left.morphism_id,
                            "right": right.morphism_id,
                        }
                    )
                    continue
                out = morphism_by_id.get(out_id)
                if out is None:
                    issues.append({"issue": "composition_result_unknown", "result": out_id})
                    continue
                if out.source != left.source or out.target != right.target:
                    issues.append(
                        {
                            "issue": "composition_result_endpoint_mismatch",
                            "left": left.morphism_id,
                            "right": right.morphism_id,
                            "result": out_id,
                        }
                    )

        identity_law_pass = self._identity_law_passes()
        associativity_pass = self._associativity_passes()
        if not identity_law_pass:
            issues.append({"issue": "identity_law_failed"})
        if not associativity_pass:
            issues.append({"issue": "associativity_failed"})

        return {
            "category_id": self.category_id,
            "valid": not issues,
            "object_count": len(self.objects),
            "morphism_count": len(self.morphisms),
            "composition_entry_count": len(self.composition_table),
            "identity_law_pass": identity_law_pass,
            "associativity_pass": associativity_pass,
            "issue_count": len(issues),
            "issues": issues,
        }

    def _identity_law_passes(self) -> bool:
        for morphism in self.morphisms:
            left_id = self.identity_for(morphism.source)
            right_id = self.identity_for(morphism.target)
            if left_id is None or right_id is None:
                return False
            if self.compose(left_id, morphism.morphism_id) != morphism.morphism_id:
                return False
            if self.compose(morphism.morphism_id, right_id) != morphism.morphism_id:
                return False
        return True

    def _associativity_passes(self) -> bool:
        morphism_ids = {morphism.morphism_id for morphism in self.morphisms}
        for first in self.morphisms:
            for second in self.morphisms:
                if first.target != second.source:
                    continue
                first_second = self.compose(first.morphism_id, second.morphism_id)
                if first_second not in morphism_ids:
                    return False
                for third in self.morphisms:
                    if second.target != third.source:
                        continue
                    second_third = self.compose(second.morphism_id, third.morphism_id)
                    if second_third not in morphism_ids:
                        return False
                    left = self.compose(first_second, third.morphism_id)
                    right = self.compose(first.morphism_id, second_third)
                    if left != right:
                        return False
        return True


@dataclass(frozen=True)
class FiniteFunctor:
    functor_id: str
    source: ExplicitFiniteCategory
    target: ExplicitFiniteCategory
    object_map: dict[str, str]
    morphism_map: dict[str, str]

    def validate(self) -> dict[str, Any]:
        issues: list[dict[str, str]] = []
        target_morphisms = self.target.morphism_by_id
        source_morphisms = self.source.morphism_by_id
        for obj in self.source.objects:
            if self.object_map.get(obj) not in self.target.objects:
                issues.append({"issue": "object_map_missing_or_unknown", "object": obj})
        for morphism in self.source.morphisms:
            mapped = self.morphism_map.get(morphism.morphism_id)
            if mapped not in target_morphisms:
                issues.append({"issue": "morphism_map_missing_or_unknown", "morphism_id": morphism.morphism_id})
                continue
            mapped_morphism = target_morphisms[mapped]
            if mapped_morphism.source != self.object_map[morphism.source]:
                issues.append({"issue": "morphism_map_source_mismatch", "morphism_id": morphism.morphism_id})
            if mapped_morphism.target != self.object_map[morphism.target]:
                issues.append({"issue": "morphism_map_target_mismatch", "morphism_id": morphism.morphism_id})

        identity_pass = True
        for obj in self.source.objects:
            source_identity = self.source.identity_for(obj)
            target_identity = self.target.identity_for(self.object_map[obj])
            if source_identity is None or self.morphism_map.get(source_identity) != target_identity:
                identity_pass = False

        composition_pass = True
        for left in source_morphisms.values():
            for right in source_morphisms.values():
                if left.target != right.source:
                    continue
                composed = self.source.compose(left.morphism_id, right.morphism_id)
                if composed is None:
                    composition_pass = False
                    continue
                mapped_left = self.morphism_map[left.morphism_id]
                mapped_right = self.morphism_map[right.morphism_id]
                mapped_composed = self.morphism_map[composed]
                target_composed = self.target.compose(mapped_left, mapped_right)
                if mapped_composed != target_composed:
                    composition_pass = False

        if not identity_pass:
            issues.append({"issue": "functor_identity_law_failed"})
        if not composition_pass:
            issues.append({"issue": "functor_composition_law_failed"})
        return {
            "functor_id": self.functor_id,
            "valid": not issues,
            "identity_preservation_pass": identity_pass,
            "composition_preservation_pass": composition_pass,
            "issue_count": len(issues),
            "issues": issues,
        }


@dataclass(frozen=True)
class NaturalTransformation:
    transformation_id: str
    source_functor: FiniteFunctor
    target_functor: FiniteFunctor
    components: dict[str, str]

    def validate(self) -> dict[str, Any]:
        target_category = self.source_functor.target
        target_morphisms = target_category.morphism_by_id
        issues: list[dict[str, str]] = []
        for obj in self.source_functor.source.objects:
            component_id = self.components.get(obj)
            component = target_morphisms.get(component_id or "")
            if component is None:
                issues.append({"issue": "component_missing_or_unknown", "object": obj})
                continue
            if component.source != self.source_functor.object_map[obj]:
                issues.append({"issue": "component_source_mismatch", "object": obj})
            if component.target != self.target_functor.object_map[obj]:
                issues.append({"issue": "component_target_mismatch", "object": obj})

        square_rows = []
        for morphism in self.source_functor.source.morphisms:
            x = morphism.source
            y = morphism.target
            alpha_x = self.components.get(x)
            alpha_y = self.components.get(y)
            f_source = self.source_functor.morphism_map[morphism.morphism_id]
            f_target = self.target_functor.morphism_map[morphism.morphism_id]
            left_path = target_category.compose(alpha_x, f_target) if alpha_x else None
            right_path = target_category.compose(f_source, alpha_y) if alpha_y else None
            commutes = left_path is not None and left_path == right_path
            square_rows.append(
                {
                    "morphism_id": morphism.morphism_id,
                    "alpha_x_then_target_f": left_path,
                    "source_f_then_alpha_y": right_path,
                    "commutes": commutes,
                }
            )
            if not commutes:
                issues.append({"issue": "naturality_square_failed", "morphism_id": morphism.morphism_id})
        return {
            "transformation_id": self.transformation_id,
            "valid": not issues,
            "naturality_square_count": len(square_rows),
            "naturality_square_pass_count": sum(1 for row in square_rows if row["commutes"]),
            "naturality_squares": square_rows,
            "issue_count": len(issues),
            "issues": issues,
        }


def build_finite_theorem_fragment_payload(
    *,
    root: Path,
    eval_id: str = "finite_theorem_fragment_20260612",
    out: Path | None = None,
    write_artifact: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    source_category, target_category, functor = _build_functor_example()
    source_validation = source_category.validate()
    target_validation = target_category.validate()
    functor_validation = functor.validate()
    naturality = NaturalTransformation(
        transformation_id="identity_natural_transformation",
        source_functor=functor,
        target_functor=functor,
        components={obj: target_category.identity_for(functor.object_map[obj]) or "" for obj in source_category.objects},
    ).validate()
    bad_naturality = NaturalTransformation(
        transformation_id="bad_component_negative_control",
        source_functor=functor,
        target_functor=functor,
        components={
            "A": "u",
            "B": target_category.identity_for("Y") or "",
            "C": target_category.identity_for("Z") or "",
        },
    ).validate()
    poset = _build_diamond_poset()
    limits_colimits = _finite_limit_colimit_suite(poset)
    adjunction = _finite_adjunction_suite()
    monoidal = _finite_monoidal_suite()
    markov = _finite_markov_blackwell_suite()
    geometry = _finite_information_geometry_suite()
    nl = _natural_language_diagram_suite()
    lean_verifier = build_finite_theorem_lean_verifier_payload(
        root=root,
        eval_id=f"{eval_id}_lean_verifier",
        run_lean_if_available=True,
    )

    metrics = {
        "source_category_valid": source_validation["valid"],
        "target_category_valid": target_validation["valid"],
        "identity_law_pass": source_validation["identity_law_pass"] and target_validation["identity_law_pass"],
        "associativity_pass": source_validation["associativity_pass"] and target_validation["associativity_pass"],
        "functor_identity_pass": functor_validation["identity_preservation_pass"],
        "functor_composition_pass": functor_validation["composition_preservation_pass"],
        "naturality_pass": naturality["valid"],
        "naturality_negative_control_rejected": bad_naturality["valid"] is False,
        "finite_limit_count": limits_colimits["limit_count"],
        "finite_limit_colimit_pass": limits_colimits["pass"],
        "adjunction_pass": adjunction["adjunction_pass"],
        "adjunction_negative_control_rejected": adjunction["negative_control_rejected"],
        "monoidal_pass": monoidal["pass"],
        "markov_category_fragment_pass": markov["markov_category_fragment_pass"],
        "blackwell_exact_witness_pass": markov["blackwell_exact_witness_pass"],
        "blackwell_negative_control_rejected": markov["blackwell_negative_control_rejected"],
        "fisher_geometry_metric_laws_pass": geometry["metric_laws_pass"],
        "nl_diagram_extraction_count": nl["extracted_count"],
        "nl_diagram_certificate_pass_rate": nl["certificate_pass_rate"],
        "nl_negative_control_abstained": nl["negative_control_abstained"],
        "external_lean_check_passed": lean_verifier["metrics"]["external_lean_check_passed"],
        "external_lean_theorem_count": lean_verifier["metrics"]["lean_theorem_count"],
    }
    fragment_allowed = all(
        [
            metrics["source_category_valid"],
            metrics["target_category_valid"],
            metrics["identity_law_pass"],
            metrics["associativity_pass"],
            metrics["functor_identity_pass"],
            metrics["functor_composition_pass"],
            metrics["naturality_pass"],
            metrics["naturality_negative_control_rejected"],
            metrics["finite_limit_colimit_pass"],
            metrics["adjunction_pass"],
            metrics["adjunction_negative_control_rejected"],
            metrics["monoidal_pass"],
            metrics["markov_category_fragment_pass"],
            metrics["blackwell_exact_witness_pass"],
            metrics["blackwell_negative_control_rejected"],
            metrics["fisher_geometry_metric_laws_pass"],
            metrics["nl_diagram_certificate_pass_rate"] == 1.0,
            metrics["nl_negative_control_abstained"],
            metrics["external_lean_check_passed"],
        ]
    )
    metrics.update(
        {
            "finite_theorem_fragment_claim_allowed": fragment_allowed,
            "lean_verified_finite_theorem_fragment_claim_allowed": (
                fragment_allowed and lean_verifier["metrics"]["finite_theorem_fragment_lean_verified"]
            ),
            "external_proof_assistant_integrated": lean_verifier["pass"],
            "full_theorem_prover_claim_allowed": False,
        }
    )
    gates = {
        "category_laws_checked": metrics["identity_law_pass"] and metrics["associativity_pass"],
        "functor_laws_checked": metrics["functor_identity_pass"] and metrics["functor_composition_pass"],
        "naturality_checked": metrics["naturality_pass"] and metrics["naturality_negative_control_rejected"],
        "limits_colimits_checked": metrics["finite_limit_colimit_pass"],
        "adjunction_checked": metrics["adjunction_pass"] and metrics["adjunction_negative_control_rejected"],
        "monoidal_structure_checked": metrics["monoidal_pass"],
        "finite_markov_blackwell_checked": (
            metrics["markov_category_fragment_pass"]
            and metrics["blackwell_exact_witness_pass"]
            and metrics["blackwell_negative_control_rejected"]
        ),
        "fisher_geometry_metric_laws_checked": metrics["fisher_geometry_metric_laws_pass"],
        "natural_language_diagram_certificates_checked": (
            metrics["nl_diagram_certificate_pass_rate"] == 1.0 and metrics["nl_negative_control_abstained"]
        ),
        "external_lean_verifier_passes": metrics["external_lean_check_passed"] is True
        and metrics["external_lean_theorem_count"] >= 20,
        "finite_fragment_claim_allowed": metrics["finite_theorem_fragment_claim_allowed"],
        "full_theorem_prover_claim_blocked_for_unbounded_scope": (
            metrics["full_theorem_prover_claim_allowed"] is False
        ),
    }
    payload = {
        "eval_id": eval_id,
        "eval_kind": "finite_theorem_fragment",
        "performance_validation": True,
        "validation_scope": (
            "Systematic finite theorem fragment for the formal layer.  It checks category laws, functor laws, "
            "naturality, selected finite limits/colimits, finite adjunctions, strict monoidal structure, "
            "finite Markov/Blackwell witnesses, Fisher-Rao metric laws, and bounded NL-to-diagram certificates. "
            "It remains a finite fragment, not a full proof assistant."
        ),
        "requested_capability_status": {
            "identity_composition_functor_naturality": "implemented_for_explicit_finite_categories",
            "limits_colimits_adjunction_monoidal": "implemented_for_finite_poset_and_strict_monoidal_fragments",
            "markov_blackwell_fisher": "implemented_for_finite_stochastic_kernels_and_bernoulli_metric_checks",
            "natural_language_to_machine_checkable_certificate": "implemented_as_bounded_rule_extractor_with_abstention",
            "full_theorem_prover": "blocked_until_external_proof_assistant_integration",
        },
        "category_law_fragment": {
            "source_category": _category_to_dict(source_category),
            "target_category": _category_to_dict(target_category),
            "source_validation": source_validation,
            "target_validation": target_validation,
            "functor": {
                "functor_id": functor.functor_id,
                "object_map": functor.object_map,
                "morphism_map": functor.morphism_map,
                "validation": functor_validation,
            },
            "naturality": naturality,
            "naturality_negative_control": bad_naturality,
        },
        "finite_limits_colimits": limits_colimits,
        "finite_adjunction": adjunction,
        "finite_monoidal_structure": monoidal,
        "finite_markov_blackwell": markov,
        "information_geometry_fragment": geometry,
        "natural_language_diagram_extraction": nl,
        "external_lean_verifier": lean_verifier,
        "claim_gate": {
            "finite_theorem_fragment_claim_allowed": metrics["finite_theorem_fragment_claim_allowed"],
            "lean_verified_finite_theorem_fragment_claim_allowed": metrics[
                "lean_verified_finite_theorem_fragment_claim_allowed"
            ],
            "allowed_claim": (
                "Lean-verified finite theorem fragment over explicit finite categories, selected finite "
                "categorical constructions, finite stochastic kernels, and bounded NL diagram certificates"
            ),
            "full_theorem_prover_claim_allowed": metrics["full_theorem_prover_claim_allowed"],
            "full_theorem_prover_blockers": [
                "no arbitrary infinite-category or higher-category support",
                "no dependent-type theorem development",
                "no complete semantic equivalence proof for arbitrary natural language",
                "no unbounded Markov-category theorem library",
            ],
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
    }
    if write_artifact:
        output = _resolve(root, out or DEFAULT_OUT)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["artifact_path"] = _display_path(root, output)
    return payload


def extract_natural_language_diagram(text: str) -> dict[str, Any]:
    lower = text.lower()
    if _has_any(lower, [
        "not automatically",
        "not the same as",
        "does not imply",
        " is not a randomized",
        " is not an incremental",
        " is not an error",
        " is not a mass",
        " is not branch",
        " is not a phase",
        " is not failover",
    ]):
        return {
            "text_hash": stable_hash(text),
            "status": "not_applicable",
            "reason": "explicit negated analogy or unsupported near-neighbor pattern",
            "certificate": None,
        }
    if _has_any(lower, ["le chatelier", "lenz", "oppose", "opposing response", "counteract", "equilibrium", "negative feedback", "opposite response", "stabilizing", "抵消", "平衡"]):
        return _nl_certificate(
            family="negative_feedback_regulation",
            text=text,
            objects=("perturbation", "opposing_response", "regulated_state"),
            arrows=(("trigger_response", "perturbation", "opposing_response"), ("restore_invariant", "opposing_response", "regulated_state")),
            invariants=("directional_opposition", "equilibrium_tendency"),
        )
    if _has_any(lower, ["resnet", "skip connection", "residual", "identity path"]):
        return _nl_certificate(
            family="residual_transport",
            text=text,
            objects=("input_state", "transformed_state", "output_state"),
            arrows=(("transform", "input_state", "transformed_state"), ("identity_carry", "input_state", "output_state")),
            invariants=("gradient_transport", "information_preservation"),
        )
    if _has_any(lower, ["error correction", "checksum", "parity", "redundancy check", "syndrome", "recovered message", "targeted correction"]):
        return _nl_certificate(
            family="error_correction_feedback",
            text=text,
            objects=("noisy_message", "redundancy_signal", "corrected_message"),
            arrows=(("detect_error", "noisy_message", "redundancy_signal"), ("repair_state", "redundancy_signal", "corrected_message")),
            invariants=("message_identity_recovered", "noise_not_reinforced"),
        )
    if _has_any(lower, ["regularization", "smoothness penalty", "overfit", "weight decay", "implicit bias", "penalty term", "reducing variance", "reduces variance", "complexity penalty"]):
        return _nl_certificate(
            family="regularization_smoothing",
            text=text,
            objects=("flexible_model", "complexity_penalty", "generalized_solution"),
            arrows=(("penalize_complexity", "flexible_model", "complexity_penalty"), ("select_smoother", "complexity_penalty", "generalized_solution")),
            invariants=("variance_reduced", "signal_structure_preserved"),
        )
    if _has_any(lower, ["autocorrelation", "random noise", "denoise", "jepa", "gaussian", "latent structure", "stable feature", "stable signal", "independent noise", "persistent signal", "random fluctuations", "signal extraction", "raw measurement", "correlation filter"]):
        return _nl_certificate(
            family="noise_invariant_signal_extraction",
            text=text,
            objects=("raw_measurement", "correlation_filter", "stable_signal"),
            arrows=(("filter_noise", "raw_measurement", "correlation_filter"), ("extract_signal", "correlation_filter", "stable_signal")),
            invariants=("random_noise_cancels", "stable_feature_persists"),
        )
    if _has_any(lower, ["bottleneck", "saturation", "enzyme", "queue capacity", "throughput ceiling", "rate limiting", "limiting step"]):
        return _nl_certificate(
            family="bottleneck_capacity_limit",
            text=text,
            objects=("input_flow", "capacity_limiter", "bounded_output"),
            arrows=(("load_limiter", "input_flow", "capacity_limiter"), ("cap_throughput", "capacity_limiter", "bounded_output")),
            invariants=("limiting_step_controls_rate", "extra_input_has_diminishing_return"),
        )
    if _has_any(lower, ["strangler", "bridge retrofit", "adapter layer", "incremental migration", "compatibility bridge", "old api", "legacy behavior", "legacy state", "controlled transfer"]):
        return _nl_certificate(
            family="bridge_decomposition",
            text=text,
            objects=("legacy_state", "bridge_interface", "target_state"),
            arrows=(("wrap_legacy", "legacy_state", "bridge_interface"), ("gradually_transfer", "bridge_interface", "target_state")),
            invariants=("interface_continuity", "blast_radius_reduction"),
        )
    if _has_any(lower, ["randomized", "clinical trial", "a/b test", "control group", "treatment group", "effect estimate", "causal comparison", "measured lift"]):
        return _nl_certificate(
            family="randomized_counterfactual_evaluation",
            text=text,
            objects=("population", "random_assignment", "causal_comparison"),
            arrows=(("split_units", "population", "random_assignment"), ("estimate_effect", "random_assignment", "causal_comparison")),
            invariants=("exchangeability", "counterfactual_contrast"),
        )
    if _has_any(lower, ["mass balance", "budget balance", "conservation", "flow conservation", "stock and flow", "cash runway", "no free creation", "input-output"]):
        return _nl_certificate(
            family="conservation_balance",
            text=text,
            objects=("inflow", "conserved_stock", "outflow_accounting"),
            arrows=(("accumulate_stock", "inflow", "conserved_stock"), ("balance_outflow", "conserved_stock", "outflow_accounting")),
            invariants=("input_output_accounting", "no_free_creation"),
        )
    if _has_any(lower, ["error correction", "checksum", "parity", "redundancy check", "syndrome", "recovered message", "targeted correction"]):
        return _nl_certificate(
            family="error_correction_feedback",
            text=text,
            objects=("noisy_message", "redundancy_signal", "corrected_message"),
            arrows=(("detect_error", "noisy_message", "redundancy_signal"), ("repair_state", "redundancy_signal", "corrected_message")),
            invariants=("message_identity_recovered", "noise_not_reinforced"),
        )
    if _has_any(lower, ["branch and bound", "beam search", "pruning", "search space", "dominance bound", "smaller frontier", "optimal paths", "bound evidence"]):
        return _nl_certificate(
            family="search_pruning_by_bounds",
            text=text,
            objects=("candidate_space", "bound_evidence", "reduced_frontier"),
            arrows=(("score_bound", "candidate_space", "bound_evidence"), ("prune_dominated", "bound_evidence", "reduced_frontier")),
            invariants=("optimal_candidate_preserved", "dominated_paths_removed"),
        )
    if _has_any(lower, ["phase transition", "critical threshold", "tipping point", "percolation", "threshold effect", "locally linear", "regime boundary", "structural boundary", "control parameter"]):
        return _nl_certificate(
            family="threshold_phase_transition",
            text=text,
            objects=("control_parameter", "critical_boundary", "new_regime"),
            arrows=(("approach_threshold", "control_parameter", "critical_boundary"), ("switch_regime", "critical_boundary", "new_regime")),
            invariants=("local_change_not_linear_near_threshold", "regime_boundary_matters"),
        )
    if _has_any(lower, ["compiler", "subassembly", "modular composition", "interface contract", "assembly line", "local contract", "manufacturing"]):
        return _nl_certificate(
            family="modular_composition",
            text=text,
            objects=("component_spec", "interface_contract", "assembled_system"),
            arrows=(("compile_component", "component_spec", "interface_contract"), ("compose_module", "interface_contract", "assembled_system")),
            invariants=("local_contract_preserved", "composition_enables_scaling"),
        )
    if _has_any(lower, ["failover", "replication", "fault tolerance", "backup path", "redundant channel", "recover the service", "single failure", "continuity"]):
        return _nl_certificate(
            family="redundant_fault_tolerance",
            text=text,
            objects=("primary_path", "redundant_path", "service_continuity"),
            arrows=(("mirror_state", "primary_path", "redundant_path"), ("recover_service", "redundant_path", "service_continuity")),
            invariants=("single_failure_masked", "continuity_preserved"),
        )
    if _has_any(lower, ["regularization", "smoothness penalty", "overfit", "weight decay", "implicit bias", "penalty term", "reducing variance", "reduces variance", "complexity penalty"]):
        return _nl_certificate(
            family="regularization_smoothing",
            text=text,
            objects=("flexible_model", "complexity_penalty", "generalized_solution"),
            arrows=(("penalize_complexity", "flexible_model", "complexity_penalty"), ("select_smoother", "complexity_penalty", "generalized_solution")),
            invariants=("variance_reduced", "signal_structure_preserved"),
        )
    return {
        "text_hash": stable_hash(text),
        "status": "not_applicable",
        "reason": "no supported finite formal pattern found",
        "certificate": None,
    }


def _build_functor_example() -> tuple[ExplicitFiniteCategory, ExplicitFiniteCategory, FiniteFunctor]:
    source = _arrow_category(
        category_id="source_strategy_category",
        objects=("A", "B", "C"),
        first="f",
        second="g",
        composed="gf",
    )
    target = _arrow_category(
        category_id="target_strategy_category",
        objects=("X", "Y", "Z"),
        first="u",
        second="v",
        composed="vu",
    )
    functor = FiniteFunctor(
        functor_id="strategy_structure_preserving_functor",
        source=source,
        target=target,
        object_map={"A": "X", "B": "Y", "C": "Z"},
        morphism_map={
            "id_A": "id_X",
            "id_B": "id_Y",
            "id_C": "id_Z",
            "f": "u",
            "g": "v",
            "gf": "vu",
        },
    )
    return source, target, functor


def _arrow_category(
    *,
    category_id: str,
    objects: tuple[str, str, str],
    first: str,
    second: str,
    composed: str,
) -> ExplicitFiniteCategory:
    a, b, c = objects
    morphisms = (
        FiniteMorphism(f"id_{a}", a, a, "identity"),
        FiniteMorphism(f"id_{b}", b, b, "identity"),
        FiniteMorphism(f"id_{c}", c, c, "identity"),
        FiniteMorphism(first, a, b),
        FiniteMorphism(second, b, c),
        FiniteMorphism(composed, a, c, "composite"),
    )
    composition: dict[str, str] = {}
    for morphism in morphisms:
        composition[_composition_key(f"id_{morphism.source}", morphism.morphism_id)] = morphism.morphism_id
        composition[_composition_key(morphism.morphism_id, f"id_{morphism.target}")] = morphism.morphism_id
    composition[_composition_key(first, second)] = composed
    return ExplicitFiniteCategory(category_id=category_id, objects=objects, morphisms=morphisms, composition_table=composition)


def _build_diamond_poset() -> dict[str, Any]:
    objects = ("bottom", "A", "B", "top")
    leq_pairs = {
        ("bottom", "bottom"),
        ("bottom", "A"),
        ("bottom", "B"),
        ("bottom", "top"),
        ("A", "A"),
        ("A", "top"),
        ("B", "B"),
        ("B", "top"),
        ("top", "top"),
    }
    return {"objects": objects, "leq_pairs": leq_pairs}


def _finite_limit_colimit_suite(poset: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "terminal_top": _is_terminal(poset, "top"),
        "initial_bottom": _is_initial(poset, "bottom"),
        "product_A_B_is_bottom": _is_product(poset, "A", "B", "bottom"),
        "coproduct_A_B_is_top": _is_coproduct(poset, "A", "B", "top"),
        "pullback_A_top_B_top_is_bottom": _is_pullback(poset, "A", "B", "top", "bottom"),
        "pushout_bottom_A_bottom_B_is_top": _is_pushout(poset, "bottom", "A", "B", "top"),
    }
    return {
        "fragment": "finite_poset_category",
        "objects": list(poset["objects"]),
        "limit_count": len(checks),
        "checks": checks,
        "pass": all(checks.values()),
        "not_claimed": [
            "arbitrary complete or cocomplete categories",
            "general Kan extensions",
            "large-category limit computation",
        ],
    }


def _finite_adjunction_suite() -> dict[str, Any]:
    p_objects = ("p0", "p1")
    q_objects = ("q0", "q1", "q2")
    p_order = {("p0", "p0"), ("p0", "p1"), ("p1", "p1")}
    q_order = {("q0", "q0"), ("q0", "q1"), ("q0", "q2"), ("q1", "q1"), ("q1", "q2"), ("q2", "q2")}
    f_map = {"p0": "q0", "p1": "q2"}
    g_map = {"q0": "p0", "q1": "p0", "q2": "p1"}
    bad_g_map = {"q0": "p0", "q1": "p1", "q2": "p1"}
    rows = []
    for p in p_objects:
        for q in q_objects:
            left = (f_map[p], q) in q_order
            right = (p, g_map[q]) in p_order
            rows.append({"p": p, "q": q, "F_p_leq_q": left, "p_leq_G_q": right, "equivalent": left == right})
    bad_rows = []
    for p in p_objects:
        for q in q_objects:
            left = (f_map[p], q) in q_order
            right = (p, bad_g_map[q]) in p_order
            bad_rows.append({"p": p, "q": q, "F_p_leq_q": left, "p_leq_bad_G_q": right, "equivalent": left == right})
    return {
        "fragment": "finite_poset_adjunction_galois_connection",
        "left_adjoint_F": f_map,
        "right_adjoint_G": g_map,
        "hom_equivalence_rows": rows,
        "adjunction_pass": all(row["equivalent"] for row in rows),
        "negative_control_bad_G": bad_g_map,
        "negative_control_rejected": not all(row["equivalent"] for row in bad_rows),
        "negative_control_rows": bad_rows,
    }


def _finite_monoidal_suite() -> dict[str, Any]:
    morphisms = ("e", "a")

    def op(left: str, right: str) -> str:
        return "e" if left == right else "a"

    associativity = all(op(op(x, y), z) == op(x, op(y, z)) for x in morphisms for y in morphisms for z in morphisms)
    unit = all(op("e", x) == x and op(x, "e") == x for x in morphisms)
    interchange = all(
        op(op(f, g), op(h, k)) == op(op(f, h), op(g, k))
        for f in morphisms
        for g in morphisms
        for h in morphisms
        for k in morphisms
    )
    return {
        "fragment": "strict_one_object_symmetric_monoidal_category",
        "object": "*",
        "morphisms": list(morphisms),
        "composition": {f"{left};{right}": op(left, right) for left in morphisms for right in morphisms},
        "tensor": {f"{left}⊗{right}": op(left, right) for left in morphisms for right in morphisms},
        "associativity_pass": associativity,
        "unit_pass": unit,
        "interchange_law_pass": interchange,
        "pass": associativity and unit and interchange,
    }


def _finite_markov_blackwell_suite() -> dict[str, Any]:
    identity = [[1.0, 0.0], [0.0, 1.0]]
    noisy = [[0.82, 0.18], [0.24, 0.76]]
    postprocess = [[0.9, 0.1], [0.2, 0.8]]
    degraded = _matrix_multiply(noisy, postprocess)
    identity_to_noisy = blackwell_witness(identity, noisy)
    noisy_to_degraded = blackwell_witness(noisy, degraded)
    noisy_to_identity = blackwell_witness(noisy, identity)
    copy = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    delete = [[1.0], [1.0]]
    checks = {
        "identity_row_stochastic": _is_row_stochastic(identity),
        "noisy_row_stochastic": _is_row_stochastic(noisy),
        "degraded_row_stochastic": _is_row_stochastic(degraded),
        "copy_row_stochastic": _is_row_stochastic(copy),
        "delete_row_stochastic": _is_row_stochastic(delete),
        "identity_composition_left": _matrix_close(_matrix_multiply(identity, noisy), noisy),
        "identity_composition_right": _matrix_close(_matrix_multiply(noisy, identity), noisy),
        "copy_deterministic": all(sum(1 for value in row if abs(value - 1.0) <= 1e-9) == 1 for row in copy),
        "delete_terminal": all(len(row) == 1 and abs(row[0] - 1.0) <= 1e-9 for row in delete),
    }
    return {
        "fragment": "finite_stochastic_kernel_fragment",
        "kernels": {
            "identity": identity,
            "noisy": noisy,
            "postprocess": postprocess,
            "degraded": degraded,
        },
        "markov_checks": checks,
        "markov_category_fragment_pass": all(checks.values()),
        "blackwell_exact_witnesses": {
            "identity_dominates_noisy": identity_to_noisy,
            "noisy_dominates_degraded": noisy_to_degraded,
            "noisy_dominates_identity_negative_control": noisy_to_identity,
        },
        "blackwell_exact_witness_pass": identity_to_noisy["dominates"] and noisy_to_degraded["dominates"],
        "blackwell_negative_control_rejected": noisy_to_identity["dominates"] is False,
        "not_claimed": [
            "arbitrary measurable Markov categories",
            "all statistical experiment comparisons",
            "continuous-state Blackwell order",
        ],
    }


def blackwell_witness(left: list[list[float]], right: list[list[float]]) -> dict[str, Any]:
    if len(left) != 2 or len(left[0]) != 2 or len(right) != 2:
        return {"dominates": False, "reason": "only_2x2_square_left_kernel_supported"}
    det = left[0][0] * left[1][1] - left[0][1] * left[1][0]
    if abs(det) <= 1e-12:
        return {"dominates": False, "reason": "left_kernel_not_invertible_in_2x2_fragment"}
    inverse = [
        [left[1][1] / det, -left[0][1] / det],
        [-left[1][0] / det, left[0][0] / det],
    ]
    witness = _matrix_multiply(inverse, right, precision=12)
    reconstructed = _matrix_multiply(left, witness, precision=12)
    row_stochastic = _is_row_stochastic(witness, eps=1e-8)
    reconstructs = _matrix_close(reconstructed, right, eps=1e-8)
    return {
        "dominates": row_stochastic and reconstructs,
        "witness": _round_matrix(witness),
        "reconstructed": _round_matrix(reconstructed),
        "row_stochastic_witness": row_stochastic,
        "reconstructs_target": reconstructs,
    }


def _finite_information_geometry_suite() -> dict[str, Any]:
    p, q, r = 0.2, 0.7, 0.4
    d_pq = _fisher_rao_bernoulli(p, q)
    d_qp = _fisher_rao_bernoulli(q, p)
    d_pp = _fisher_rao_bernoulli(p, p)
    d_pr = _fisher_rao_bernoulli(p, r)
    d_rq = _fisher_rao_bernoulli(r, q)
    laws = {
        "nonnegative": d_pq >= 0.0,
        "symmetric": abs(d_pq - d_qp) <= 1e-12,
        "zero_self_distance": abs(d_pp) <= 1e-12,
        "triangle_sample": d_pq <= d_pr + d_rq + 1e-12,
    }
    return {
        "fragment": "bernoulli_fisher_rao_measurement",
        "fisher_rao": {
            "d_0_2_0_7": round(d_pq, 6),
            "d_0_7_0_2": round(d_qp, 6),
            "d_0_2_0_2": round(d_pp, 6),
            "d_0_2_0_4": round(d_pr, 6),
            "d_0_4_0_7": round(d_rq, 6),
        },
        "kl_0_2_0_7": round(_bernoulli_kl(p, q), 6),
        "kl_0_7_0_2": round(_bernoulli_kl(q, p), 6),
        "metric_laws": laws,
        "metric_laws_pass": all(laws.values()),
        "not_truth_oracle": True,
    }


def _natural_language_diagram_suite() -> dict[str, Any]:
    examples = [
        "Le Chatelier principle and Lenz law both describe a perturbation that triggers an opposing response toward equilibrium.",
        "ResNet skip connection preserves an identity path while a residual transform changes representation.",
        "Autocorrelation can suppress random noise and recover a stable signal; JEPA-style latent prediction uses a related invariance idea.",
        "I prefer a blue button because it looks nice.",
    ]
    rows = [extract_natural_language_diagram(text) for text in examples]
    extracted = [row for row in rows if row["status"] == "formalized"]
    certificate_passes = [row["certificate"]["validation"]["valid"] for row in extracted]
    return {
        "example_count": len(examples),
        "extracted_count": len(extracted),
        "not_applicable_count": sum(1 for row in rows if row["status"] == "not_applicable"),
        "certificate_pass_count": sum(1 for passed in certificate_passes if passed),
        "certificate_pass_rate": round(sum(1 for passed in certificate_passes if passed) / max(1, len(extracted)), 4),
        "negative_control_abstained": rows[-1]["status"] == "not_applicable",
        "rows": rows,
    }


def _nl_certificate(
    *,
    family: str,
    text: str,
    objects: tuple[str, str, str],
    arrows: tuple[tuple[str, str, str], tuple[str, str, str]],
    invariants: tuple[str, ...],
) -> dict[str, Any]:
    first, second = arrows
    composed = f"{first[0]}_then_{second[0]}"
    category = _arrow_category(
        category_id=f"nl_{family}",
        objects=objects,
        first=first[0],
        second=second[0],
        composed=composed,
    )
    validation = category.validate()
    obligations = {
        "objects_morphisms_typed": validation["issue_count"] == 0,
        "identity": validation["identity_law_pass"],
        "associativity": validation["associativity_pass"],
        "invariants_recorded": len(invariants) >= 2,
        "bounded_scope": True,
    }
    return {
        "text_hash": stable_hash(text),
        "status": "formalized",
        "family": family,
        "certificate": {
            "certificate_id": f"nlcert_{stable_hash([family, text])}",
            "category": _category_to_dict(category),
            "invariants": list(invariants),
            "proof_obligations": [
                {"name": name, "status": "pass" if passed else "fail"}
                for name, passed in obligations.items()
            ],
            "validation": validation,
            "scope": "bounded rule-based extraction; abstain when no supported finite pattern is detected",
        },
    }


def _is_terminal(poset: dict[str, Any], candidate: str) -> bool:
    return all(_leq(poset, obj, candidate) for obj in poset["objects"])


def _is_initial(poset: dict[str, Any], candidate: str) -> bool:
    return all(_leq(poset, candidate, obj) for obj in poset["objects"])


def _is_product(poset: dict[str, Any], left: str, right: str, candidate: str) -> bool:
    if not (_leq(poset, candidate, left) and _leq(poset, candidate, right)):
        return False
    return all(
        _leq(poset, obj, candidate)
        for obj in poset["objects"]
        if _leq(poset, obj, left) and _leq(poset, obj, right)
    )


def _is_coproduct(poset: dict[str, Any], left: str, right: str, candidate: str) -> bool:
    if not (_leq(poset, left, candidate) and _leq(poset, right, candidate)):
        return False
    return all(
        _leq(poset, candidate, obj)
        for obj in poset["objects"]
        if _leq(poset, left, obj) and _leq(poset, right, obj)
    )


def _is_pullback(poset: dict[str, Any], left: str, right: str, common: str, candidate: str) -> bool:
    return _leq(poset, left, common) and _leq(poset, right, common) and _is_product(poset, left, right, candidate)


def _is_pushout(poset: dict[str, Any], common: str, left: str, right: str, candidate: str) -> bool:
    return _leq(poset, common, left) and _leq(poset, common, right) and _is_coproduct(poset, left, right, candidate)


def _leq(poset: dict[str, Any], left: str, right: str) -> bool:
    return (left, right) in poset["leq_pairs"]


def _matrix_multiply(
    left: list[list[float]],
    right: list[list[float]],
    *,
    precision: int = 10,
) -> list[list[float]]:
    return [
        [
            round(sum(row[k] * right[k][col_index] for k in range(len(right))), precision)
            for col_index in range(len(right[0]))
        ]
        for row in left
    ]


def _matrix_close(left: list[list[float]], right: list[list[float]], *, eps: float = 1e-9) -> bool:
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        return False
    return all(abs(a - b) <= eps for row_l, row_r in zip(left, right) for a, b in zip(row_l, row_r))


def _is_row_stochastic(matrix: list[list[float]], *, eps: float = 1e-9) -> bool:
    return all(all(value >= -eps for value in row) and abs(sum(row) - 1.0) <= eps for row in matrix)


def _round_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [[round(value, 6) for value in row] for row in matrix]


def _fisher_rao_bernoulli(p: float, q: float) -> float:
    return 2.0 * abs(math.asin(math.sqrt(p)) - math.asin(math.sqrt(q)))


def _bernoulli_kl(p: float, q: float) -> float:
    return p * math.log(max(p, 1e-12) / max(q, 1e-12)) + (1 - p) * math.log(
        max(1 - p, 1e-12) / max(1 - q, 1e-12)
    )


def _category_to_dict(category: ExplicitFiniteCategory) -> dict[str, Any]:
    return {
        "category_id": category.category_id,
        "objects": list(category.objects),
        "morphisms": [
            {
                "id": morphism.morphism_id,
                "source": morphism.source,
                "target": morphism.target,
                "kind": morphism.kind,
            }
            for morphism in category.morphisms
        ],
        "composition_table": dict(category.composition_table),
    }


def _composition_key(left: str, right: str) -> str:
    return f"{left};{right}"


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build finite theorem fragment artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="finite_theorem_fragment_20260612")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_finite_theorem_fragment_payload(
        root=root,
        eval_id=args.eval_id,
        out=Path(args.out),
        write_artifact=True,
    )
    print(
        json.dumps(
            {
                "eval_id": payload["eval_id"],
                "pass": payload["pass"],
                "metrics": payload["metrics"],
                "failed_gates": payload["failed_gates"],
                "out": payload.get("artifact_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
