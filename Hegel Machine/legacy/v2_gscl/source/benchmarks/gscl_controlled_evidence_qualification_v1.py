"""Non-scoring qualification of the controlled GSCL evidence path.

This module extends the existing Phase-0 qualification lineage.  It is an
iterative implementation/source qualification, not a formal study, a public
intrinsic measurement, or efficacy evidence.  The corpus contains 10 primary
cases, 10 paired counterfactual negatives, and 5 paired missingness controls.
Its 100 runtime items are only four serialization/field-alias views of those
25 cases.

Baseline performance is descriptive and never gates qualification.  Only
executability, deterministic common-input replay, operator-receipt integrity,
and the controlled full-lane implementation contract can fail the harness.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.generalized_structural_correspondence_v1 import (
    CorrespondenceDisposition,
    ExactRational,
    LawBinding,
    ObservationStatus,
    ResidualDisposition,
    StructuralEpisode,
    build_gscl_schema_registry_v1,
    canonical_structural_signature,
    compare_structural_bindings,
    strict_canonical_bytes,
    strict_content_hash,
    validate_law_binding,
)
from assumption_agent.gscl_baselines_v1 import (
    BASELINE_CONTRACT,
    BASELINE_CONTRACT_HASH,
    build_baseline_implementation_closure,
    run_legacy_keyword_baseline,
    run_semantic_only_batch,
    verify_legacy_baseline_contract,
)
from assumption_agent.gscl_evidence_extractor_v1 import (
    EXTRACTOR_CONTRACT_HASH,
    BindingSearch,
    BoundStructuralCase,
    StructuralExtraction,
    StructuralProposal,
    bind_structural_episode,
    extract_structural_episode,
    extractor_implementation_sha256,
)
from assumption_agent.structural_law_residuals_v1 import (
    ResidualPolicy,
    build_law_residual_receipt,
    evaluate_bound_law,
)
from assumption_agent.universal_assumption_ontology_v1 import (
    build_universal_assumption_ontology_v1,
)

from .gscl_controlled_evidence_corpus_v1 import (
    CORPUS_VERSION,
    PAIRED_CONTROL_ROOT_COUNT,
    PAIRED_NEGATIVE_ROOT_COUNT,
    PRIMARY_ROOT_COUNT,
    ROOT_COUNT,
    T05,
    VIEW_COUNT,
    VIEWS_PER_ROOT,
    ControlledRoot,
    ControlledView,
    atomic_fact_tokens,
    build_controlled_roots,
    controlled_view_gold_linkage,
    gold_pack_contract,
    jaccard,
    raw_pack_contract,
    render_controlled_views,
    validate_no_runtime_answer_leak,
    validate_pair_operator_receipts,
)


QUALIFICATION_VERSION = "gscl.controlled.evidence.qualification.v3"
QUALIFICATION_CONTRACT = {
    "version": QUALIFICATION_VERSION,
    "lineage": "gscl_phase0_offline_qualification_v1",
    "scope": "controlled_atomic_fact_path",
    "new_study": False,
    "formal_result": False,
    "efficacy_evidence": False,
    "public_intrinsic_measurement": False,
    "case_composition": {
        "primary": PRIMARY_ROOT_COUNT,
        "paired_counterfactual_negative": PAIRED_NEGATIVE_ROOT_COUNT,
        "paired_missingness_control": PAIRED_CONTROL_ROOT_COUNT,
        "paired_law_groups": 5,
    },
    "serialization_view_count": VIEW_COUNT,
    "serialization_views_per_case": VIEWS_PER_ROOT,
    "serialization_view_semantics": (
        "serialization_and_field_alias_invariance_only"
    ),
    "natural_language_paraphrase_claimed": False,
    "narrative_overlap_claimed": False,
    "lanes": [
        "extractor_only",
        "bounded_csp_binder",
        "end_to_end_residual_verified",
        "semantic_only",
        "legacy_keyword_structural_morphism",
        "flat_label_no_verifier",
    ],
    "controlled_full_lane_checks": [
        "ten_primary_cases",
        "ten_paired_counterfactual_negatives",
        "five_paired_missingness_controls",
        "fifteen_operator_receipts",
        "exact_candidate_family",
        "exact_role_binding",
        "value_less_required_output_bound_unknown",
        "exact_value_never_laundered_unknown",
        "exact_observable_denotation",
        "exact_quantity_and_relation_denotation",
        "exact_utf8_atomic_evidence_spans",
        "complete_inference_provenance",
        "four_view_serialization_invariance",
        "five_primary_law_group_correspondences",
        "ten_paired_negative_rejections",
        "same_process_byte_exact_replay",
    ],
    "baseline_acceptance": (
        "executable_deterministic_common_input_only"
    ),
    "baseline_performance_is_effect_gate": False,
    "declared_capability_surface": {
        "controlled_synthetic_corpus_access": True,
        "frozen_local_model_access": True,
        "external_benchmark_source_access": False,
        "network_access": False,
        "api_access": False,
        "online_evaluator_access": False,
        "validation_access": False,
        "test_access": False,
    },
    "runtime_access_audited": False,
}
QUALIFICATION_CONTRACT_HASH = strict_content_hash(
    QUALIFICATION_CONTRACT
)


@dataclass(frozen=True)
class _ViewOutcome:
    view: ControlledView
    view_kind: str
    root: ControlledRoot
    extraction: StructuralExtraction
    binding_search: BindingSearch
    bound_case: BoundStructuralCase | None
    actual_disposition: str
    actual_decision: str
    denotation_hash: str | None
    issue_ids: tuple[str, ...]

    def replay_payload(self) -> dict[str, Any]:
        return {
            "source_sha256": self.view.source_sha256,
            "candidate_law_ids": list(
                self.extraction.candidate_law_ids
            ),
            "base_episode_hash": (
                None
                if self.extraction.base_episode is None
                else self.extraction.base_episode.episode_hash
            ),
            "bound_episode_hash": (
                None
                if self.bound_case is None
                else self.bound_case.episode.episode_hash
            ),
            "binding_hash": (
                None
                if self.bound_case is None
                else self.bound_case.binding.binding_hash
            ),
            "assignment_count": self.binding_search.assignment_count,
            "actual_disposition": self.actual_disposition,
            "actual_decision": self.actual_decision,
            "denotation_hash": self.denotation_hash,
            "issue_ids": list(self.issue_ids),
        }


def _policy(root: ControlledRoot) -> ResidualPolicy:
    return ResidualPolicy(
        law_id=root.law_id,
        relation_threshold=(
            ExactRational(1)
            if root.law_id == T05
            else ExactRational(0)
        ),
    )


def _target_id(target: Any) -> str:
    for field in (
        "object_id",
        "relation_id",
        "quantity_id",
        "hyperrelation_id",
        "constraint_id",
    ):
        value = getattr(target, field, None)
        if isinstance(value, str):
            return value
    raise TypeError("target does not have an id")


def _record_ids(
    root: ControlledRoot,
    kind: str,
    *,
    predicate: str | None = None,
) -> set[str]:
    result = set()
    for record in root.records:
        if record["kind"] != kind:
            continue
        if (
            predicate is not None
            and record["attrs"].get("predicate") != predicate
        ):
            continue
        result.add(str(record["id"]))
    return result


def _observable_support_ids(
    root: ControlledRoot, observable_id: str
) -> set[str]:
    fixed = {
        "gscl.v1.t14_finite_equivariance": {
            "input_action": {"map.alpha"},
            "output_action": {"map.beta"},
            "outputs_before": {"observation.alpha"},
            "outputs_after": {"observation.beta"},
        },
        "gscl.v1.t17_monotone_order": {
            "comparable_output_pairs": {
                "observation.alpha",
                "observation.beta",
            },
            "declared_direction": {"assertion.gamma"},
        },
        "gscl.v1.t09_path_composition": {
            "finite_domain": {"assertion.gamma"},
            "first_map": {"map.alpha"},
            "second_map": {"map.beta"},
            "direct_map": {"map.gamma"},
        },
        "gscl.v1.t05_pair_interaction": {
            "components": {
                "node.alpha",
                "node.beta",
                "node.gamma",
            },
            "designated_pair": {"assertion.gamma"},
            "interaction_expectation": {
                "assertion.delta"
            },
        },
    }
    if root.law_id == "gscl.v1.t15_closed_balance":
        if observable_id == "boundary_declaration":
            return {"assertion.gamma"}
        if observable_id == "quantity_ledger":
            return (
                set(root.quantity_expectations)
                | {
                    str(row["id"])
                    for row in root.records
                    if row["kind"] == "transfer"
                    and not str(row["id"]).endswith(".decoy")
                }
            )
    if (
        root.law_id == "gscl.v1.t05_pair_interaction"
        and observable_id == "held_fold_utilities"
    ):
        return {
            str(row["id"])
            for row in root.records
            if row["kind"] == "subset_outcome"
            and not str(row["id"]).endswith(".decoy")
        }
    return set(fixed[root.law_id][observable_id])


def _span_offsets(
    episode: StructuralEpisode, span_ids: Sequence[str]
) -> set[tuple[int, int]]:
    by_id = {
        span.span_id: (span.start_byte, span.end_byte)
        for span in episode.evidence_spans
    }
    return {by_id[span_id] for span_id in span_ids}


def _gold_denotation_issues(
    root: ControlledRoot,
    view: ControlledView,
    episode: StructuralEpisode,
    binding: LawBinding,
    proposal: StructuralProposal,
    registry: Any,
) -> tuple[str, ...]:
    issues: list[str] = []
    role_map = {
        row.role_id: row.target_id for row in binding.role_bindings
    }
    if role_map != dict(root.role_targets):
        issues.append("controlled_role_binding_mismatch")

    expected_target_sets = {
        "object": _record_ids(root, "node"),
        "relation": _record_ids(root, "edge"),
        "quantity": {
            str(row["id"])
            for row in root.records
            if row["kind"] == "observation"
            and "values" not in row["attrs"]
        },
        "hyperrelation": _record_ids(
            root, "assertion", predicate="association"
        ),
        "constraint": {"constraint.derived"},
    }
    actual_target_sets = {
        "object": {_target_id(row) for row in episode.objects},
        "relation": {
            _target_id(row) for row in episode.relations
        },
        "quantity": {
            _target_id(row) for row in episode.quantities
        },
        "hyperrelation": {
            _target_id(row) for row in episode.hyperrelations
        },
        "constraint": {
            _target_id(row) for row in episode.constraints
        },
    }
    if actual_target_sets != expected_target_sets:
        issues.append("controlled_target_inventory_mismatch")

    for observable_id, expected in root.observable_expectations.items():
        try:
            actual = episode.require_observable(observable_id)
        except KeyError:
            issues.append("controlled_observable_missing")
            continue
        actual_payload = {
            "value_type": actual.value_type.value,
            "payload": actual.value_payload,
            "status": actual.observation_status.value,
            "dimension": actual.dimension,
            "unit": actual.unit,
        }
        if actual_payload != dict(expected):
            issues.append("controlled_observable_mismatch")
    if {
        row.observable_id for row in episode.observables
    } != set(root.observable_expectations):
        issues.append("controlled_observable_inventory_mismatch")

    quantities = {
        row.quantity_id: row for row in episode.quantities
    }
    for quantity_id, expected in root.quantity_expectations.items():
        actual = quantities.get(quantity_id)
        if actual is None:
            issues.append("controlled_quantity_missing")
            continue
        actual_payload = {
            "owner": actual.owner_object_id,
            "value": actual.value.safe_payload(),
            "dimension": actual.dimension,
            "unit": actual.unit,
        }
        if actual_payload != dict(expected):
            issues.append("controlled_quantity_mismatch")
    relations = {
        row.relation_id: row for row in episode.relations
    }
    for relation_id, expected in root.relation_expectations.items():
        actual = relations.get(relation_id)
        if actual is None:
            issues.append("controlled_relation_missing")
            continue
        actual_payload = {
            "type": actual.relation_type,
            "source": actual.source_object_id,
            "target": actual.target_object_id,
            "order": actual.order_index,
        }
        if actual_payload != dict(expected):
            issues.append("controlled_relation_mismatch")

    atomic_targets: tuple[Any, ...] = (
        *episode.objects,
        *episode.relations,
        *episode.quantities,
        *episode.hyperrelations,
    )
    for target in atomic_targets:
        target_id = _target_id(target)
        expected_atomic_offsets = (
            set()
            if target.observation_status
            is ObservationStatus.UNKNOWN
            else {view.record_spans[target_id]}
        )
        if _span_offsets(
            episode, target.evidence_span_ids
        ) != expected_atomic_offsets:
            issues.append("controlled_atomic_span_mismatch")

    for constraint in episode.constraints:
        expected_span_ids = {
            *proposal.field_span_ids,
            *(
                span_id
                for row in binding.role_bindings
                if row.target_id != constraint.constraint_id
                for span_id in row.evidence_span_ids
            ),
            *(
                span_id
                for observable in episode.observables
                for span_id in observable.evidence_span_ids
            ),
        }
        if set(constraint.evidence_span_ids) != expected_span_ids:
            issues.append("controlled_constraint_span_mismatch")

    for observable in episode.observables:
        expected = root.observable_expectations[
            observable.observable_id
        ]
        expected_offsets = (
            set()
            if expected["status"] == "unknown"
            else {
                view.record_spans[record_id]
                for record_id in _observable_support_ids(
                    root, observable.observable_id
                )
            }
        )
        if _span_offsets(
            episode, observable.evidence_span_ids
        ) != expected_offsets:
            issues.append("controlled_observable_span_mismatch")

    inferred_items: tuple[Any, ...] = (
        *episode.objects,
        *episode.relations,
        *episode.quantities,
        *episode.hyperrelations,
        *episode.constraints,
        *episode.observables,
    )
    for item in inferred_items:
        if item.observation_status is ObservationStatus.INFERRED:
            provenance = item.inference_provenance
            if provenance is None or provenance.validate():
                issues.append(
                    "controlled_inference_provenance_incomplete"
                )
        elif item.observation_status is ObservationStatus.UNKNOWN:
            if (
                item.inference_provenance is not None
                or item.evidence_span_ids
            ):
                issues.append(
                    "controlled_unknown_evidence_contract_invalid"
                )

    schema = registry.require_law(root.law_id)
    signature = canonical_structural_signature(
        registry, schema, episode, binding
    )
    if signature["law_id"] != root.law_id:
        issues.append("controlled_signature_law_mismatch")
    return tuple(sorted(set(issues)))


def _denotation_hash(
    root: ControlledRoot,
    episode: StructuralEpisode,
    binding: LawBinding,
    registry: Any,
) -> str:
    schema = registry.require_law(root.law_id)
    observables = [
        {
            "observable_id": row.observable_id,
            "value_type": row.value_type.value,
            "value_payload": row.value_payload,
            "observation_status": row.observation_status.value,
            "dimension": row.dimension,
            "unit": row.unit,
        }
        for row in sorted(
            episode.observables,
            key=lambda item: item.observable_id,
        )
    ]
    quantities = [
        {
            "quantity_role": next(
                role.role_id
                for role in binding.role_bindings
                if role.target_id == row.quantity_id
            ),
            "owner_role": next(
                role.role_id
                for role in binding.role_bindings
                if role.target_id == row.owner_object_id
            ),
            "dimension": row.dimension,
            "unit": row.unit,
            "value": (
                None
                if row.value is None
                else row.value.safe_payload()
            ),
            "observation_status": row.observation_status.value,
        }
        for row in sorted(
            (
                row
                for row in episode.quantities
                if row.quantity_id
                in {
                    role.target_id
                    for role in binding.role_bindings
                }
            ),
            key=lambda item: item.quantity_id,
        )
    ]
    return strict_content_hash(
        {
            "structural_signature": canonical_structural_signature(
                registry, schema, episode, binding
            ),
            "observables": observables,
            "quantities": quantities,
        }
    )


def _run_view(
    root: ControlledRoot,
    view: ControlledView,
    view_kind: str,
    registry: Any,
) -> _ViewOutcome:
    issues: list[str] = []
    extraction = extract_structural_episode(
        view.source_bytes, view.media_type, registry=registry
    )
    issues.extend(extraction.issue_ids)
    if extraction.base_episode is None:
        issues.append("controlled_base_episode_missing")
    if extraction.candidate_law_ids != (root.law_id,):
        issues.append("controlled_candidate_law_mismatch")

    binding_search = bind_structural_episode(registry, extraction)
    issues.extend(binding_search.issue_ids)
    bound_case = (
        binding_search.bound_cases[0]
        if len(binding_search.bound_cases) == 1
        else None
    )
    if bound_case is None:
        issues.append("controlled_unique_binding_failed")

    disposition = "unavailable"
    decision = "unavailable"
    denotation_hash = None
    if bound_case is not None:
        issues.extend(
            _gold_denotation_issues(
                root,
                view,
                bound_case.episode,
                bound_case.binding,
                bound_case.proposal,
                registry,
            )
        )
        denotation_hash = _denotation_hash(
            root,
            bound_case.episode,
            bound_case.binding,
            registry,
        )
        try:
            evaluation = evaluate_bound_law(
                registry,
                registry.require_law(root.law_id),
                bound_case.episode,
                bound_case.binding,
                _policy(root),
            )
            disposition = evaluation.disposition.value
            decision = (
                "accepted"
                if evaluation.disposition
                is ResidualDisposition.SATISFIED
                else (
                    "abstain"
                    if evaluation.disposition
                    in {
                        ResidualDisposition.INCONCLUSIVE,
                        ResidualDisposition.NOT_APPLICABLE,
                    }
                    else "rejected"
                )
            )
        except PermissionError:
            disposition = "rejected"
            decision = "rejected"
        except Exception as exc:  # pragma: no cover - safe failure receipt
            disposition = "error"
            decision = "error"
            issues.append(
                "controlled_unexpected_evaluation_exception."
                + type(exc).__name__
            )
    if disposition != root.expected_disposition:
        issues.append("controlled_disposition_mismatch")
    if decision != root.expected_decision:
        issues.append("controlled_decision_mismatch")
    return _ViewOutcome(
        view=view,
        view_kind=view_kind,
        root=root,
        extraction=extraction,
        binding_search=binding_search,
        bound_case=bound_case,
        actual_disposition=disposition,
        actual_decision=decision,
        denotation_hash=denotation_hash,
        issue_ids=tuple(sorted(set(issues))),
    )


def _run_all_views(
    roots: Sequence[ControlledRoot],
    views: Sequence[ControlledView],
    registry: Any,
) -> tuple[_ViewOutcome, ...]:
    roots_by_id = {root.root_id: root for root in roots}
    links_by_item = {
        row.item_id: row
        for row in controlled_view_gold_linkage(roots)
    }
    return tuple(
        _run_view(
            roots_by_id[links_by_item[view.item_id].root_id],
            view,
            links_by_item[view.item_id].view_kind,
            registry,
        )
        for view in views
    )


def _evidence_ids(
    episode: StructuralEpisode, binding: LawBinding
) -> tuple[str, ...]:
    values = {
        span_id
        for row in binding.role_bindings
        for span_id in row.evidence_span_ids
    }
    for row in binding.observable_bindings:
        values.update(
            episode.require_observable(
                row.observable_id
            ).evidence_span_ids
        )
    return tuple(sorted(values))


def _residual_receipt(
    outcome: _ViewOutcome,
    registry: Any,
) -> Any:
    assert outcome.bound_case is not None
    return build_law_residual_receipt(
        registry,
        registry.require_law(outcome.root.law_id),
        outcome.bound_case.episode,
        outcome.bound_case.binding,
        _policy(outcome.root),
        receipt_id=f"receipt.{outcome.root.root_id}",
        evidence_span_ids=_evidence_ids(
            outcome.bound_case.episode,
            outcome.bound_case.binding,
        ),
    )


def _correspondence_diagnostics(
    outcomes: Sequence[_ViewOutcome],
    registry: Any,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    issues: list[str] = []
    canonical = {
        outcome.root.root_id: outcome
        for outcome in outcomes
        if outcome.view_kind == "json_canonical"
    }
    roots_by_law: dict[str, dict[str, _ViewOutcome]] = {}
    for outcome in canonical.values():
        roots_by_law.setdefault(outcome.root.law_id, {})[
            outcome.root.root_kind
        ] = outcome

    primary_accepted = 0
    negative_rejected = 0
    internal_rows = []
    for law_id, family in sorted(roots_by_law.items()):
        left = family["primary_a"]
        right = family["primary_b"]
        if left.bound_case is None or right.bound_case is None:
            issues.append(
                "controlled_primary_correspondence_input_missing"
            )
            continue
        correspondence = compare_structural_bindings(
            registry,
            registry.require_law(law_id),
            left.bound_case.episode,
            left.bound_case.binding,
            _residual_receipt(left, registry),
            right.bound_case.episode,
            right.bound_case.binding,
            _residual_receipt(right, registry),
            correspondence_id=(
                "correspondence.primary."
                + strict_content_hash(law_id)[:16]
            ),
            source_policy=_policy(left.root),
            target_policy=_policy(right.root),
        )
        accepted = (
            correspondence.disposition
            is CorrespondenceDisposition.ACCEPTED
        )
        primary_accepted += accepted
        if not accepted:
            issues.append(
                "controlled_primary_correspondence_failed"
            )

        negative_results = []
        for primary_key, negative_key in (
            ("primary_a", "hard_negative_a"),
            ("primary_b", "hard_negative_b"),
        ):
            primary = family[primary_key]
            negative = family[negative_key]
            rejected = False
            if (
                negative.actual_disposition == "violated"
                and negative.bound_case is not None
                and primary.bound_case is not None
            ):
                result = compare_structural_bindings(
                    registry,
                    registry.require_law(law_id),
                    primary.bound_case.episode,
                    primary.bound_case.binding,
                    _residual_receipt(primary, registry),
                    negative.bound_case.episode,
                    negative.bound_case.binding,
                    _residual_receipt(negative, registry),
                    correspondence_id=(
                        "correspondence.negative."
                        + strict_content_hash(
                            {
                                "law_id": law_id,
                                "negative_kind": negative_key,
                            }
                        )[:16]
                    ),
                    source_policy=_policy(primary.root),
                    target_policy=_policy(negative.root),
                )
                rejected = (
                    result.disposition
                    is CorrespondenceDisposition.REJECTED
                )
            negative_rejected += rejected
            if not rejected:
                issues.append(
                    "controlled_negative_correspondence_not_rejected"
                )
            negative_results.append(rejected)
        internal_rows.append(
            {
                "law_commitment": strict_content_hash(law_id),
                "primary_accepted": accepted,
                "paired_negative_rejections": negative_results,
            }
        )
    return (
        {
            "paired_law_group_count": len(roots_by_law),
            "primary_correspondence_groups_accepted": (
                primary_accepted
            ),
            "primary_correspondence_group_total": 5,
            "paired_negative_correspondences_rejected": (
                negative_rejected
            ),
            "paired_negative_correspondence_total": 10,
            "diagnostic_commitment": strict_content_hash(
                internal_rows
            ),
            "atomic_fact_overlap_is_not_a_gate": True,
        },
        tuple(sorted(set(issues))),
    )


def _overlap_summary(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {"count": 0, "minimum": 0, "maximum": 0, "mean": 0}
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) // len(values),
    }


def _construction_diagnostics(
    roots: Sequence[ControlledRoot],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    issues: list[str] = []
    roots_by_id = {root.root_id: root for root in roots}
    primary = [
        root for root in roots if root.pair_role == "primary"
    ]
    negatives = [
        root
        for root in roots
        if root.pair_role == "counterfactual_negative"
    ]
    controls = [
        root
        for root in roots
        if root.pair_role == "missingness_control"
    ]
    law_groups = {root.law_id for root in roots}
    if (
        len(primary) != PRIMARY_ROOT_COUNT
        or len(negatives) != PAIRED_NEGATIVE_ROOT_COUNT
        or len(controls) != PAIRED_CONTROL_ROOT_COUNT
        or len(law_groups) != 5
    ):
        issues.append("controlled_case_composition_invalid")

    receipt_issues = validate_pair_operator_receipts(roots)
    issues.extend(receipt_issues)
    paired = negatives + controls
    valid_receipt_count = 0
    receipt_hashes = []
    for root in paired:
        primary_root = roots_by_id.get(
            root.paired_primary_root_id or ""
        )
        if primary_root is not None and not validate_pair_operator_receipts(
            (primary_root, root)
        ):
            valid_receipt_count += 1
        if root.operator_receipt is not None:
            receipt_hashes.append(root.operator_receipt.receipt_hash)

    primary_overlaps = []
    by_law: dict[str, dict[str, ControlledRoot]] = {}
    for root in roots:
        by_law.setdefault(root.law_id, {})[
            root.root_kind
        ] = root
    for family in by_law.values():
        primary_overlaps.append(
            int(
                round(
                    jaccard(
                        atomic_fact_tokens(family["primary_a"]),
                        atomic_fact_tokens(family["primary_b"]),
                    )
                    * 1_000_000
                )
            )
        )

    negative_overlaps = [
        int(
            round(
                jaccard(
                    atomic_fact_tokens(
                        roots_by_id[root.paired_primary_root_id or ""]
                    ),
                    atomic_fact_tokens(root),
                )
                * 1_000_000
            )
        )
        for root in negatives
    ]
    control_overlaps = [
        int(
            round(
                jaccard(
                    atomic_fact_tokens(
                        roots_by_id[root.paired_primary_root_id or ""]
                    ),
                    atomic_fact_tokens(root),
                )
                * 1_000_000
            )
        )
        for root in controls
    ]
    return (
        {
            "case_count": len(roots),
            "case_count_semantics": (
                "10_primary_plus_10_paired_negative_plus_"
                "5_paired_missingness_control"
            ),
            "primary_case_count": len(primary),
            "paired_negative_case_count": len(negatives),
            "paired_missingness_control_count": len(controls),
            "paired_law_group_count": len(law_groups),
            "operator_receipt_count": len(receipt_hashes),
            "valid_operator_receipt_count": valid_receipt_count,
            "operator_receipt_commitment": strict_content_hash(
                sorted(receipt_hashes)
            ),
            "serialization_view_count": VIEW_COUNT,
            "serialization_views_per_case": VIEWS_PER_ROOT,
            "serialization_views_are_not_independent_cases": True,
            "natural_language_paraphrase_claimed": False,
            "atomic_fact_overlap_microunits": {
                "primary_pair_groups": _overlap_summary(
                    primary_overlaps
                ),
                "paired_counterfactual_negatives": _overlap_summary(
                    negative_overlaps
                ),
                "paired_missingness_controls": _overlap_summary(
                    control_overlaps
                ),
            },
            "atomic_fact_overlap_is_descriptive_only": True,
        },
        tuple(sorted(set(issues))),
    )


ExtractorCallable = Callable[..., StructuralExtraction]
BinderCallable = Callable[..., BindingSearch]


def run_flat_label_no_verifier_lane(
    *,
    roots: Sequence[ControlledRoot],
    views: Sequence[ControlledView],
    registry: Any,
    extractor_fn: ExtractorCallable = extract_structural_episode,
    binder_fn: BinderCallable = bind_structural_episode,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Independently rerun extractor+binder without any law residual."""

    issues: list[str] = []
    roots_by_id = {root.root_id: root for root in roots}
    links_by_item = {
        row.item_id: row
        for row in controlled_view_gold_linkage(roots)
    }
    rows = []
    for view in views:
        root = roots_by_id[links_by_item[view.item_id].root_id]
        extraction = extractor_fn(
            view.source_bytes,
            view.media_type,
            registry=registry,
        )
        search = binder_fn(registry, extraction)
        case = (
            search.bound_cases[0]
            if len(search.bound_cases) == 1
            else None
        )
        generic_issues: tuple[str, ...] = ()
        if case is not None:
            generic_issues = validate_law_binding(
                registry,
                registry.require_law(case.binding.law_id),
                case.episode,
                case.binding,
            )
        accepted = (
            case is not None
            and not extraction.issue_ids
            and not search.issue_ids
            and not case.episode.validate()
            and not generic_issues
        )
        expected_missing_required_role = (
            root.root_id == "root.t14.control"
            and search.issue_ids == ("binder_no_valid_binding",)
        )
        if (
            extraction.issue_ids
            or (search.issue_ids and not expected_missing_required_role)
            or generic_issues
        ):
            issues.append("flat_lane_execution_contract_failed")
        rows.append(
            {
                "source_sha256": hashlib.sha256(
                    view.source_bytes
                ).hexdigest(),
                "predicted_law_id": (
                    None if case is None else case.binding.law_id
                ),
                "gold_law_id": root.law_id,
                "pair_role": root.pair_role,
                "accepted": accepted,
            }
        )

    def _count(role: str, *, accepted: bool) -> int:
        return sum(
            row["pair_role"] == role
            and row["accepted"] is accepted
            for row in rows
        )

    return (
        {
            "evaluation_unit": "serialization_view_descriptive_only",
            "common_input_serialization_view_count": len(rows),
            "independent_extractor_binder_reexecution": True,
            "extractor_success_serialization_views": sum(
                row["predicted_law_id"] is not None for row in rows
            ),
            "family_correct_serialization_views": sum(
                row["predicted_law_id"] == row["gold_law_id"]
                for row in rows
            ),
            "family_total_serialization_views": len(rows),
            "accepted_serialization_views": sum(
                row["accepted"] for row in rows
            ),
            "primary_accepts": _count(
                "primary", accepted=True
            ),
            "primary_serialization_view_total": (
                PRIMARY_ROOT_COUNT * VIEWS_PER_ROOT
            ),
            "paired_negative_false_accepts": _count(
                "counterfactual_negative", accepted=True
            ),
            "paired_negative_serialization_view_total": (
                PAIRED_NEGATIVE_ROOT_COUNT * VIEWS_PER_ROOT
            ),
            "paired_missingness_false_accepts": _count(
                "missingness_control", accepted=True
            ),
            "paired_missingness_serialization_view_total": (
                PAIRED_CONTROL_ROOT_COUNT * VIEWS_PER_ROOT
            ),
            "prediction_commitment": strict_content_hash(rows),
            "residual_or_hard_negative_verifier_called": False,
        },
        tuple(sorted(set(issues))),
    )


def _baseline_diagnostics(
    roots: Sequence[ControlledRoot],
    views: Sequence[ControlledView],
    registry: Any,
    ontology: Any,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    issues: list[str] = list(verify_legacy_baseline_contract())
    implementation_closure_before, closure_issues = (
        build_baseline_implementation_closure()
    )
    issues.extend(closure_issues)
    roots_by_id = {root.root_id: root for root in roots}
    links_by_item = {
        row.item_id: row
        for row in controlled_view_gold_linkage(roots)
    }
    lossless_count = 0
    item_texts: dict[str, str] = {}
    for view in views:
        text = view.source_bytes.decode("utf-8", errors="strict")
        if text.encode("utf-8") == view.source_bytes:
            lossless_count += 1
        else:
            issues.append("baseline_lossless_utf8_roundtrip_failed")
        item_texts[view.view_id] = text

    canonical_by_root = {
        links_by_item[view.item_id].root_id: view
        for view in views
        if links_by_item[view.item_id].view_kind == "json_canonical"
    }
    pair_keys = []
    by_law: dict[str, dict[str, ControlledRoot]] = {}
    for root in roots:
        by_law.setdefault(root.law_id, {})[
            root.root_kind
        ] = root
    for law_id, family in sorted(by_law.items()):
        left = canonical_by_root[family["primary_a"].root_id]
        right = canonical_by_root[family["primary_b"].root_id]
        pair_keys.append(
            (
                "pair."
                + strict_content_hash(
                    {"law_id": law_id, "kind": "primary"}
                )[:24],
                left.view_id,
                right.view_id,
            )
        )
    for root in roots:
        if root.pair_role == "primary":
            continue
        left = canonical_by_root[root.paired_primary_root_id or ""]
        right = canonical_by_root[root.root_id]
        pair_keys.append(
            (
                "pair."
                + strict_content_hash(root.pair_id or "")[:24],
                left.view_id,
                right.view_id,
            )
        )

    semantic = run_semantic_only_batch(
        item_texts=item_texts,
        pair_keys=pair_keys,
        registry=registry,
        ontology=ontology,
    )
    if not semantic.actual_chunk_batch_replay_exact:
        issues.append(
            "semantic_baseline_actual_chunk_batch_replay_failed"
        )
    if (
        not semantic.full_token_coverage
        or semantic.truncated_chunk_count != 0
        or semantic.maximum_chunk_token_count
        > semantic.maximum_sequence_length
    ):
        issues.append("semantic_baseline_source_coverage_failed")
    semantic_by_item = {
        row.item_key: row for row in semantic.predictions
    }
    semantic_correct = sum(
        semantic_by_item[view.view_id].predicted_law_id
        == roots_by_id[
            links_by_item[view.item_id].root_id
        ].law_id
        for view in views
    )
    semantic_serialization_consistent = sum(
        len(
            {
                semantic_by_item[view.view_id].predicted_law_id
                for view in views
                if links_by_item[view.item_id].root_id
                == root.root_id
            }
        )
        == 1
        for root in roots
    )

    legacy_execution_audit: dict[str, int] = {}
    legacy_first = run_legacy_keyword_baseline(
        item_texts, execution_audit=legacy_execution_audit
    )
    legacy_second = run_legacy_keyword_baseline(
        item_texts, execution_audit=legacy_execution_audit
    )
    expected_legacy_compute_calls = 2 * len(item_texts)
    actual_legacy_compute_calls = legacy_execution_audit.get(
        "prediction_compute_calls", 0
    )
    if actual_legacy_compute_calls != expected_legacy_compute_calls:
        issues.append(
            "legacy_baseline_actual_recomputation_failed"
        )
    if strict_canonical_bytes(
        [row.safe_payload() for row in legacy_first]
    ) != strict_canonical_bytes(
        [row.safe_payload() for row in legacy_second]
    ):
        issues.append("legacy_baseline_replay_failed")
    legacy_by_item = {
        row.item_key: row for row in legacy_first
    }
    legacy_mapped_correct = sum(
        legacy_by_item[view.view_id].predicted_law_id
        == roots_by_id[
            links_by_item[view.item_id].root_id
        ].law_id
        for view in views
    )
    legacy_accepted = [
        row for row in legacy_first if row.accepted
    ]
    legacy_accepted_correct = sum(
        row.predicted_law_id
        == roots_by_id[
            next(
                links_by_item[view.item_id].root_id
                for view in views
                if view.view_id == row.item_key
            )
        ].law_id
        for row in legacy_accepted
    )
    legacy_serialization_consistent = sum(
        len(
            {
                legacy_by_item[view.view_id].predicted_law_id
                for view in views
                if links_by_item[view.item_id].root_id
                == root.root_id
            }
        )
        == 1
        for root in roots
    )
    compatible_laws = set(
        BASELINE_CONTRACT["legacy_keyword"][
            "compatible_law_map"
        ].values()
    )
    registry_laws = {row.law_id for row in registry.schemas}

    flat, flat_issues = run_flat_label_no_verifier_lane(
        roots=roots,
        views=views,
        registry=registry,
    )
    issues.extend(flat_issues)

    implementation_closure_after, closure_after_issues = (
        build_baseline_implementation_closure()
    )
    issues.extend(closure_after_issues)
    if (
        implementation_closure_before["closure_hash"]
        != implementation_closure_after["closure_hash"]
        or strict_canonical_bytes(implementation_closure_before)
        != strict_canonical_bytes(implementation_closure_after)
    ):
        issues.append(
            "baseline_implementation_closure_changed_during_run"
        )

    semantic_payload = semantic.safe_payload()
    common_input_rows = [
        {
            "source_sha256": hashlib.sha256(
                view.source_bytes
            ).hexdigest(),
            "source_size": len(view.source_bytes),
        }
        for view in sorted(views, key=lambda row: row.view_id)
    ]
    return (
        {
            "evaluation_unit": "serialization_view_descriptive_only",
            "common_input_serialization_view_count": len(views),
            "lossless_utf8_roundtrip_count": lossless_count,
            "common_input_hash_algorithm": "sha256_raw_bytes",
            "common_input_commitment": strict_content_hash(
                common_input_rows
            ),
            "semantic_only": {
                "family_correct_serialization_views": semantic_correct,
                "family_total_serialization_views": len(views),
                "serialization_consistent_cases": (
                    semantic_serialization_consistent
                ),
                "serialization_case_total": len(roots),
                "prototype_hash": semantic.prototype_hash,
                "embedding_matrix_hash": (
                    semantic.embedding_matrix_hash
                ),
                "chunk_embedding_matrix_hash": (
                    semantic.chunk_embedding_matrix_hash
                ),
                "runtime_receipt_hash": (
                    semantic.runtime_receipt_hash
                ),
                "canary_receipt_hash": (
                    semantic.canary_receipt_hash
                ),
                "prediction_commitment": strict_content_hash(
                    semantic_payload["predictions"]
                ),
                "pair_count": len(pair_keys),
                "pair_similarity_commitment": strict_content_hash(
                    semantic_payload["pair_similarities"]
                ),
                "actual_chunk_batch_replay_exact": (
                    semantic.actual_chunk_batch_replay_exact
                ),
                "maximum_sequence_length": (
                    semantic.maximum_sequence_length
                ),
                "source_text_count": semantic.source_text_count,
                "source_texts_requiring_chunking": (
                    semantic.source_texts_requiring_chunking
                ),
                "source_chunk_count": semantic.source_chunk_count,
                "maximum_chunk_token_count": (
                    semantic.maximum_chunk_token_count
                ),
                "truncated_chunk_count": (
                    semantic.truncated_chunk_count
                ),
                "full_token_coverage": (
                    semantic.full_token_coverage
                ),
                "chunk_plan_commitment": (
                    semantic.chunk_plan_commitment
                ),
                "threshold_tuned": False,
            },
            "legacy_keyword": {
                "mapped_top1_accuracy": {
                    "correct": legacy_mapped_correct,
                    "total": len(views),
                },
                "mapped_top1_prediction_count": sum(
                    row.predicted_law_id is not None
                    for row in legacy_first
                ),
                "accepted_accuracy": {
                    "correct": legacy_accepted_correct,
                    "total": len(legacy_accepted),
                },
                "serialization_consistent_cases": (
                    legacy_serialization_consistent
                ),
                "serialization_case_total": len(roots),
                "compatible_law_family_coverage": {
                    "covered": len(compatible_laws & registry_laws),
                    "total": len(registry_laws),
                },
                "prediction_commitment": strict_content_hash(
                    [
                        row.safe_payload()
                        for row in legacy_first
                    ]
                ),
                "same_process_replay_exact": (
                    "legacy_baseline_replay_failed" not in issues
                ),
                "actual_recomputation_verified": (
                    "legacy_baseline_actual_recomputation_failed"
                    not in issues
                ),
                "actual_prediction_compute_count": (
                    actual_legacy_compute_calls
                ),
                "expected_prediction_compute_count": (
                    expected_legacy_compute_calls
                ),
                "new_markers_added": False,
            },
            "flat_label_no_verifier": flat,
            "implementation_closure": (
                implementation_closure_after
            ),
            "implementation_closure_stable_during_run": (
                "baseline_implementation_closure_changed_during_run"
                not in issues
            ),
            "baseline_performance_is_qualification_gate": False,
            "baseline_performance_is_effect_gate": False,
            "descriptive_performance_values_are_not_acceptance_checks": (
                True
            ),
        },
        tuple(sorted(set(issues))),
    )


def _safe_issue_class(issue: str) -> str:
    """Collapse any private case/view suffix into an aggregate class."""

    return issue.split(".", 1)[0]


def run_controlled_evidence_qualification() -> dict[str, Any]:
    """Run the controlled 25-case/100-serialization-view harness."""

    ontology = build_universal_assumption_ontology_v1()
    registry = build_gscl_schema_registry_v1(ontology)
    roots = build_controlled_roots()
    views = render_controlled_views(roots)
    raw_contract = raw_pack_contract(views)
    gold_contract = gold_pack_contract(roots)

    issues: list[str] = [
        _safe_issue_class(issue)
        for issue in validate_no_runtime_answer_leak(views)
    ]
    construction, construction_issues = _construction_diagnostics(
        roots
    )
    issues.extend(
        _safe_issue_class(issue)
        for issue in construction_issues
    )

    first = _run_all_views(roots, views, registry)
    second = _run_all_views(roots, views, registry)
    first_payload = [
        outcome.replay_payload() for outcome in first
    ]
    second_payload = [
        outcome.replay_payload() for outcome in second
    ]
    if strict_canonical_bytes(first_payload) != strict_canonical_bytes(
        second_payload
    ):
        issues.append("controlled_same_process_replay_failed")
    issues.extend(
        _safe_issue_class(issue)
        for outcome in first
        for issue in outcome.issue_ids
    )

    case_denotation_hashes: dict[str, set[str | None]] = {}
    case_dispositions: dict[str, set[str]] = {}
    for outcome in first:
        case_denotation_hashes.setdefault(
            outcome.root.root_id, set()
        ).add(outcome.denotation_hash)
        case_dispositions.setdefault(
            outcome.root.root_id, set()
        ).add(outcome.actual_disposition)
    if any(
        len(values) != 1 or None in values
        for values in case_denotation_hashes.values()
    ):
        issues.append(
            "controlled_serialization_denotation_invariance_failed"
        )
    if any(
        len(values) != 1
        for values in case_dispositions.values()
    ):
        issues.append(
            "controlled_serialization_disposition_invariance_failed"
        )

    correspondence, correspondence_issues = (
        _correspondence_diagnostics(first, registry)
    )
    issues.extend(
        _safe_issue_class(issue)
        for issue in correspondence_issues
    )
    baselines, baseline_issues = _baseline_diagnostics(
        roots, views, registry, ontology
    )
    issues.extend(
        _safe_issue_class(issue) for issue in baseline_issues
    )
    issue_ids = tuple(sorted(set(issues)))

    primary_cases = [
        root for root in roots if root.pair_role == "primary"
    ]
    negative_cases = [
        root
        for root in roots
        if root.pair_role == "counterfactual_negative"
    ]
    missingness_cases = [
        root
        for root in roots
        if root.pair_role == "missingness_control"
    ]

    def _case_decision_count(
        cases: Sequence[ControlledRoot], decision: str
    ) -> int:
        return sum(
            all(
                outcome.actual_decision == decision
                for outcome in first
                if outcome.root.root_id == root.root_id
            )
            for root in cases
        )

    full_metrics = {
        "case_count": len(roots),
        "case_count_semantics": (
            "10_primary_plus_10_paired_negative_plus_"
            "5_paired_missingness_control"
        ),
        "primary_case_total": len(primary_cases),
        "paired_negative_case_total": len(negative_cases),
        "paired_missingness_control_total": len(
            missingness_cases
        ),
        "paired_law_group_count": len(
            {root.law_id for root in roots}
        ),
        "serialization_view_count": len(views),
        "serialization_views_per_case": VIEWS_PER_ROOT,
        "serialization_views_are_not_independent_cases": True,
        "extractor_success_serialization_views": sum(
            outcome.extraction.succeeded for outcome in first
        ),
        "unique_binding_serialization_views": sum(
            outcome.bound_case is not None for outcome in first
        ),
        "required_value_missing_bound_unknown_serialization_views": sum(
            outcome.root.root_id == "root.t14.control"
            and outcome.bound_case is not None
            and any(
                quantity.quantity_id == "observation.beta"
                and quantity.observation_status
                is ObservationStatus.UNKNOWN
                and quantity.value is None
                and not quantity.evidence_span_ids
                for quantity in outcome.bound_case.episode.quantities
            )
            and outcome.actual_decision == "abstain"
            and not outcome.issue_ids
            for outcome in first
        ),
        "exact_denotation_serialization_views": sum(
            not outcome.issue_ids for outcome in first
        ),
        "primary_cases_accepted": _case_decision_count(
            primary_cases, "accepted"
        ),
        "paired_negative_cases_rejected": _case_decision_count(
            negative_cases, "rejected"
        ),
        "paired_missingness_controls_abstained": (
            _case_decision_count(missingness_cases, "abstain")
        ),
        "serialization_denotation_invariant_cases": sum(
            len(values) == 1 and None not in values
            for values in case_denotation_hashes.values()
        ),
        "serialization_disposition_invariant_cases": sum(
            len(values) == 1
            for values in case_dispositions.values()
        ),
        "same_process_byte_exact_replay": (
            "controlled_same_process_replay_failed"
            not in issue_ids
        ),
        "outcome_commitment": strict_content_hash(first_payload),
    }

    controlled_ready = not issue_ids
    receipt: dict[str, Any] = {
        "qualification_version": QUALIFICATION_VERSION,
        "qualification_contract_hash": (
            QUALIFICATION_CONTRACT_HASH
        ),
        "qualification_scope": "controlled_atomic_fact_path",
        "status": (
            "PASS_CONTROLLED_EVIDENCE_PATH"
            if controlled_ready
            else "FAIL_CONTROLLED_EVIDENCE_PATH"
        ),
        "new_study": False,
        "formal_result": False,
        "efficacy_evidence": False,
        "public_intrinsic_measurement": False,
        "controlled_implementation_ready": controlled_ready,
        "source_qualification_ready": controlled_ready,
        "source_qualification_scope": (
            "controlled_synthetic_atomic_fact_corpus_only"
        ),
        "public_source_qualified": False,
        "public_intrinsic_freeze_ready": False,
        "public_intrinsic_freeze_blocker": (
            "runtime_access_not_audited_and_public_source_not_qualified"
        ),
        "extractor_contract_hash": EXTRACTOR_CONTRACT_HASH,
        "extractor_implementation_sha256": (
            extractor_implementation_sha256()
        ),
        "baseline_contract_hash": BASELINE_CONTRACT_HASH,
        "corpus_version": CORPUS_VERSION,
        "raw_pack_hash": raw_contract["raw_pack_hash"],
        "gold_pack_hash": gold_contract["gold_pack_hash"],
        "ontology_hash": ontology.ontology_hash,
        "registry_hash": registry.registry_hash,
        "construction_diagnostics": construction,
        "full_lane": full_metrics,
        "correspondence_diagnostics": correspondence,
        "baseline_diagnostics": baselines,
        "declared_capability_surface": dict(
            QUALIFICATION_CONTRACT[
                "declared_capability_surface"
            ]
        ),
        "runtime_access_audited": False,
        "issue_count": len(issue_ids),
        "issue_ids": list(issue_ids),
        "issue_commitment": strict_content_hash(list(issue_ids)),
    }
    receipt["self_hash"] = strict_content_hash(receipt)
    return receipt


__all__ = [
    "QUALIFICATION_CONTRACT",
    "QUALIFICATION_CONTRACT_HASH",
    "QUALIFICATION_VERSION",
    "run_controlled_evidence_qualification",
    "run_flat_label_no_verifier_lane",
]
