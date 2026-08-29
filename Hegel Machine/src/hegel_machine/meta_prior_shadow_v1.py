"""Bounded shadow mechanics for data-driven robustification and idealization.

The slice derives both operations from measured residual rows, compiles immutable
``TheoryPatch`` objects, and opens independent holdout rows only for replay.  It
does not mutate a theory graph or authorize promotion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .bootstrap import initial_theory
from .diagnostics import (
    DiagnosticThresholds,
    InadequacyVerdict,
    OntologyInadequacyReport,
    ResidualProfile,
    diagnose_ontology_inadequacy,
)
from .engines import (
    IdealizationContract,
    IdealizationDecision,
    RobustModelKind,
    RobustificationCandidate,
    RobustificationDecision,
    UncertaintyKind,
    evaluate_idealization,
    evaluate_robustification,
)
from .hashing import stable_hash
from .schema import (
    AuthorityAssignment,
    AuthorityRole,
    EvidenceKind,
    EvidenceReceipt,
    EvidenceSplit,
    Observation,
    PatchCoordinate,
    PreregisteredPrediction,
    ProbeSpec,
    ReductionMap,
    TheoryPatch,
    freeze_pairs,
)

SHADOW_STATUS = "BOUNDED_DUAL_BRANCH_SHADOW_PASS"
SHADOW_TASK_ID = "meta_prior_bounded_theory_operation_shadow"


@dataclass(frozen=True, slots=True)
class RobustificationPolicy:
    candidate_id: str
    scope: tuple[str, ...]
    failure_boundary: tuple[str, ...]
    model_kind: RobustModelKind
    uncertainty_kind: UncertaintyKind
    minimum_tail_gain: float
    minimum_safety_gain: float
    minimum_coverage: float
    maximum_nominal_loss: float
    maximum_reduction_error: float


@dataclass(frozen=True, slots=True)
class IdealizationPolicy:
    contract_id: str
    candidate_id: str
    scope: tuple[str, ...]
    failure_boundary: tuple[str, ...]
    operation: str
    deleted_degrees_of_freedom: tuple[str, ...]
    preserved_observables: tuple[str, ...]
    scale_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    discrepancy_budget: float
    minimum_complexity_gain: float
    maximum_reduction_error: float
    full_model_recovery: str


@dataclass(frozen=True, slots=True)
class RobustificationDiscovery:
    candidate: RobustificationCandidate
    decision: RobustificationDecision
    diagnosis: OntologyInadequacyReport
    parent_nominal_utility: float
    parent_tail_utility: float
    parent_safety_utility: float
    discovery_observation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IdealizationDiscovery:
    contract: IdealizationContract
    decision: IdealizationDecision
    diagnosis: OntologyInadequacyReport
    reduction_map: ReductionMap
    scope: tuple[str, ...]
    discovery_observation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RobustificationShadowResult:
    discovery: RobustificationDiscovery
    patch: TheoryPatch
    holdout_probe: ProbeSpec
    holdout_receipt: EvidenceReceipt
    holdout_observation_ids: tuple[str, ...]
    rejected_control: RobustificationDecision

    @property
    def passed(self) -> bool:
        return (
            self.discovery.decision.accepted
            and self.discovery.diagnosis.verdict
            is InadequacyVerdict.ROBUSTIFICATION_CANDIDATE
            and self.holdout_receipt.passed
            and not self.rejected_control.accepted
            and self.patch.coordinate is PatchCoordinate.ROBUSTIFICATION
        )


@dataclass(frozen=True, slots=True)
class IdealizationShadowResult:
    discovery: IdealizationDiscovery
    patch: TheoryPatch
    holdout_probe: ProbeSpec
    holdout_receipt: EvidenceReceipt
    holdout_observation_ids: tuple[str, ...]
    rejected_control: IdealizationDecision

    @property
    def passed(self) -> bool:
        return (
            self.discovery.decision.accepted
            and self.discovery.diagnosis.verdict
            is InadequacyVerdict.IDEALIZATION_CANDIDATE
            and self.holdout_receipt.passed
            and not self.rejected_control.accepted
            and self.patch.coordinate is PatchCoordinate.IDEALIZATION
        )


@dataclass(frozen=True, slots=True)
class MetaPriorShadowReport:
    status: str
    robustification: RobustificationShadowResult
    idealization: IdealizationShadowResult
    external_evidence_claimed: bool = False
    formal_state_mutated: bool = False
    q05b_claimed: bool = False
    q1_claimed: bool = False
    c1_claimed: bool = False
    active_claimed: bool = False


def _values(observation: Observation) -> dict[str, Any]:
    return dict(observation.observables)


def _number(values: Mapping[str, Any], name: str) -> float:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _boolean(values: Mapping[str, Any], name: str) -> bool:
    value = values[name]
    if type(value) is not bool:
        raise TypeError(f"{name} must be boolean")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("a measured residual group is empty")
    return sum(values) / len(values)


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _require_split(
    observations: tuple[Observation, ...], split: EvidenceSplit
) -> None:
    if not observations or any(item.split is not split for item in observations):
        raise ValueError(
            f"residual observations must be a nonempty {split.value} tuple"
        )


def _kind(observation: Observation) -> str:
    value = _values(observation).get("case_kind")
    if not isinstance(value, str) or not value:
        raise TypeError("case_kind must be a nonempty string")
    return value


def _content_ids(observations: Sequence[Observation]) -> tuple[str, ...]:
    return tuple(item.content_id for item in observations)


def derive_robustification_from_residuals(
    *,
    parent_version_id: str,
    parent_scope: tuple[str, ...],
    observations: tuple[Observation, ...],
    policy: RobustificationPolicy,
) -> RobustificationDiscovery:
    """Derive all utilities and gains from TRAIN residual rows."""

    _require_split(observations, EvidenceSplit.TRAIN)
    normal = tuple(item for item in observations if _kind(item) == "normal")
    stress = tuple(item for item in observations if _kind(item) == "stress")
    reduction = tuple(item for item in observations if _kind(item) == "reduction")
    if not normal or not stress or not reduction:
        raise ValueError(
            "robustification discovery needs normal, stress, and reduction rows"
        )

    normal_values = tuple(_values(item) for item in normal)
    stress_values = tuple(_values(item) for item in stress)
    reduction_values = tuple(_values(item) for item in reduction)
    parent_nominal = _mean(
        tuple(_number(row, "parent_utility") for row in normal_values)
    )
    candidate_nominal = _mean(
        tuple(_number(row, "candidate_utility") for row in normal_values)
    )
    parent_tail = _mean(
        tuple(_number(row, "parent_utility") for row in stress_values)
    )
    candidate_tail = _mean(
        tuple(_number(row, "candidate_utility") for row in stress_values)
    )
    parent_safety = _mean(
        tuple(float(_boolean(row, "parent_safe")) for row in stress_values)
    )
    candidate_safety = _mean(
        tuple(float(_boolean(row, "candidate_safe")) for row in stress_values)
    )
    parent_coverage = _mean(
        tuple(float(_boolean(row, "parent_covered")) for row in stress_values)
    )
    candidate_coverage = _mean(
        tuple(float(_boolean(row, "candidate_covered")) for row in stress_values)
    )
    description_costs = tuple(
        max(
            0.0,
            (
                _number(row, "candidate_description_units")
                - _number(row, "parent_description_units")
            )
            / _number(row, "parent_description_units"),
        )
        for row in normal_values + stress_values
    )
    reduction_error = max(
        abs(
            _number(row, "candidate_prediction")
            - _number(row, "parent_prediction")
        )
        for row in reduction_values
        if _number(row, "uncertainty_radius") == 0.0
    )
    reduction_map = ReductionMap(
        reduction_id=f"reduction_{policy.candidate_id}",
        parent_version_id=parent_version_id,
        child_candidate_id=policy.candidate_id,
        old_scope=parent_scope,
        mapping_description="zero uncertainty radius recovers the parent prediction",
        executable_check_id="replay_zero_radius_predictions_v1",
        maximum_error=reduction_error,
    )
    candidate = RobustificationCandidate(
        candidate_id=policy.candidate_id,
        parent_version_id=parent_version_id,
        model_kind=policy.model_kind,
        uncertainty_kind=policy.uncertainty_kind,
        scope=policy.scope,
        stress_test_ids=_content_ids(stress),
        reduction_map=reduction_map,
        nominal_utility=candidate_nominal,
        tail_utility=candidate_tail,
        safety_utility=candidate_safety,
        coverage=candidate_coverage,
        complexity_cost=_mean(description_costs),
    )
    decision = evaluate_robustification(
        candidate,
        minimum_tail_gain=policy.minimum_tail_gain,
        minimum_safety_gain=policy.minimum_safety_gain,
        minimum_coverage=policy.minimum_coverage,
        maximum_nominal_loss=policy.maximum_nominal_loss,
        parent_nominal_utility=parent_nominal,
        parent_tail_utility=parent_tail,
        parent_safety_utility=parent_safety,
        maximum_reduction_error=policy.maximum_reduction_error,
    )
    coherent = _mean(
        tuple(
            float(
                _number(row, "candidate_utility")
                > _number(row, "parent_utility")
            )
            for row in stress_values
        )
    )
    profile = ResidualProfile(
        profile_id=stable_hash(_content_ids(observations), prefix="robust_residual_"),
        refit_gain=0.0,
        uncertainty_coverage=parent_coverage,
        scope_repair_gain=0.0,
        mixture_gain=0.0,
        low_order_composition_gain=0.0,
        idealization_gain=0.0,
        robustification_tail_gain=_clamp_unit(candidate_tail - parent_tail),
        added_probe_gain=0.0,
        cross_seed_stability=float(
            len({item.seed for item in stress}) == len(stress)
            and all(item.seed is not None for item in stress)
        ),
        structural_coherence=coherent,
        uncertainty_excess=1.0 - parent_coverage,
        compression_gain=0.0,
        preregistered_prediction_gain=0.0,
        case_count=len(stress),
        outlier_fraction=0.0,
    )
    diagnosis = diagnose_ontology_inadequacy(
        profile,
        DiagnosticThresholds(repair_gain=policy.minimum_tail_gain),
    )
    return RobustificationDiscovery(
        candidate,
        decision,
        diagnosis,
        parent_nominal,
        parent_tail,
        parent_safety,
        _content_ids(observations),
    )


def derive_idealization_from_residuals(
    *,
    parent_version_id: str,
    parent_scope: tuple[str, ...],
    observations: tuple[Observation, ...],
    policy: IdealizationPolicy,
) -> IdealizationDiscovery:
    """Derive probe discrepancy and resource gains from TRAIN residual rows."""

    _require_split(observations, EvidenceSplit.TRAIN)
    preservation = tuple(
        item for item in observations if _kind(item) == "preservation"
    )
    counterexamples = tuple(
        item for item in observations if _kind(item) == "boundary_counterexample"
    )
    if not preservation or not counterexamples:
        raise ValueError(
            "idealization discovery needs preservation and boundary-counterexample rows"
        )
    values = tuple(_values(item) for item in preservation)
    maximum_discrepancy = max(
        abs(_number(row, "idealized_output") - _number(row, "full_output"))
        for row in values
    )

    def relative_gain(full_name: str, smaller_name: str) -> float:
        full = sum(_number(row, full_name) for row in values)
        smaller = sum(_number(row, smaller_name) for row in values)
        if full <= 0:
            raise ValueError(f"{full_name} total must be positive")
        return (full - smaller) / full

    compute_gain = relative_gain("full_compute_units", "idealized_compute_units")
    sample_gain = relative_gain("full_sample_units", "idealized_sample_units")
    complexity_gain = relative_gain(
        "full_description_units", "idealized_description_units"
    )
    reduction_error = max(
        abs(_number(row, "recovered_output") - _number(row, "full_output"))
        for row in values
    )
    reduction_map = ReductionMap(
        reduction_id=f"reduction_{policy.candidate_id}",
        parent_version_id=parent_version_id,
        child_candidate_id=policy.candidate_id,
        old_scope=parent_scope,
        mapping_description=policy.full_model_recovery,
        executable_check_id="replay_full_model_recovery_v1",
        maximum_error=reduction_error,
    )
    contract = IdealizationContract(
        contract_id=policy.contract_id,
        parent_version_id=parent_version_id,
        candidate_id=policy.candidate_id,
        operation=policy.operation,
        deleted_degrees_of_freedom=policy.deleted_degrees_of_freedom,
        preserved_observables=policy.preserved_observables,
        scale_ids=policy.scale_ids,
        task_ids=policy.task_ids,
        maximum_probe_discrepancy=maximum_discrepancy,
        discrepancy_budget=policy.discrepancy_budget,
        failure_boundary=policy.failure_boundary,
        counterexample_ids=_content_ids(counterexamples),
        compute_gain=compute_gain,
        sample_gain=sample_gain,
        complexity_gain=complexity_gain,
        full_model_recovery=policy.full_model_recovery,
    )
    decision = evaluate_idealization(
        contract,
        minimum_complexity_gain=policy.minimum_complexity_gain,
    )
    if reduction_error > policy.maximum_reduction_error:
        decision = IdealizationDecision(
            False,
            contract.contract_id,
            "failed:reduction_replay",
        )
    profile = ResidualProfile(
        profile_id=stable_hash(_content_ids(observations), prefix="ideal_residual_"),
        refit_gain=0.0,
        uncertainty_coverage=0.0,
        scope_repair_gain=0.0,
        mixture_gain=0.0,
        low_order_composition_gain=0.0,
        idealization_gain=_clamp_unit(
            min(complexity_gain, max(compute_gain, sample_gain))
        ),
        robustification_tail_gain=0.0,
        added_probe_gain=0.0,
        cross_seed_stability=0.0,
        structural_coherence=_mean(
            tuple(
                float(
                    abs(
                        _number(row, "idealized_output")
                        - _number(row, "full_output")
                    )
                    <= policy.discrepancy_budget
                )
                for row in values
            )
        ),
        uncertainty_excess=0.0,
        compression_gain=_clamp_unit(complexity_gain),
        preregistered_prediction_gain=0.0,
        case_count=len(preservation),
        outlier_fraction=0.0,
    )
    diagnosis = diagnose_ontology_inadequacy(
        profile,
        DiagnosticThresholds(repair_gain=policy.minimum_complexity_gain),
    )
    return IdealizationDiscovery(
        contract,
        decision,
        diagnosis,
        reduction_map,
        policy.scope,
        _content_ids(observations),
    )


def _authorities(branch: str) -> tuple[AuthorityAssignment, ...]:
    return tuple(
        AuthorityAssignment(role, f"meta_prior_shadow_{branch}_{role.value}")
        for role in AuthorityRole
    )


def compile_robustification_patch(
    discovery: RobustificationDiscovery,
    *,
    prediction: PreregisteredPrediction,
    hard_negative_ids: tuple[str, ...],
    failure_boundary: tuple[str, ...],
) -> TheoryPatch:
    candidate = discovery.candidate
    return TheoryPatch(
        patch_id=f"patch_{candidate.candidate_id}",
        candidate_id=candidate.candidate_id,
        parent_version_id=candidate.parent_version_id,
        coordinate=PatchCoordinate.ROBUSTIFICATION,
        claim="bounded uncertainty expansion repairs measured tail undercoverage",
        scope=candidate.scope,
        failure_boundary=failure_boundary,
        predictions=(prediction,),
        hard_negative_ids=hard_negative_ids,
        reduction_map_id=candidate.reduction_map.reduction_id,
        conditional_description_length=candidate.complexity_cost,
        payload=freeze_pairs(
            {
                "operation": candidate.model_kind.value,
                "uncertainty_kind": candidate.uncertainty_kind.value,
                "diagnostic_report_id": discovery.diagnosis.report_id,
                "discovery_observation_ids": ",".join(
                    discovery.discovery_observation_ids
                ),
                "shadow_only": True,
            }
        ),
        authority_assignments=_authorities("robustification"),
    )


def compile_idealization_patch(
    discovery: IdealizationDiscovery,
    *,
    prediction: PreregisteredPrediction,
    hard_negative_ids: tuple[str, ...],
) -> TheoryPatch:
    contract = discovery.contract
    return TheoryPatch(
        patch_id=f"patch_{contract.candidate_id}",
        candidate_id=contract.candidate_id,
        parent_version_id=contract.parent_version_id,
        coordinate=PatchCoordinate.IDEALIZATION,
        claim="bounded quotient preserves task probes while reducing resources",
        scope=discovery.scope,
        failure_boundary=contract.failure_boundary,
        predictions=(prediction,),
        hard_negative_ids=hard_negative_ids,
        reduction_map_id=discovery.reduction_map.reduction_id,
        conditional_description_length=max(0.0, 1.0 - contract.complexity_gain),
        payload=freeze_pairs(
            {
                "operation": contract.operation,
                "diagnostic_report_id": discovery.diagnosis.report_id,
                "discovery_observation_ids": ",".join(
                    discovery.discovery_observation_ids
                ),
                "deleted_degrees_of_freedom": ",".join(
                    contract.deleted_degrees_of_freedom
                ),
                "preserved_observables": ",".join(
                    contract.preserved_observables
                ),
                "shadow_only": True,
            }
        ),
        authority_assignments=_authorities("idealization"),
    )


def _probe_and_receipt(
    *,
    parent_version_id: str,
    evaluator_epoch: str,
    candidate_id: str,
    prediction: PreregisteredPrediction,
    branch: str,
    observation_ids: tuple[str, ...],
    value: float,
    opened_at_cutoff: str,
) -> tuple[ProbeSpec, EvidenceReceipt]:
    probe = ProbeSpec(
        probe_id=f"probe_meta_prior_{branch}_holdout_v1",
        version="1",
        input_type="independent_preregistered_residual_rows",
        output_type="bounded_unseen_success_fraction",
        metric="unseen_prediction_success",
        task_ids=(SHADOW_TASK_ID,),
        evaluator_epoch=evaluator_epoch,
        anchor_ids=(prediction.content_id,),
        data_cutoff=prediction.registered_at_cutoff,
        semantic_only=False,
    )
    receipt = EvidenceReceipt(
        receipt_id=f"receipt_meta_prior_{branch}_holdout_v1",
        theory_version_id=parent_version_id,
        candidate_id=candidate_id,
        evaluator_epoch=evaluator_epoch,
        probe_id=probe.probe_id,
        probe_version=probe.version,
        data_cutoff=opened_at_cutoff,
        split=EvidenceSplit.HOLDOUT,
        kind=EvidenceKind.EXECUTABLE_TEST,
        metric="unseen_prediction_success",
        value=value,
        threshold=1.0,
        higher_is_better=True,
        passed=value >= 1.0,
        independent=True,
        observation_ids=observation_ids,
        preregistration_id=prediction.content_id,
        actor_id=f"meta_prior_shadow_{branch}_independent_evaluator",
    )
    return probe, receipt


def qualify_robustification_holdout(
    *,
    discovery: RobustificationDiscovery,
    observations: tuple[Observation, ...],
    policy: RobustificationPolicy,
    prediction: PreregisteredPrediction,
    evaluator_epoch: str,
    opened_at_cutoff: str,
) -> tuple[ProbeSpec, EvidenceReceipt]:
    _require_split(observations, EvidenceSplit.HOLDOUT)
    if any(_kind(item) != "stress" for item in observations):
        raise ValueError("robustification holdout accepts only stress rows")
    passed = []
    for observation in observations:
        row = _values(observation)
        passed.append(
            _number(row, "candidate_utility") - _number(row, "parent_utility")
            >= policy.minimum_tail_gain
            and float(_boolean(row, "candidate_safe"))
            - float(_boolean(row, "parent_safe"))
            >= policy.minimum_safety_gain
            and _boolean(row, "candidate_covered")
        )
    return _probe_and_receipt(
        parent_version_id=discovery.candidate.parent_version_id,
        evaluator_epoch=evaluator_epoch,
        candidate_id=discovery.candidate.candidate_id,
        prediction=prediction,
        branch="robustification_stress",
        observation_ids=_content_ids(observations),
        value=_mean(tuple(float(item) for item in passed)),
        opened_at_cutoff=opened_at_cutoff,
    )


def qualify_idealization_holdout(
    *,
    discovery: IdealizationDiscovery,
    observations: tuple[Observation, ...],
    policy: IdealizationPolicy,
    prediction: PreregisteredPrediction,
    evaluator_epoch: str,
    opened_at_cutoff: str,
) -> tuple[ProbeSpec, EvidenceReceipt]:
    _require_split(observations, EvidenceSplit.HOLDOUT)
    if any(_kind(item) != "preservation" for item in observations):
        raise ValueError("idealization holdout accepts only preservation rows")
    passed = []
    for observation in observations:
        row = _values(observation)
        passed.append(
            abs(_number(row, "idealized_output") - _number(row, "full_output"))
            <= policy.discrepancy_budget
            and abs(_number(row, "recovered_output") - _number(row, "full_output"))
            <= policy.maximum_reduction_error
        )
    return _probe_and_receipt(
        parent_version_id=discovery.contract.parent_version_id,
        evaluator_epoch=evaluator_epoch,
        candidate_id=discovery.contract.candidate_id,
        prediction=prediction,
        branch="idealization",
        observation_ids=_content_ids(observations),
        value=_mean(tuple(float(item) for item in passed)),
        opened_at_cutoff=opened_at_cutoff,
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _tuple_of_strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise TypeError(f"{name} must be a nonempty string list")
    return tuple(value)


def _observations(
    rows: object,
    *,
    split: EvidenceSplit,
    branch: str,
    data_cutoff: str,
) -> tuple[Observation, ...]:
    if not isinstance(rows, list) or not rows:
        raise TypeError(f"{branch} rows must be a nonempty list")
    built = []
    for item in rows:
        row = dict(_mapping(item, f"{branch} row"))
        observation_id = row.pop("observation_id")
        seed = row.pop("seed", None)
        if not isinstance(observation_id, str) or not observation_id:
            raise TypeError("observation_id must be a nonempty string")
        built.append(
            Observation.from_mapping(
                observation_id=observation_id,
                source_uri=f"config:meta_prior_bounded_shadow_v1#{branch}",
                split=split,
                data_cutoff=data_cutoff,
                observables=row,
                seed=seed,
            )
        )
    return tuple(built)


def _prediction(
    section: Mapping[str, Any], registered_at_cutoff: str
) -> PreregisteredPrediction:
    body = _mapping(section["prediction"], "prediction")
    expected_range = body["expected_range"]
    if not isinstance(expected_range, list) or len(expected_range) != 2:
        raise TypeError("prediction expected_range must contain two values")
    return PreregisteredPrediction(
        prediction_id=str(body["prediction_id"]),
        input_condition=str(body["input_condition"]),
        outcome_name=str(body["outcome_name"]),
        expected_direction=str(body["expected_direction"]),
        expected_range=(float(expected_range[0]), float(expected_range[1])),
        failure_criterion=str(body["failure_criterion"]),
        registered_at_cutoff=registered_at_cutoff,
    )


def run_bounded_meta_prior_shadow(
    config: Mapping[str, Any],
) -> MetaPriorShadowReport:
    """Replay the configured local fixture without writing formal state."""

    if config.get("schema") != "hegel-meta-prior-bounded-shadow/1":
        raise ValueError("unexpected bounded shadow config schema")
    if config.get("claim_boundary") != "SHADOW_ONLY_LOCAL_MECHANICS":
        raise ValueError("bounded shadow config exceeds its claim boundary")
    if (
        config.get("measurement_boundary")
        != "BOUNDED_FIXTURE_MEASUREMENTS_NOT_EXTERNAL_EVIDENCE"
    ):
        raise ValueError("bounded shadow rows cannot claim external evidence")
    registered = str(config["registered_at_cutoff"])
    opened = str(config["opened_at_cutoff"])
    if registered >= opened:
        raise ValueError("holdout must open after preregistration")
    parent = initial_theory(data_cutoff=opened)

    robust = _mapping(config["robustification"], "robustification")
    robust_policy_body = _mapping(robust["policy"], "robustification policy")
    robust_policy = RobustificationPolicy(
        candidate_id=str(robust["candidate_id"]),
        scope=_tuple_of_strings(robust["scope"], "robustification scope"),
        failure_boundary=_tuple_of_strings(
            robust["failure_boundary"], "robustification failure boundary"
        ),
        model_kind=RobustModelKind(str(robust["model_kind"])),
        uncertainty_kind=UncertaintyKind(str(robust["uncertainty_kind"])),
        minimum_tail_gain=float(robust_policy_body["minimum_tail_gain"]),
        minimum_safety_gain=float(robust_policy_body["minimum_safety_gain"]),
        minimum_coverage=float(robust_policy_body["minimum_coverage"]),
        maximum_nominal_loss=float(robust_policy_body["maximum_nominal_loss"]),
        maximum_reduction_error=float(
            robust_policy_body["maximum_reduction_error"]
        ),
    )
    robust_discovery_rows = _observations(
        robust["discovery_rows"],
        split=EvidenceSplit.TRAIN,
        branch="robustification-discovery",
        data_cutoff=registered,
    )
    robust_control_rows = _observations(
        robust["rejection_control_rows"],
        split=EvidenceSplit.TRAIN,
        branch="robustification-rejection-control",
        data_cutoff=registered,
    )
    robust_discovery = derive_robustification_from_residuals(
        parent_version_id=parent.version_id,
        parent_scope=parent.scope,
        observations=robust_discovery_rows,
        policy=robust_policy,
    )
    robust_control = derive_robustification_from_residuals(
        parent_version_id=parent.version_id,
        parent_scope=parent.scope,
        observations=robust_control_rows,
        policy=robust_policy,
    )
    robust_prediction = _prediction(robust, registered)
    robust_patch = compile_robustification_patch(
        robust_discovery,
        prediction=robust_prediction,
        hard_negative_ids=_content_ids(robust_control_rows),
        failure_boundary=robust_policy.failure_boundary,
    )
    robust_holdout_rows = _observations(
        robust["holdout_rows"],
        split=EvidenceSplit.HOLDOUT,
        branch="robustification-holdout",
        data_cutoff=opened,
    )
    robust_probe, robust_receipt = qualify_robustification_holdout(
        discovery=robust_discovery,
        observations=robust_holdout_rows,
        policy=robust_policy,
        prediction=robust_prediction,
        evaluator_epoch=parent.evaluator.epoch,
        opened_at_cutoff=opened,
    )

    ideal = _mapping(config["idealization"], "idealization")
    ideal_policy_body = _mapping(ideal["policy"], "idealization policy")
    ideal_policy = IdealizationPolicy(
        contract_id=str(ideal["contract_id"]),
        candidate_id=str(ideal["candidate_id"]),
        scope=_tuple_of_strings(ideal["scope"], "idealization scope"),
        failure_boundary=_tuple_of_strings(
            ideal["failure_boundary"], "idealization failure boundary"
        ),
        operation=str(ideal["operation"]),
        deleted_degrees_of_freedom=_tuple_of_strings(
            ideal["deleted_degrees_of_freedom"],
            "idealization deleted degrees of freedom",
        ),
        preserved_observables=_tuple_of_strings(
            ideal["preserved_observables"], "idealization preserved observables"
        ),
        scale_ids=_tuple_of_strings(ideal["scale_ids"], "idealization scales"),
        task_ids=_tuple_of_strings(ideal["task_ids"], "idealization tasks"),
        discrepancy_budget=float(ideal_policy_body["discrepancy_budget"]),
        minimum_complexity_gain=float(
            ideal_policy_body["minimum_complexity_gain"]
        ),
        maximum_reduction_error=float(
            ideal_policy_body["maximum_reduction_error"]
        ),
        full_model_recovery=str(ideal["full_model_recovery"]),
    )
    ideal_discovery_rows = _observations(
        ideal["discovery_rows"],
        split=EvidenceSplit.TRAIN,
        branch="idealization-discovery",
        data_cutoff=registered,
    )
    ideal_control_rows = _observations(
        ideal["rejection_control_rows"],
        split=EvidenceSplit.TRAIN,
        branch="idealization-rejection-control",
        data_cutoff=registered,
    )
    ideal_discovery = derive_idealization_from_residuals(
        parent_version_id=parent.version_id,
        parent_scope=parent.scope,
        observations=ideal_discovery_rows,
        policy=ideal_policy,
    )
    ideal_control = derive_idealization_from_residuals(
        parent_version_id=parent.version_id,
        parent_scope=parent.scope,
        observations=ideal_control_rows,
        policy=ideal_policy,
    )
    ideal_prediction = _prediction(ideal, registered)
    ideal_patch = compile_idealization_patch(
        ideal_discovery,
        prediction=ideal_prediction,
        hard_negative_ids=_content_ids(ideal_control_rows),
    )
    ideal_holdout_rows = _observations(
        ideal["holdout_rows"],
        split=EvidenceSplit.HOLDOUT,
        branch="idealization-holdout",
        data_cutoff=opened,
    )
    ideal_probe, ideal_receipt = qualify_idealization_holdout(
        discovery=ideal_discovery,
        observations=ideal_holdout_rows,
        policy=ideal_policy,
        prediction=ideal_prediction,
        evaluator_epoch=parent.evaluator.epoch,
        opened_at_cutoff=opened,
    )

    robust_result = RobustificationShadowResult(
        robust_discovery,
        robust_patch,
        robust_probe,
        robust_receipt,
        _content_ids(robust_holdout_rows),
        robust_control.decision,
    )
    ideal_result = IdealizationShadowResult(
        ideal_discovery,
        ideal_patch,
        ideal_probe,
        ideal_receipt,
        _content_ids(ideal_holdout_rows),
        ideal_control.decision,
    )
    status = (
        SHADOW_STATUS
        if robust_result.passed and ideal_result.passed
        else "REJECTED"
    )
    return MetaPriorShadowReport(status, robust_result, ideal_result)
