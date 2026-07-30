"""Immutable domain objects for the Hegel Machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, TypeAlias

from .hashing import stable_hash

Scalar: TypeAlias = str | int | float | bool | None
FrozenPairs: TypeAlias = tuple[tuple[str, Scalar], ...]


def freeze_pairs(values: dict[str, Scalar] | FrozenPairs) -> FrozenPairs:
    if isinstance(values, tuple):
        pairs = values
    elif isinstance(values, dict):
        pairs = tuple(values.items())
    else:
        raise TypeError("frozen payload must be a dict or immutable tuple")
    keys: list[str] = []
    checked: list[tuple[str, Scalar]] = []
    for key, value in pairs:
        if not isinstance(key, str) or not key:
            raise TypeError("frozen payload keys must be nonempty strings")
        if key in keys:
            raise ValueError(f"duplicate frozen payload key: {key}")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"frozen payload value for {key} is not a scalar")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"frozen payload value for {key} is not finite")
        keys.append(key)
        checked.append((key, value))
    return tuple(sorted(checked))


def require_tuple(value: object, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be an immutable tuple")


def require_finite(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


class EvidenceSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"
    HARD_NEGATIVE = "hard_negative"
    COUNTERFACTUAL = "counterfactual"
    OLD_SUCCESS = "old_success"


class EvidenceKind(str, Enum):
    PROOF = "proof"
    EXECUTABLE_TEST = "executable_test"
    PHYSICAL_OR_SIMULATION = "physical_or_simulation"
    HELDOUT_HUMAN = "heldout_human"
    INDEPENDENT_LLM = "independent_llm"
    SEMANTIC_RETRIEVAL = "semantic_retrieval"


class LawKind(str, Enum):
    SYMMETRY = "symmetry_equivariance"
    MONOTONICITY = "monotonicity_order"
    CONSERVATION = "conservation_balance"
    COMPLEMENTARITY = "complementarity_nonadditivity"
    NEGATIVE_FEEDBACK = "negative_feedback"
    LOCALITY = "locality_markov"


class PatchCoordinate(str, Enum):
    PARAMETER = "parameter"
    NOISE = "noise_or_data"
    SCOPE = "scope"
    MIXTURE = "mixture"
    COMPOSITION = "composition"
    REPRESENTATION = "representation"
    ROBUSTIFICATION = "robustification"
    IDEALIZATION = "idealization"
    PROBE = "probe"
    LANGUAGE = "language"
    EVALUATOR = "evaluator"
    REVISION = "belief_or_theory_revision"


class AuthorityRole(str, Enum):
    GENERATOR = "generator"
    FORMALIZER = "formalizer"
    FALSIFIER = "falsifier"
    EVALUATOR = "evaluator"
    PROMOTER = "promoter"


class PromotionDecision(str, Enum):
    REJECT = "reject"
    BRANCH_ONLY = "branch_only"
    CANDIDATE = "candidate_framework"
    ACTIVE_SCOPED = "active_scoped_framework"


class FrameworkStatus(str, Enum):
    DRAFT = "draft_branch"
    CANDIDATE_BRANCH = "candidate_branch"
    BRANCH_ONLY = "branch_only"
    CANDIDATE = "candidate_framework"
    ACTIVE_SCOPED = "active_scoped_framework"
    GENERAL = "general_framework"
    DEMOTED = "demoted_to_branch"
    DEPRECATED = "deprecated"
    REJECTED = "rejected_boundary_only"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    source_uri: str
    split: EvidenceSplit
    data_cutoff: str
    observables: FrozenPairs
    seed: int | None = None
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        require_tuple(self.observables, "observation observables")
        object.__setattr__(self, "observables", freeze_pairs(self.observables))
        if not self.observation_id or not self.source_uri or not self.data_cutoff:
            raise ValueError("observation id, source, and data cutoff are required")
        if not self.observables:
            raise ValueError("an observation must expose at least one observable")
        freeze_pairs(self.observables)

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="observation_")

    @classmethod
    def from_mapping(
        cls,
        *,
        observation_id: str,
        source_uri: str,
        split: EvidenceSplit,
        data_cutoff: str,
        observables: dict[str, Scalar],
        seed: int | None = None,
        provenance_hash: str = "",
    ) -> "Observation":
        return cls(
            observation_id=observation_id,
            source_uri=source_uri,
            split=split,
            data_cutoff=data_cutoff,
            observables=freeze_pairs(observables),
            seed=seed,
            provenance_hash=provenance_hash,
        )


@dataclass(frozen=True, slots=True)
class ScaleContext:
    scale_id: str
    task_id: str
    axes: tuple[str, ...]
    aggregation: str
    validity_scope: tuple[str, ...]
    selected_on_split: EvidenceSplit = EvidenceSplit.TRAIN

    def __post_init__(self) -> None:
        require_tuple(self.axes, "scale axes")
        require_tuple(self.validity_scope, "scale validity scope")
        if not self.scale_id or not self.task_id or not self.axes:
            raise ValueError("scale id, task id, and axes are required")
        if self.selected_on_split not in {EvidenceSplit.TRAIN, EvidenceSplit.VALIDATION}:
            raise ValueError("scale selection may not use holdout outcomes")


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    probe_id: str
    version: str
    input_type: str
    output_type: str
    metric: str
    task_ids: tuple[str, ...]
    evaluator_epoch: str
    anchor_ids: tuple[str, ...]
    data_cutoff: str
    cost: float = 1.0
    semantic_only: bool = False

    def __post_init__(self) -> None:
        require_tuple(self.task_ids, "probe task ids")
        require_tuple(self.anchor_ids, "probe anchor ids")
        if not all((self.probe_id, self.version, self.metric, self.evaluator_epoch)):
            raise ValueError("probe identity, metric, and evaluator epoch are required")
        if self.cost <= 0:
            raise ValueError("probe cost must be positive")
        require_finite(self.cost, "probe cost")


@dataclass(frozen=True, slots=True)
class ViolationFunctionalSpec:
    functional_id: str
    law_kind: LawKind
    required_observables: tuple[str, ...]
    output_semantics: str
    tolerance: float

    def __post_init__(self) -> None:
        require_tuple(self.required_observables, "functional observables")
        require_finite(self.tolerance, "violation tolerance")
        if self.tolerance < 0:
            raise ValueError("violation tolerance cannot be negative")
        if not self.required_observables:
            raise ValueError("a violation functional needs observables")


@dataclass(frozen=True, slots=True)
class RelationLaw:
    law_id: str
    kind: LawKind
    symbol: str
    arity: int
    roles: tuple[str, ...]
    executable_definition: str
    violation_functional_id: str
    scope: tuple[str, ...]
    scale_ids: tuple[str, ...]
    required_observables: tuple[str, ...]
    role_observable_requirements: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        for name in (
            "roles",
            "scope",
            "scale_ids",
            "required_observables",
            "role_observable_requirements",
        ):
            require_tuple(getattr(self, name), f"law {name}")
        if self.arity < 1 or self.arity != len(self.roles):
            raise ValueError("law arity must equal the number of typed roles")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("law roles must be distinct")
        if not self.executable_definition or not self.violation_functional_id:
            raise ValueError("law needs executable and violation definitions")
        requirement_roles = tuple(role for role, _ in self.role_observable_requirements)
        if (
            set(requirement_roles) != set(self.roles)
            or len(requirement_roles) != len(set(requirement_roles))
        ):
            raise ValueError("every law role needs an observable-witness contract")
        for _, witnesses in self.role_observable_requirements:
            require_tuple(witnesses, "law role witnesses")
        required = set(self.required_observables)
        if any(
            not witnesses or not set(witnesses).issubset(required)
            for _, witnesses in self.role_observable_requirements
        ):
            raise ValueError("role witness contract cites unknown observables")


@dataclass(frozen=True, slots=True)
class PreregisteredPrediction:
    prediction_id: str
    input_condition: str
    outcome_name: str
    expected_direction: str
    expected_range: tuple[float, float] | None
    failure_criterion: str
    registered_at_cutoff: str

    def __post_init__(self) -> None:
        if self.expected_range is not None:
            require_tuple(self.expected_range, "prediction expected range")
        if not all(
            (
                self.prediction_id,
                self.input_condition,
                self.outcome_name,
                self.expected_direction,
                self.failure_criterion,
                self.registered_at_cutoff,
            )
        ):
            raise ValueError("prediction preregistration is incomplete")
        if self.expected_range and self.expected_range[0] > self.expected_range[1]:
            raise ValueError("prediction range is reversed")
        if self.expected_range:
            require_finite(self.expected_range[0], "prediction lower bound")
            require_finite(self.expected_range[1], "prediction upper bound")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="pred_")


@dataclass(frozen=True, slots=True)
class IdentifiabilityCertificate:
    certificate_id: str
    identifiable_up_to: tuple[str, ...]
    required_interventions: tuple[str, ...]
    equivalent_representations: tuple[str, ...]
    remaining_gauge_freedom: tuple[str, ...]
    verified: bool


@dataclass(frozen=True, slots=True)
class ReductionMap:
    reduction_id: str
    parent_version_id: str
    child_candidate_id: str
    old_scope: tuple[str, ...]
    mapping_description: str
    executable_check_id: str
    maximum_error: float

    def __post_init__(self) -> None:
        require_tuple(self.old_scope, "reduction old scope")
        if not self.old_scope or not self.executable_check_id:
            raise ValueError("a reduction map needs old scope and an executable check")
        if self.maximum_error < 0:
            raise ValueError("maximum reduction error cannot be negative")
        require_finite(self.maximum_error, "maximum reduction error")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="reduction_")


@dataclass(frozen=True, slots=True)
class RelationLawCandidate:
    candidate_id: str
    symbol: str
    kind: LawKind | None
    arity: int
    roles: tuple[str, ...]
    executable_definition: str
    violation_functional: ViolationFunctionalSpec
    scopes: tuple[str, ...]
    scale_ids: tuple[str, ...]
    positive_observation_ids: tuple[str, ...]
    hard_negative_ids: tuple[str, ...]
    predictions: tuple[PreregisteredPrediction, ...]
    reduction_map: ReductionMap
    description_length: float
    identifiability: IdentifiabilityCertificate

    def __post_init__(self) -> None:
        for name in (
            "roles",
            "scopes",
            "scale_ids",
            "positive_observation_ids",
            "hard_negative_ids",
            "predictions",
        ):
            require_tuple(getattr(self, name), f"candidate {name}")
        if self.arity < 1 or self.arity != len(self.roles):
            raise ValueError("candidate arity and roles disagree")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("candidate roles must be unique")
        if not self.hard_negative_ids or not self.predictions:
            raise ValueError("candidate needs hard negatives and novel predictions")
        if self.description_length < 0:
            raise ValueError("description length cannot be negative")
        require_finite(self.description_length, "description length")


@dataclass(frozen=True, slots=True)
class EvaluatorSpec:
    evaluator_id: str
    epoch: str
    version: str
    scope: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    failure_modes: tuple[str, ...]
    adversarial_case_ids: tuple[str, ...]
    frozen_at_cutoff: str

    def __post_init__(self) -> None:
        for name in (
            "scope",
            "anchor_ids",
            "failure_modes",
            "adversarial_case_ids",
        ):
            require_tuple(getattr(self, name), f"evaluator {name}")
        if not self.anchor_ids or not self.adversarial_case_ids:
            raise ValueError("evaluator needs independent anchors and adversarial cases")


@dataclass(frozen=True, slots=True)
class EvidenceReceipt:
    receipt_id: str
    theory_version_id: str
    candidate_id: str
    evaluator_epoch: str
    probe_id: str
    probe_version: str
    data_cutoff: str
    split: EvidenceSplit
    kind: EvidenceKind
    metric: str
    value: float
    threshold: float
    higher_is_better: bool
    passed: bool
    independent: bool
    observation_ids: tuple[str, ...]
    preregistration_id: str | None = None
    actor_id: str = ""

    def __post_init__(self) -> None:
        require_tuple(self.observation_ids, "receipt observation ids")
        require_finite(self.value, "receipt value")
        require_finite(self.threshold, "receipt threshold")
        if not self.receipt_id or not self.actor_id:
            raise ValueError("receipt id and evidence actor are required")
        if not self.probe_id or not self.probe_version:
            raise ValueError("receipt must bind a registered probe and version")
        expected = (
            self.value >= self.threshold
            if self.higher_is_better
            else self.value <= self.threshold
        )
        if expected != self.passed:
            raise ValueError("receipt pass flag disagrees with value and threshold")
        if not self.observation_ids:
            raise ValueError("receipt must point to measured observations")
        if self.split is EvidenceSplit.HOLDOUT and not self.preregistration_id:
            raise ValueError("holdout evidence must point to preregistration")

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="receipt_")


@dataclass(frozen=True, slots=True)
class AuthorityAssignment:
    role: AuthorityRole
    actor_id: str

    def __post_init__(self) -> None:
        if not self.actor_id:
            raise ValueError("authority actor id is required")


@dataclass(frozen=True, slots=True)
class TheoryPatch:
    patch_id: str
    candidate_id: str
    parent_version_id: str
    coordinate: PatchCoordinate
    claim: str
    scope: tuple[str, ...]
    failure_boundary: tuple[str, ...]
    predictions: tuple[PreregisteredPrediction, ...]
    hard_negative_ids: tuple[str, ...]
    reduction_map_id: str
    conditional_description_length: float
    payload: FrozenPairs
    authority_assignments: tuple[AuthorityAssignment, ...]
    ontology_report_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "scope",
            "failure_boundary",
            "predictions",
            "hard_negative_ids",
            "payload",
            "authority_assignments",
        ):
            require_tuple(getattr(self, name), f"patch {name}")
        roles = [assignment.role for assignment in self.authority_assignments]
        if (
            len(self.authority_assignments) != len(AuthorityRole)
            or set(roles) != set(AuthorityRole)
            or len(set(roles)) != len(roles)
        ):
            raise ValueError("all five authorities must be assigned")
        actors = [assignment.actor_id for assignment in self.authority_assignments]
        if any(not actor for actor in actors) or len(set(actors)) != len(actors):
            raise ValueError("the five authorities must be distinct actors")
        if not self.scope or not self.failure_boundary:
            raise ValueError("patch scope and failure boundary are required")
        if not self.predictions or not self.hard_negative_ids:
            raise ValueError("patch needs predictions and hard negatives")
        if self.conditional_description_length < 0:
            raise ValueError("conditional description length cannot be negative")
        require_finite(
            self.conditional_description_length,
            "conditional description length",
        )
        if len({prediction.content_id for prediction in self.predictions}) != len(
            self.predictions
        ):
            raise ValueError("patch repeats a preregistered prediction")
        if len(set(self.hard_negative_ids)) != len(self.hard_negative_ids):
            raise ValueError("patch repeats a hard negative")
        object.__setattr__(self, "payload", freeze_pairs(self.payload))
        if self.coordinate is PatchCoordinate.LANGUAGE and not self.ontology_report_id:
            raise ValueError("language extension requires an ontology inadequacy report")

    @property
    def prediction_ids(self) -> tuple[str, ...]:
        return tuple(prediction.content_id for prediction in self.predictions)

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="patch_")


@dataclass(frozen=True, slots=True)
class TheoryState:
    schema_version: str
    parent_version_id: str | None
    signature: tuple[str, ...]
    model_classes: tuple[str, ...]
    representations: tuple[str, ...]
    relation_laws: tuple[RelationLaw, ...]
    hypothesis_families: tuple[str, ...]
    probes: tuple[ProbeSpec, ...]
    violation_functionals: tuple[ViolationFunctionalSpec, ...]
    scales: tuple[ScaleContext, ...]
    scope: tuple[str, ...]
    evaluator: EvaluatorSpec
    observational_equivalences: tuple[tuple[str, ...], ...] = ()
    negative_memory: tuple[str, ...] = ()
    reduction_maps: tuple[ReductionMap, ...] = ()
    conditional_description_length: float = 0.0
    data_cutoff: str = ""

    def __post_init__(self) -> None:
        for name in (
            "signature",
            "model_classes",
            "representations",
            "relation_laws",
            "hypothesis_families",
            "probes",
            "violation_functionals",
            "scales",
            "scope",
            "observational_equivalences",
            "negative_memory",
            "reduction_maps",
        ):
            require_tuple(getattr(self, name), f"theory {name}")
        for equivalence in self.observational_equivalences:
            require_tuple(equivalence, "observational equivalence class")
        law_ids = [law.law_id for law in self.relation_laws]
        probe_ids = [probe.probe_id for probe in self.probes]
        if len(law_ids) != len(set(law_ids)) or len(probe_ids) != len(set(probe_ids)):
            raise ValueError("theory law and probe identifiers must be unique")
        if any(probe.evaluator_epoch != self.evaluator.epoch for probe in self.probes):
            raise ValueError("all active probes must belong to the evaluator epoch")
        if not self.data_cutoff:
            raise ValueError("theory data cutoff is required")
        require_finite(
            self.conditional_description_length,
            "theory conditional description length",
        )

    @property
    def version_id(self) -> str:
        return stable_hash(self, prefix="theory_")


def payload_dict(payload: FrozenPairs) -> dict[str, Any]:
    return dict(payload)
