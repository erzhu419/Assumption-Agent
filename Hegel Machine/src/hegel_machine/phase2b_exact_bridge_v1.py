"""Exact root/identity residual, tolerance, and selector bridge for Phase-2B.

This module consumes a canonical bundle-level exact uncertainty receipt.  It
never reads numeric payloads from the public wire for verifier evaluation and
never converts an exact interval back to binary floating point.  The sole
binary64 boundary is the frozen theory tolerance, whose represented value is
recovered once with ``Fraction.from_float``.

All six frozen law residuals are extended with conservative rational interval
arithmetic.  Domain predicates that are not true for every value in an input
interval fail closed.  Root scales and explicit identity paths are supported;
every other transform remains an error cell in the complete adapter grid.
This is an implementation receipt, not sealed evidence or a Phase-2B exit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from fractions import Fraction
from typing import Final

from .hashing import stable_hash
from .phase2b_adapter import (
    AdapterDisposition,
    AdapterEnumerationResult,
    CandidateHypothesis,
    LawWireBinding,
    ObservableChannelBinding,
    Phase2BAdapterRegistry,
    enumerate_candidate_hypotheses,
)
from .phase2b_uncertainty_compiler import (
    BundleUncertaintyCompilation,
    BundleUncertaintyDisposition,
    DEFAULT_EXACT_UNCERTAINTY_POLICY,
    ExactObservationCompilation,
    ExactRationalInterval,
    FROZEN_PHASE2B_EXACT_FREEZE_ID,
    FROZEN_RATIONAL_GRID_ID,
    ObservationValueKind,
    compile_bundle_uncertainty,
)
from .phase2b_wire import (
    BooleanValue,
    AggregationEdge,
    AggregationGraph,
    EntityCandidate,
    MeasurementUncertainty,
    Missingness,
    NumericInterval,
    NumericValue,
    PublicEvidenceBundle,
    SpatialSupport,
    TaskTarget,
    TemporalSupport,
    TransformOperation,
    TransformSpec,
    TypedObservation,
    UncertaintyModel,
    UnitDimension,
)
from .schema import (
    EvidenceSplit,
    EvaluatorSpec,
    LawKind,
    ProbeSpec,
    ReductionMap,
    RelationLaw,
    ScaleContext,
    TheoryState,
    ViolationFunctionalSpec,
    require_tuple,
)


EXACT_BRIDGE_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-root-identity-exact-rational-bridge/1"
)
EXACT_VERIFIER_SEMANTICS_VERSION: Final = (
    "phase2b_exact_interval_verifier_semantics_v1"
)
FROZEN_PARENT_VERIFIER_REGISTRY_ID: Final = (
    "verifier_registry_sha256_"
    "2bb3e96544cfc9c52718efdc2880fb33607bdc597bf47c30b640ee07cb654fc2"
)


class ExactObservableKind(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    BOOLEAN = "boolean"


class ExactCandidateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


class ExactBridgeDisposition(str, Enum):
    COMPLETE = "complete"
    ABSTAIN = "abstain"


class ExactSelectionDisposition(str, Enum):
    UNIQUE_IDENTIFICATION = "unique_identification"
    ADMISSIBLE_SCALE_SET = "admissible_scale_set"
    ABSTAIN = "abstain"


EXACT_OBSERVABLE_KINDS: Final = (
    (
        LawKind.SYMMETRY,
        (
            ("common_codomains", ExactObservableKind.BOOLEAN),
            ("forward", ExactObservableKind.VECTOR),
            ("transformed", ExactObservableKind.VECTOR),
        ),
    ),
    (
        LawKind.MONOTONICITY,
        tuple(
            (name, ExactObservableKind.SCALAR)
            for name in ("direction", "x_high", "x_low", "y_high", "y_low")
        ),
    ),
    (
        LawKind.CONSERVATION,
        (
            ("boundary_observed", ExactObservableKind.BOOLEAN),
            ("inflows", ExactObservableKind.VECTOR),
            ("outflows", ExactObservableKind.VECTOR),
            ("sinks", ExactObservableKind.VECTOR),
            ("sources", ExactObservableKind.VECTOR),
            ("storage_delta", ExactObservableKind.SCALAR),
        ),
    ),
    (
        LawKind.COMPLEMENTARITY,
        tuple(
            (name, ExactObservableKind.SCALAR)
            for name in (
                "expected_interaction",
                "interaction_margin",
                "u_a",
                "u_ab",
                "u_b",
                "u_empty",
            )
        ),
    ),
    (
        LawKind.NEGATIVE_FEEDBACK,
        (
            ("controlled_quantity_observed", ExactObservableKind.BOOLEAN),
            ("deviation_after_response", ExactObservableKind.SCALAR),
            ("deviation_before_response", ExactObservableKind.SCALAR),
            ("disturbance_delta", ExactObservableKind.SCALAR),
            ("disturbance_precedes_response", ExactObservableKind.BOOLEAN),
            (
                "local_stability_window_observed",
                ExactObservableKind.BOOLEAN,
            ),
            ("mitigation_margin", ExactObservableKind.SCALAR),
            ("response_delta", ExactObservableKind.SCALAR),
            ("response_margin", ExactObservableKind.SCALAR),
            ("same_controlled_quantity", ExactObservableKind.BOOLEAN),
            ("system_induced_response", ExactObservableKind.BOOLEAN),
        ),
    ),
    (
        LawKind.LOCALITY,
        (
            ("blanket_observed", ExactObservableKind.BOOLEAN),
            ("conditional_a", ExactObservableKind.VECTOR),
            ("conditional_b", ExactObservableKind.VECTOR),
            ("same_blanket_state", ExactObservableKind.BOOLEAN),
        ),
    ),
)

EXACT_FORMULA_BINDINGS: Final = (
    (
        LawKind.SYMMETRY,
        "hegel_machine.laws.evaluate_symmetry",
        "normalized maximum equivariance residual",
    ),
    (
        LawKind.MONOTONICITY,
        "hegel_machine.laws.evaluate_monotonicity",
        "normalized order violation",
    ),
    (
        LawKind.CONSERVATION,
        "hegel_machine.laws.evaluate_conservation",
        "normalized signed balance residual",
    ),
    (
        LawKind.COMPLEMENTARITY,
        "hegel_machine.laws.evaluate_complementarity",
        "preregistered pair-interaction sign/margin violation",
    ),
    (
        LawKind.NEGATIVE_FEEDBACK,
        "hegel_machine.laws.evaluate_negative_feedback",
        "strict sign-opposition and mitigation-margin violation",
    ),
    (
        LawKind.LOCALITY,
        "hegel_machine.laws.evaluate_locality",
        "conditional total variation outside a fixed Markov blanket",
    ),
)


@dataclass(frozen=True, order=True, slots=True)
class ExactFractionAtom:
    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise TypeError("exact fraction fields must be integers")
        if self.denominator <= 0:
            raise ValueError("exact fraction denominator must be positive")
        reduced = Fraction(self.numerator, self.denominator)
        if (reduced.numerator, reduced.denominator) != (
            self.numerator,
            self.denominator,
        ):
            raise ValueError("ExactFractionAtom must already be reduced")

    @classmethod
    def from_fraction(cls, value: Fraction) -> "ExactFractionAtom":
        if type(value) is not Fraction:
            raise TypeError("exact fraction atom requires Fraction")
        return cls(value.numerator, value.denominator)

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True, slots=True)
class ExactInterval:
    lower: ExactFractionAtom
    upper: ExactFractionAtom

    def __post_init__(self) -> None:
        if type(self.lower) is not ExactFractionAtom or type(
            self.upper
        ) is not ExactFractionAtom:
            raise TypeError("exact interval endpoints have the wrong type")
        if self.lower_fraction > self.upper_fraction:
            raise ValueError("exact interval lower exceeds upper")

    @classmethod
    def from_fractions(
        cls,
        lower: Fraction,
        upper: Fraction,
    ) -> "ExactInterval":
        if type(lower) is not Fraction or type(upper) is not Fraction:
            raise TypeError("exact interval requires Fraction endpoints")
        return cls(
            ExactFractionAtom.from_fraction(lower),
            ExactFractionAtom.from_fraction(upper),
        )

    @classmethod
    def point(cls, value: Fraction) -> "ExactInterval":
        return cls.from_fractions(value, value)

    @property
    def lower_fraction(self) -> Fraction:
        return self.lower.as_fraction()

    @property
    def upper_fraction(self) -> Fraction:
        return self.upper.as_fraction()

    @property
    def is_point(self) -> bool:
        return self.lower == self.upper


@dataclass(frozen=True, slots=True)
class ExactBridgePolicy:
    schema_version: str = EXACT_BRIDGE_SCHEMA_VERSION
    verifier_semantics_version: str = EXACT_VERIFIER_SEMANTICS_VERSION
    uncertainty_policy_id: str = DEFAULT_EXACT_UNCERTAINTY_POLICY.policy_id
    phase2b_exact_freeze_id: str = FROZEN_PHASE2B_EXACT_FREEZE_ID
    rational_grid_id: str = FROZEN_RATIONAL_GRID_ID
    supported_transforms: tuple[TransformOperation, ...] = (
        TransformOperation.IDENTITY,
    )
    maximum_candidate_count: int = 50_000
    maximum_scale_count: int = 64
    maximum_edge_count: int = 256
    maximum_transform_catalog_count: int = 256
    maximum_observation_count: int = 4_096
    maximum_entity_candidate_count: int = 4_096
    maximum_role_id_count: int = 512
    maximum_quantity_id_count: int = 4_096
    maximum_registry_observable_channel_count: int = 128
    maximum_registry_law_binding_count: int = 6
    maximum_total_registry_role_bindings: int = 64
    maximum_total_registry_observable_requirements: int = 128
    maximum_theory_law_count: int = 6
    maximum_theory_functional_count: int = 6
    maximum_theory_top_level_items: int = 4_096
    maximum_authority_tree_nodes: int = 200_000
    maximum_authority_text_characters: int = 2_000_000
    maximum_authority_integer_bit_length: int = 4_096
    maximum_observation_reference_width: int = 512
    maximum_total_role_memberships: int = 65_536
    maximum_adapter_scan_work: int = 1_000_000
    maximum_vector_width: int = 256
    maximum_total_observation_components: int = 65_536
    maximum_operations_per_candidate: int = 2_500
    maximum_total_operations: int = 1_000_000
    maximum_fraction_bit_length: int = 4_096

    def __post_init__(self) -> None:
        if self.schema_version != EXACT_BRIDGE_SCHEMA_VERSION:
            raise ValueError("exact bridge schema drift")
        if self.verifier_semantics_version != EXACT_VERIFIER_SEMANTICS_VERSION:
            raise ValueError("exact verifier semantics drift")
        if self.uncertainty_policy_id != DEFAULT_EXACT_UNCERTAINTY_POLICY.policy_id:
            raise ValueError("exact uncertainty policy identity drift")
        if self.phase2b_exact_freeze_id != FROZEN_PHASE2B_EXACT_FREEZE_ID:
            raise ValueError("exact bridge freeze identity drift")
        if self.rational_grid_id != FROZEN_RATIONAL_GRID_ID:
            raise ValueError("exact bridge rational grid identity drift")
        if self.supported_transforms != (TransformOperation.IDENTITY,):
            raise ValueError("exact bridge v1 is root/identity only")
        if (
            self.maximum_candidate_count,
            self.maximum_scale_count,
            self.maximum_edge_count,
            self.maximum_transform_catalog_count,
            self.maximum_observation_count,
            self.maximum_entity_candidate_count,
            self.maximum_role_id_count,
            self.maximum_quantity_id_count,
            self.maximum_registry_observable_channel_count,
            self.maximum_registry_law_binding_count,
            self.maximum_total_registry_role_bindings,
            self.maximum_total_registry_observable_requirements,
            self.maximum_theory_law_count,
            self.maximum_theory_functional_count,
            self.maximum_theory_top_level_items,
            self.maximum_authority_tree_nodes,
            self.maximum_authority_text_characters,
            self.maximum_authority_integer_bit_length,
            self.maximum_observation_reference_width,
            self.maximum_total_role_memberships,
            self.maximum_adapter_scan_work,
            self.maximum_vector_width,
            self.maximum_total_observation_components,
            self.maximum_operations_per_candidate,
            self.maximum_total_operations,
            self.maximum_fraction_bit_length,
        ) != (
            50_000,
            64,
            256,
            256,
            4_096,
            4_096,
            512,
            4_096,
            128,
            6,
            64,
            128,
            6,
            6,
            4_096,
            200_000,
            2_000_000,
            4_096,
            512,
            65_536,
            1_000_000,
            256,
            65_536,
            2_500,
            1_000_000,
            4_096,
        ):
            raise ValueError("exact bridge linear resource budget drift")

    @property
    def verifier_semantics_id(self) -> str:
        return stable_hash(
            (
                self.verifier_semantics_version,
                FROZEN_PARENT_VERIFIER_REGISTRY_ID,
                EXACT_OBSERVABLE_KINDS,
                EXACT_FORMULA_BINDINGS,
            ),
            prefix="phase2b_exact_verifier_semantics_",
        )

    @property
    def policy_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_bridge_policy_")


DEFAULT_EXACT_BRIDGE_POLICY: Final = ExactBridgePolicy()


class _ResourceLimit(RuntimeError):
    pass


_EXACT_AUTHORITY_DATACLASS_TYPES: Final = frozenset(
    {
        AggregationEdge,
        AggregationGraph,
        BooleanValue,
        EntityCandidate,
        EvaluatorSpec,
        LawWireBinding,
        MeasurementUncertainty,
        NumericInterval,
        NumericValue,
        ObservableChannelBinding,
        Phase2BAdapterRegistry,
        ProbeSpec,
        PublicEvidenceBundle,
        ReductionMap,
        RelationLaw,
        ScaleContext,
        SpatialSupport,
        TaskTarget,
        TemporalSupport,
        TheoryState,
        TransformSpec,
        TypedObservation,
        UnitDimension,
        ViolationFunctionalSpec,
    }
)
_EXACT_AUTHORITY_ENUM_TYPES: Final = frozenset(
    {
        EvidenceSplit,
        LawKind,
        Missingness,
        TransformOperation,
        UncertaintyModel,
    }
)
_THEORY_SEQUENCE_FIELD_NAMES: Final = (
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
)


@dataclass(slots=True)
class _AuthorityTreeBudget:
    policy: ExactBridgePolicy
    nodes: int = 0
    text_characters: int = 0

    def visit(self, value: object, path: str) -> None:
        stack = [(value, path)]
        while stack:
            current, current_path = stack.pop()
            self.nodes += 1
            if self.nodes > self.policy.maximum_authority_tree_nodes:
                raise _ResourceLimit("RESOURCE_LIMIT:authority_tree_nodes")
            exact_type = type(current)
            if exact_type is str:
                self.text_characters += len(current)
                if (
                    self.text_characters
                    > self.policy.maximum_authority_text_characters
                ):
                    raise _ResourceLimit(
                        "RESOURCE_LIMIT:authority_text_characters"
                    )
                continue
            if exact_type is int:
                if (
                    current.bit_length()
                    > self.policy.maximum_authority_integer_bit_length
                ):
                    raise _ResourceLimit(
                        "RESOURCE_LIMIT:authority_integer_bit_length"
                    )
                continue
            if exact_type in (bool, float, type(None)):
                continue
            if exact_type in _EXACT_AUTHORITY_ENUM_TYPES:
                continue
            if exact_type is tuple:
                remaining = self.policy.maximum_authority_tree_nodes - self.nodes
                if len(current) > remaining:
                    raise _ResourceLimit("RESOURCE_LIMIT:authority_tree_nodes")
                stack.extend((item, current_path) for item in reversed(current))
                continue
            if exact_type in _EXACT_AUTHORITY_DATACLASS_TYPES:
                stack.extend(
                    (
                        getattr(current, field.name),
                        f"{exact_type.__name__}.{field.name}",
                    )
                    for field in reversed(fields(current))
                )
                continue
            raise TypeError(
                f"{current_path} is not an exact frozen authority schema node"
            )


@dataclass(slots=True)
class _ArithmeticBudget:
    policy: ExactBridgePolicy
    total_operations: int = 0
    candidate_operations: int = 0

    def start_candidate(self) -> None:
        self.candidate_operations = 0

    def interval(self, lower: Fraction, upper: Fraction) -> ExactInterval:
        self.total_operations += 1
        self.candidate_operations += 1
        if (
            self.total_operations > self.policy.maximum_total_operations
            or self.candidate_operations
            > self.policy.maximum_operations_per_candidate
        ):
            raise _ResourceLimit("RESOURCE_LIMIT:exact_operation_budget")
        self.check_fraction(lower)
        self.check_fraction(upper)
        return ExactInterval.from_fractions(lower, upper)

    def check_fraction(self, value: Fraction) -> None:
        if (
            value.numerator.bit_length()
            > self.policy.maximum_fraction_bit_length
            or value.denominator.bit_length()
            > self.policy.maximum_fraction_bit_length
        ):
            raise _ResourceLimit("RESOURCE_LIMIT:exact_fraction_bit_length")


@dataclass(frozen=True, slots=True)
class ExactSelectionPolicy:
    minimum_structural_margin: ExactFractionAtom = ExactFractionAtom(1)
    maximum_candidate_count: int = 50_000
    require_complete_family_coverage: bool = True
    require_binding_competitor: bool = True
    require_scale_competitor: bool = True

    def __post_init__(self) -> None:
        if type(self.minimum_structural_margin) is not ExactFractionAtom:
            raise TypeError("exact selector margin has the wrong type")
        if self.minimum_structural_margin.as_fraction() < 0:
            raise ValueError("exact selector margin must be nonnegative")
        if type(self.maximum_candidate_count) is not int or (
            self.maximum_candidate_count <= 0
        ):
            raise ValueError("exact selector candidate budget must be positive")
        for value in (
            self.require_complete_family_coverage,
            self.require_binding_competitor,
            self.require_scale_competitor,
        ):
            if type(value) is not bool:
                raise TypeError("exact selector flags must be Boolean")

    @property
    def policy_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_selector_policy_")


DEFAULT_EXACT_SELECTION_POLICY: Final = ExactSelectionPolicy()


@dataclass(frozen=True, slots=True)
class ExactCandidateEvaluation:
    candidate_id: str
    law_id: str
    law_kind: LawKind
    violation_functional_id: str
    role_binding: tuple[tuple[str, str], ...]
    scale_hypothesis_id: str
    footprint_id: str
    source_observation_ids: tuple[str, ...]
    used_observation_compilation_ids: tuple[str, ...]
    bundle_content_id: str
    uncertainty_result_id: str
    uncertainty_policy_id: str
    phase2b_exact_freeze_id: str
    rational_grid_id: str
    theory_version_id: str
    registry_id: str
    adapter_result_id: str
    candidate_grid_commitment_id: str
    bridge_policy_id: str
    verifier_semantics_id: str
    residual: ExactInterval | None
    tolerance: ExactInterval | None
    normalized: ExactInterval | None
    completed: bool
    error_code: str | None = None

    def __post_init__(self) -> None:
        require_tuple(self.role_binding, "exact candidate role binding")
        require_tuple(
            self.source_observation_ids,
            "exact candidate source observations",
        )
        require_tuple(
            self.used_observation_compilation_ids,
            "exact candidate used observation compilations",
        )
        identities = (
            self.candidate_id,
            self.law_id,
            self.violation_functional_id,
            self.scale_hypothesis_id,
            self.footprint_id,
            self.bundle_content_id,
            self.uncertainty_result_id,
            self.uncertainty_policy_id,
            self.phase2b_exact_freeze_id,
            self.rational_grid_id,
            self.theory_version_id,
            self.registry_id,
            self.adapter_result_id,
            self.candidate_grid_commitment_id,
            self.bridge_policy_id,
            self.verifier_semantics_id,
        )
        if not all(isinstance(item, str) and item for item in identities):
            raise ValueError("exact candidate identity is incomplete")
        if self.role_binding != tuple(sorted(self.role_binding)):
            raise ValueError("exact candidate role binding is not canonical")
        if self.source_observation_ids != tuple(
            sorted(self.source_observation_ids)
        ):
            raise ValueError("exact candidate source observations are not canonical")
        if self.used_observation_compilation_ids != tuple(
            sorted(self.used_observation_compilation_ids)
        ):
            raise ValueError("exact candidate compilation IDs are not canonical")
        if type(self.completed) is not bool:
            raise TypeError("exact candidate completed flag must be Boolean")
        if self.completed:
            if (
                self.error_code is not None
                or self.residual is None
                or self.tolerance is None
                or self.normalized is None
            ):
                raise ValueError("completed exact candidate needs only score intervals")
            if self.residual.lower_fraction < 0:
                raise ValueError("exact residual cannot be negative")
            if self.tolerance.lower_fraction <= 0:
                raise ValueError("exact tolerance must be strictly positive")
        elif (
            self.error_code is None
            or self.residual is not None
            or self.tolerance is not None
            or self.normalized is not None
        ):
            raise ValueError("failed exact candidate needs only an error code")

    @property
    def normalized_interval(self) -> ExactInterval | None:
        return self.normalized

    @property
    def status(self) -> ExactCandidateStatus:
        normalized = self.normalized_interval
        if normalized is None:
            return ExactCandidateStatus.ERROR
        one = Fraction(1)
        if normalized.upper_fraction <= one:
            return ExactCandidateStatus.PASS
        if normalized.lower_fraction > one:
            return ExactCandidateStatus.FAIL
        return ExactCandidateStatus.INCONCLUSIVE

    @property
    def content_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_candidate_evaluation_")


@dataclass(frozen=True, slots=True)
class ExactBridgeCompilation:
    disposition: ExactBridgeDisposition
    reason: str
    bundle_content_id: str
    uncertainty_result_id: str
    uncertainty_policy_id: str
    phase2b_exact_freeze_id: str
    rational_grid_id: str
    theory_version_id: str
    registry_id: str
    adapter_result_id: str | None
    candidate_grid_commitment_id: str | None
    bridge_policy_id: str
    verifier_semantics_id: str
    evaluations: tuple[ExactCandidateEvaluation, ...]

    def __post_init__(self) -> None:
        require_tuple(self.evaluations, "exact bridge evaluations")
        if not all(
            isinstance(item, str) and item
            for item in (
                self.reason,
                self.bundle_content_id,
                self.uncertainty_result_id,
                self.uncertainty_policy_id,
                self.phase2b_exact_freeze_id,
                self.rational_grid_id,
                self.theory_version_id,
                self.registry_id,
                self.bridge_policy_id,
                self.verifier_semantics_id,
            )
        ):
            raise ValueError("exact bridge compilation identity is incomplete")
        if self.disposition is ExactBridgeDisposition.COMPLETE:
            if (
                not self.adapter_result_id
                or not self.candidate_grid_commitment_id
                or not self.evaluations
            ):
                raise ValueError("complete exact bridge needs a full candidate grid")
        elif (
            self.adapter_result_id is not None
            or self.candidate_grid_commitment_id is not None
            or self.evaluations
        ):
            raise ValueError("abstaining exact bridge cannot return a partial grid")

    @property
    def result_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_bridge_result_")


@dataclass(frozen=True, slots=True)
class ExactSelectorDecision:
    disposition: ExactSelectionDisposition
    reason: str
    selection_policy_id: str
    bridge_result_id: str
    bundle_content_id: str
    uncertainty_result_id: str
    candidate_grid_commitment_id: str | None
    evaluated_candidate_ids: tuple[str, ...]
    selected_law_kind: LawKind | None = None
    selected_role_binding: tuple[tuple[str, str], ...] = ()
    admissible_scale_hypothesis_ids: tuple[str, ...] = ()
    normalized_structural_margin: ExactFractionAtom | None = None

    def __post_init__(self) -> None:
        require_tuple(self.evaluated_candidate_ids, "exact decision candidate IDs")
        require_tuple(self.selected_role_binding, "exact decision role binding")
        require_tuple(
            self.admissible_scale_hypothesis_ids,
            "exact decision scale hypotheses",
        )
        if not all(
            isinstance(item, str) and item
            for item in (
                self.reason,
                self.selection_policy_id,
                self.bridge_result_id,
                self.bundle_content_id,
                self.uncertainty_result_id,
            )
        ):
            raise ValueError("exact selector decision identity is incomplete")
        if self.evaluated_candidate_ids != tuple(
            sorted(self.evaluated_candidate_ids)
        ):
            raise ValueError("exact decision candidate IDs are not canonical")
        if self.disposition is ExactSelectionDisposition.ABSTAIN:
            if any(
                (
                    self.selected_law_kind is not None,
                    bool(self.selected_role_binding),
                    bool(self.admissible_scale_hypothesis_ids),
                    self.normalized_structural_margin is not None,
                )
            ):
                raise ValueError("exact abstention cannot carry a selection")
        elif (
            self.candidate_grid_commitment_id is None
            or self.selected_law_kind is None
            or not self.selected_role_binding
            or not self.admissible_scale_hypothesis_ids
            or self.normalized_structural_margin is None
        ):
            raise ValueError("exact identified decision is incomplete")

    @property
    def decision_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_selector_decision_")


@dataclass(frozen=True, slots=True)
class ExactBridgeRun:
    disposition: ExactBridgeDisposition
    reason: str
    bundle_content_id: str
    theory_version_id: str
    registry_id: str
    bridge_policy_id: str
    selection_policy_id: str
    uncertainty_receipt: BundleUncertaintyCompilation | None
    compilation: ExactBridgeCompilation | None
    decision: ExactSelectorDecision | None

    def __post_init__(self) -> None:
        if not all(
            isinstance(item, str) and item
            for item in (
                self.reason,
                self.bundle_content_id,
                self.theory_version_id,
                self.registry_id,
                self.bridge_policy_id,
                self.selection_policy_id,
            )
        ):
            raise ValueError("exact bridge run identity is incomplete")
        if self.uncertainty_receipt is None:
            if self.compilation is not None or self.decision is not None:
                raise ValueError("preflight-only run cannot carry downstream receipts")
            if self.disposition is not ExactBridgeDisposition.ABSTAIN:
                raise ValueError("preflight-only run must abstain")
        elif self.compilation is None or self.decision is None:
            raise ValueError("post-uncertainty run needs compilation and decision")
        elif self.disposition is not self.compilation.disposition:
            raise ValueError("exact run and compilation dispositions disagree")
        if self.compilation is not None and (
            self.compilation.bundle_content_id != self.bundle_content_id
            or self.compilation.theory_version_id != self.theory_version_id
            or self.compilation.registry_id != self.registry_id
            or self.compilation.bridge_policy_id != self.bridge_policy_id
        ):
            raise ValueError("exact run compilation provenance drift")
        if self.decision is not None and self.compilation is not None and (
            self.decision.bridge_result_id != self.compilation.result_id
            or self.decision.selection_policy_id != self.selection_policy_id
        ):
            raise ValueError("exact run decision provenance drift")

    @property
    def run_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_bridge_run_")


@dataclass(frozen=True, slots=True)
class ExactBridgePreflightRejection:
    """Uncommitted rejection emitted before authority content hashing."""

    disposition: ExactBridgeDisposition
    reason: str
    bundle_id: str
    theory_schema_version: str
    registry_theory_version_id: str
    bridge_policy_id: str
    selection_policy_id: str
    bundle_content_id: None = None
    theory_version_id: None = None
    registry_id: None = None
    uncertainty_receipt: None = None
    compilation: None = None
    decision: None = None

    def __post_init__(self) -> None:
        if self.disposition is not ExactBridgeDisposition.ABSTAIN:
            raise ValueError("exact bridge preflight rejection must abstain")
        if not all(
            type(item) is str and item
            for item in (
                self.reason,
                self.bundle_id,
                self.theory_schema_version,
                self.registry_theory_version_id,
                self.bridge_policy_id,
                self.selection_policy_id,
            )
        ):
            raise ValueError("exact bridge preflight rejection is incomplete")
        if any(
            item is not None
            for item in (
                self.bundle_content_id,
                self.theory_version_id,
                self.registry_id,
                self.uncertainty_receipt,
                self.compilation,
                self.decision,
            )
        ):
            raise ValueError("preflight rejection cannot carry committed receipts")


def _preflight_rejection(
    *,
    reason: str,
    bundle: PublicEvidenceBundle,
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
    bridge_policy: ExactBridgePolicy,
    selection_policy: ExactSelectionPolicy,
) -> ExactBridgePreflightRejection:
    return ExactBridgePreflightRejection(
        disposition=ExactBridgeDisposition.ABSTAIN,
        reason=reason,
        bundle_id=bundle.bundle_id,
        theory_schema_version=theory.schema_version,
        registry_theory_version_id=registry.theory_version_id,
        bridge_policy_id=bridge_policy.policy_id,
        selection_policy_id=selection_policy.policy_id,
    )


@dataclass(frozen=True, slots=True)
class _ExactCandidateGridCell:
    candidate_id: str
    law_id: str
    law_kind: LawKind
    family_id: str
    role_binding: tuple[tuple[str, str], ...]
    scale_hypothesis_id: str
    transform_path_ids: tuple[str, ...]
    source_observation_ids: tuple[str, ...]
    required_observable_ids: tuple[str, ...]
    footprint_id: str
    registry_id: str

    @classmethod
    def from_hypothesis(
        cls,
        hypothesis: CandidateHypothesis,
    ) -> "_ExactCandidateGridCell":
        return cls(
            candidate_id=hypothesis.candidate_id,
            law_id=hypothesis.law_id,
            law_kind=hypothesis.law_kind,
            family_id=hypothesis.family_id,
            role_binding=hypothesis.role_binding,
            scale_hypothesis_id=hypothesis.scale_hypothesis_id,
            transform_path_ids=hypothesis.transform_path_ids,
            source_observation_ids=hypothesis.source_observation_ids,
            required_observable_ids=hypothesis.required_observable_ids,
            footprint_id=hypothesis.footprint_id,
            registry_id=hypothesis.registry_id,
        )


@dataclass(frozen=True, slots=True)
class _ExactCandidateGridCommitment:
    adapter_result_id: str
    bundle_content_id: str
    registry_id: str
    expected_cells: tuple[_ExactCandidateGridCell, ...]

    def __post_init__(self) -> None:
        require_tuple(self.expected_cells, "exact candidate grid cells")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.adapter_result_id,
                self.bundle_content_id,
                self.registry_id,
            )
        ):
            raise ValueError("exact candidate grid identity is incomplete")
        if not self.expected_cells:
            raise ValueError("exact candidate grid cannot be empty")
        if self.expected_cells != tuple(
            sorted(self.expected_cells, key=lambda item: item.candidate_id)
        ):
            raise ValueError("exact candidate grid cells are not canonical")
        if len(self.expected_candidate_ids) != len(
            set(self.expected_candidate_ids)
        ):
            raise ValueError("exact candidate grid repeats a candidate")

    @property
    def expected_candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.expected_cells)

    @property
    def commitment_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_candidate_grid_")


def _exact_grid_commitment(
    adapter: AdapterEnumerationResult,
) -> _ExactCandidateGridCommitment:
    if type(adapter) is not AdapterEnumerationResult:
        raise TypeError("exact grid requires exact adapter result type")
    if adapter.disposition is not AdapterDisposition.COMPLETE:
        raise ValueError("exact grid requires a complete adapter result")
    return _ExactCandidateGridCommitment(
        adapter_result_id=adapter.result_id,
        bundle_content_id=adapter.bundle_content_id,
        registry_id=adapter.registry_id,
        expected_cells=tuple(
            _ExactCandidateGridCell.from_hypothesis(hypothesis)
            for hypothesis in adapter.hypotheses
        ),
    )


def _interval(
    budget: _ArithmeticBudget,
    lower: Fraction,
    upper: Fraction,
) -> ExactInterval:
    return budget.interval(lower, upper)


def _point(budget: _ArithmeticBudget, value: int | Fraction) -> ExactInterval:
    exact = value if type(value) is Fraction else Fraction(value)
    return _interval(budget, exact, exact)


def _add(
    budget: _ArithmeticBudget,
    left: ExactInterval,
    right: ExactInterval,
) -> ExactInterval:
    return _interval(
        budget,
        left.lower_fraction + right.lower_fraction,
        left.upper_fraction + right.upper_fraction,
    )


def _sum(
    budget: _ArithmeticBudget,
    values: tuple[ExactInterval, ...],
) -> ExactInterval:
    result = _point(budget, 0)
    for value in values:
        result = _add(budget, result, value)
    return result


def _negate(budget: _ArithmeticBudget, value: ExactInterval) -> ExactInterval:
    return _interval(budget, -value.upper_fraction, -value.lower_fraction)


def _subtract(
    budget: _ArithmeticBudget,
    left: ExactInterval,
    right: ExactInterval,
) -> ExactInterval:
    return _add(budget, left, _negate(budget, right))


def _multiply(
    budget: _ArithmeticBudget,
    left: ExactInterval,
    right: ExactInterval,
) -> ExactInterval:
    products = (
        left.lower_fraction * right.lower_fraction,
        left.lower_fraction * right.upper_fraction,
        left.upper_fraction * right.lower_fraction,
        left.upper_fraction * right.upper_fraction,
    )
    return _interval(budget, min(products), max(products))


def _scale(
    budget: _ArithmeticBudget,
    value: ExactInterval,
    factor: Fraction,
) -> ExactInterval:
    if factor >= 0:
        return _interval(
            budget,
            value.lower_fraction * factor,
            value.upper_fraction * factor,
        )
    return _interval(
        budget,
        value.upper_fraction * factor,
        value.lower_fraction * factor,
    )


def _absolute(budget: _ArithmeticBudget, value: ExactInterval) -> ExactInterval:
    lower = value.lower_fraction
    upper = value.upper_fraction
    if lower <= 0 <= upper:
        return _interval(budget, Fraction(0), max(-lower, upper))
    endpoints = (abs(lower), abs(upper))
    return _interval(budget, min(endpoints), max(endpoints))


def _maximum(
    budget: _ArithmeticBudget,
    values: tuple[ExactInterval, ...],
) -> ExactInterval:
    if not values:
        raise ValueError("exact interval maximum cannot be empty")
    return _interval(
        budget,
        max(value.lower_fraction for value in values),
        max(value.upper_fraction for value in values),
    )


def _maximum_zero(
    budget: _ArithmeticBudget,
    value: ExactInterval,
) -> ExactInterval:
    return _maximum(budget, (_point(budget, 0), value))


def _normalizer(
    budget: _ArithmeticBudget,
    values: tuple[ExactInterval, ...],
) -> ExactInterval:
    return _maximum(
        budget,
        (
            _point(budget, 1),
            *tuple(_absolute(budget, value) for value in values),
        ),
    )


def _divide(
    budget: _ArithmeticBudget,
    numerator: ExactInterval,
    denominator: ExactInterval,
) -> ExactInterval:
    if denominator.lower_fraction <= 0:
        raise ValueError("exact interval denominator must be strictly positive")
    reciprocal = _interval(
        budget,
        Fraction(1, 1) / denominator.upper_fraction,
        Fraction(1, 1) / denominator.lower_fraction,
    )
    return _multiply(budget, numerator, reciprocal)


def _point_fraction(value: ExactInterval) -> Fraction | None:
    return value.lower_fraction if value.is_point else None


def _contains_zero(value: ExactInterval) -> bool:
    return value.lower_fraction <= 0 <= value.upper_fraction


def _scalar(observables: dict[str, object], name: str) -> ExactInterval:
    value = observables[name]
    if type(value) is not ExactInterval:
        raise TypeError(f"{name} is not an exact scalar")
    return value


def _vector(
    observables: dict[str, object],
    name: str,
) -> tuple[ExactInterval, ...]:
    value = observables[name]
    if not isinstance(value, tuple) or any(
        type(item) is not ExactInterval for item in value
    ):
        raise TypeError(f"{name} is not an exact vector")
    return value


def _boolean(observables: dict[str, object], name: str) -> bool:
    value = observables[name]
    if type(value) is not bool:
        raise TypeError(f"{name} is not an exact Boolean")
    return value


def _evaluate_symmetry(
    budget: _ArithmeticBudget,
    observables: dict[str, object],
) -> ExactInterval | str:
    if not _boolean(observables, "common_codomains"):
        return "verifier_abstained:common_codomains_false"
    forward = _vector(observables, "forward")
    transformed = _vector(observables, "transformed")
    if not forward or len(forward) != len(transformed):
        return "verifier_abstained:incompatible_pair_shape"
    residuals = tuple(
        _divide(
            budget,
            _absolute(budget, _subtract(budget, left, right)),
            _normalizer(budget, (left, right)),
        )
        for left, right in zip(forward, transformed, strict=True)
    )
    return _maximum(budget, residuals)


def _evaluate_monotonicity(
    budget: _ArithmeticBudget,
    observables: dict[str, object],
) -> ExactInterval | str:
    x_low = _scalar(observables, "x_low")
    x_high = _scalar(observables, "x_high")
    if x_low.upper_fraction >= x_high.lower_fraction:
        return "nonuniform_domain:input_order_not_strict"
    direction = _point_fraction(_scalar(observables, "direction"))
    if direction not in (Fraction(-1), Fraction(1)):
        return "nonuniform_domain:direction_not_exact_sign"
    y_low = _scalar(observables, "y_low")
    y_high = _scalar(observables, "y_high")
    signed_change = _scale(
        budget,
        _subtract(budget, y_high, y_low),
        direction,
    )
    return _divide(
        budget,
        _maximum_zero(budget, _negate(budget, signed_change)),
        _normalizer(budget, (y_low, y_high)),
    )


def _evaluate_conservation(
    budget: _ArithmeticBudget,
    observables: dict[str, object],
) -> ExactInterval | str:
    if not _boolean(observables, "boundary_observed"):
        return "verifier_abstained:boundary_unobserved"
    storage = _scalar(observables, "storage_delta")
    inflows = _sum(budget, _vector(observables, "inflows"))
    outflows = _sum(budget, _vector(observables, "outflows"))
    sources = _sum(budget, _vector(observables, "sources"))
    sinks = _sum(budget, _vector(observables, "sinks"))
    raw_balance = _add(
        budget,
        _subtract(
            budget,
            _add(budget, storage, outflows),
            _add(budget, inflows, sources),
        ),
        sinks,
    )
    scale = _normalizer(
        budget,
        (storage, inflows, outflows, sources, sinks),
    )
    return _divide(budget, _absolute(budget, raw_balance), scale)


def _evaluate_complementarity(
    budget: _ArithmeticBudget,
    observables: dict[str, object],
) -> ExactInterval | str:
    expected = _point_fraction(_scalar(observables, "expected_interaction"))
    if expected not in (Fraction(-1), Fraction(0), Fraction(1)):
        return "nonuniform_domain:interaction_sign_not_exact"
    margin = _scalar(observables, "interaction_margin")
    if margin.lower_fraction < 0:
        return "nonuniform_domain:negative_interaction_margin_possible"
    u0 = _scalar(observables, "u_empty")
    ua = _scalar(observables, "u_a")
    ub = _scalar(observables, "u_b")
    uab = _scalar(observables, "u_ab")
    interaction = _add(
        budget,
        _subtract(budget, _subtract(budget, uab, ua), ub),
        u0,
    )
    scale = _normalizer(budget, (u0, ua, ub, uab))
    numerator = (
        _absolute(budget, interaction)
        if expected == 0
        else _maximum_zero(
            budget,
            _subtract(
                budget,
                margin,
                _scale(budget, interaction, expected),
            ),
        )
    )
    return _divide(budget, numerator, scale)


def _evaluate_negative_feedback(
    budget: _ArithmeticBudget,
    observables: dict[str, object],
) -> ExactInterval | str:
    required_flags = (
        "controlled_quantity_observed",
        "same_controlled_quantity",
        "disturbance_precedes_response",
        "local_stability_window_observed",
    )
    for name in required_flags:
        if not _boolean(observables, name):
            return "verifier_abstained:" + name + "_false"
    response_margin = _scalar(observables, "response_margin")
    mitigation_margin = _scalar(observables, "mitigation_margin")
    if (
        response_margin.lower_fraction <= 0
        or mitigation_margin.lower_fraction <= 0
    ):
        return "nonuniform_domain:strict_positive_margin_not_guaranteed"
    disturbance = _scalar(observables, "disturbance_delta")
    response = _scalar(observables, "response_delta")
    exact_zero_branch = (
        disturbance.is_point
        and disturbance.lower_fraction == 0
        or response.is_point
        and response.lower_fraction == 0
    )
    if exact_zero_branch or not _boolean(observables, "system_induced_response"):
        return _point(budget, 1)
    if _contains_zero(disturbance) or _contains_zero(response):
        return "nonuniform_domain:zero_branch_boundary_crossed"

    net_before = _scalar(observables, "deviation_before_response")
    net_after = _scalar(observables, "deviation_after_response")
    opposition = _negate(budget, _multiply(budget, disturbance, response))
    mitigation = _subtract(
        budget,
        _absolute(budget, net_before),
        _absolute(budget, net_after),
    )
    sign_violation = _divide(
        budget,
        _maximum_zero(
            budget,
            _subtract(budget, response_margin, opposition),
        ),
        _normalizer(budget, (response_margin, opposition)),
    )
    mitigation_violation = _divide(
        budget,
        _maximum_zero(
            budget,
            _subtract(budget, mitigation_margin, mitigation),
        ),
        _normalizer(
            budget,
            (mitigation_margin, net_before, net_after),
        ),
    )
    return _maximum(budget, (sign_violation, mitigation_violation))


def _exact_distribution(
    budget: _ArithmeticBudget,
    values: tuple[ExactInterval, ...],
) -> tuple[ExactInterval, ...] | str:
    if not values:
        return "verifier_abstained:empty_probability_vector"
    if any(value.lower_fraction < 0 for value in values):
        return "nonuniform_domain:negative_probability_possible"
    total = _sum(budget, values)
    if total.lower_fraction <= 0:
        return "nonuniform_domain:positive_probability_mass_not_guaranteed"
    return tuple(_divide(budget, value, total) for value in values)


def _evaluate_locality(
    budget: _ArithmeticBudget,
    observables: dict[str, object],
) -> ExactInterval | str:
    if not _boolean(observables, "blanket_observed"):
        return "verifier_abstained:blanket_unobserved"
    if not _boolean(observables, "same_blanket_state"):
        return "verifier_abstained:blanket_state_mismatch"
    first = _exact_distribution(
        budget,
        _vector(observables, "conditional_a"),
    )
    second = _exact_distribution(
        budget,
        _vector(observables, "conditional_b"),
    )
    if isinstance(first, str):
        return first
    if isinstance(second, str):
        return second
    if len(first) != len(second):
        return "verifier_abstained:conditional_support_mismatch"
    total_variation = _scale(
        budget,
        _sum(
            budget,
            tuple(
                _absolute(budget, _subtract(budget, left, right))
                for left, right in zip(first, second, strict=True)
            )
        ),
        Fraction(1, 2),
    )
    return _interval(
        budget,
        max(Fraction(0), total_variation.lower_fraction),
        min(Fraction(1), total_variation.upper_fraction),
    )


_EXACT_VERIFIERS = {
    LawKind.SYMMETRY: _evaluate_symmetry,
    LawKind.MONOTONICITY: _evaluate_monotonicity,
    LawKind.CONSERVATION: _evaluate_conservation,
    LawKind.COMPLEMENTARITY: _evaluate_complementarity,
    LawKind.NEGATIVE_FEEDBACK: _evaluate_negative_feedback,
    LawKind.LOCALITY: _evaluate_locality,
}


def _observable_registry() -> dict[LawKind, dict[str, ExactObservableKind]]:
    return {kind: dict(rows) for kind, rows in EXACT_OBSERVABLE_KINDS}


def _validate_registry(theory: TheoryState, registry: Phase2BAdapterRegistry) -> None:
    if theory.verifier_registry_id != FROZEN_PARENT_VERIFIER_REGISTRY_ID:
        raise ValueError("parent verifier registry identity drift")
    registered = _observable_registry()
    formula_bindings = {
        kind: (executable, output_semantics)
        for kind, executable, output_semantics in EXACT_FORMULA_BINDINGS
    }
    if set(registered) != set(LawKind):
        raise AssertionError("exact bridge law registry is incomplete")
    if set(formula_bindings) != set(LawKind):
        raise AssertionError("exact bridge formula registry is incomplete")
    theory_by_id = {law.law_id: law for law in theory.relation_laws}
    binding_by_id = {binding.law_id: binding for binding in registry.law_bindings}
    functional_by_id = {
        functional.functional_id: functional
        for functional in theory.violation_functionals
    }
    if (
        len(theory.relation_laws) != len(LawKind)
        or {law.kind for law in theory.relation_laws} != set(LawKind)
    ):
        raise ValueError("exact bridge requires one frozen law per family")
    if set(theory_by_id) != set(binding_by_id):
        raise ValueError("exact bridge adapter law registry differs")
    required_observables = {
        observable
        for law in theory.relation_laws
        for observable in law.required_observables
    }
    if {
        channel.observable_id for channel in registry.observable_channels
    } != required_observables:
        raise ValueError("exact bridge observable channel registry differs")
    for law in theory.relation_laws:
        binding = binding_by_id[law.law_id]
        functional = functional_by_id.get(law.violation_functional_id)
        if set(registered[law.kind]) != set(law.required_observables):
            raise ValueError("exact bridge observable registry drift")
        executable, output_semantics = formula_bindings[law.kind]
        if law.executable_definition != executable:
            raise ValueError("exact bridge executable definition drift")
        if (
            functional is None
            or functional.law_kind is not law.kind
            or functional.required_observables != law.required_observables
            or functional.output_semantics != output_semantics
        ):
            raise ValueError("exact bridge violation functional drift")
        if (
            binding.law_kind is not law.kind
            or tuple(sorted(role for role, _ in binding.role_ids))
            != tuple(sorted(law.roles))
            or binding.required_observable_ids
            != tuple(sorted(law.required_observables))
        ):
            raise ValueError("exact bridge adapter law binding differs")


def _candidate_observation(
    *,
    bundle: PublicEvidenceBundle,
    hypothesis: CandidateHypothesis,
    law: RelationLaw,
    law_binding: LawWireBinding,
    observable_name: str,
    quantity_id: str,
) -> TypedObservation | str:
    witness_roles = tuple(
        role
        for role, observable_names in law.role_observable_requirements
        if observable_name in observable_names
    )
    if not witness_roles:
        witness_roles = law.roles
    entities_by_role = dict(hypothesis.role_binding)
    wire_roles_by_role = dict(law_binding.role_ids)
    expected_entities = tuple(
        sorted(entities_by_role[role] for role in witness_roles)
    )
    expected_wire_roles = {wire_roles_by_role[role] for role in witness_roles}
    matches = tuple(
        observation
        for observation in bundle.observations
        if observation.quantity_id == quantity_id
        and observation.entity_ids == expected_entities
        and expected_wire_roles.issubset(observation.role_candidate_ids)
        and observation.observation_id in hypothesis.source_observation_ids
    )
    if not matches:
        return "missing_observable_witness"
    if len(matches) != 1:
        return "ambiguous_observable_witness"
    return matches[0]


def _from_frozen_interval(
    budget: _ArithmeticBudget,
    value: ExactRationalInterval,
) -> ExactInterval:
    return _interval(budget, value.lower_fraction, value.upper_fraction)


def _compiled_observable(
    budget: _ArithmeticBudget,
    compiled: ExactObservationCompilation,
    kind: ExactObservableKind,
) -> object | str:
    if compiled.value_kind is ObservationValueKind.MISSING:
        return "missing_observation"
    if kind is ExactObservableKind.BOOLEAN:
        if compiled.value_kind is not ObservationValueKind.BOOLEAN:
            return "observable_shape_mismatch"
        return compiled.boolean_value
    if compiled.value_kind is not ObservationValueKind.NUMERIC_INTERVAL:
        return "observable_shape_mismatch"
    bounds = tuple(
        _from_frozen_interval(budget, item)
        for item in compiled.numeric_bounds
    )
    if kind is ExactObservableKind.SCALAR:
        return bounds[0] if len(bounds) == 1 else "observable_shape_mismatch"
    return bounds


def _tolerance(
    budget: _ArithmeticBudget,
    theory: TheoryState,
    law: RelationLaw,
) -> tuple[str, ExactInterval] | str:
    functionals = {
        item.functional_id: item for item in theory.violation_functionals
    }
    functional = functionals.get(law.violation_functional_id)
    if functional is None or functional.law_kind is not law.kind:
        return "violation_functional_registry_drift"
    raw = functional.tolerance
    if type(raw) is float:
        exact = Fraction.from_float(raw)
    elif type(raw) is int:
        exact = Fraction(raw)
    else:
        return "violation_tolerance_type_unsupported"
    if exact <= 0:
        return "strict_positive_tolerance_not_guaranteed"
    return functional.functional_id, _interval(budget, exact, exact)


def _evaluation_common(
    *,
    hypothesis: CandidateHypothesis,
    law: RelationLaw,
    functional_id: str,
    provenance: dict[str, str],
) -> dict[str, object]:
    return {
        "candidate_id": hypothesis.candidate_id,
        "law_id": hypothesis.law_id,
        "law_kind": hypothesis.law_kind,
        "violation_functional_id": functional_id,
        "role_binding": hypothesis.role_binding,
        "scale_hypothesis_id": hypothesis.scale_hypothesis_id,
        "footprint_id": hypothesis.footprint_id,
        "source_observation_ids": hypothesis.source_observation_ids,
        **provenance,
    }


def _error_evaluation(
    *,
    common: dict[str, object],
    error_code: str,
    used_compilation_ids: tuple[str, ...] = (),
) -> ExactCandidateEvaluation:
    return ExactCandidateEvaluation(
        **common,  # type: ignore[arg-type]
        used_observation_compilation_ids=tuple(sorted(used_compilation_ids)),
        residual=None,
        tolerance=None,
        normalized=None,
        completed=False,
        error_code=error_code,
    )


def _compile_hypothesis(
    *,
    bundle: PublicEvidenceBundle,
    receipt: BundleUncertaintyCompilation,
    hypothesis: CandidateHypothesis,
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
    provenance: dict[str, str],
    policy: ExactBridgePolicy,
    arithmetic: _ArithmeticBudget,
) -> ExactCandidateEvaluation:
    laws = {law.law_id: law for law in theory.relation_laws}
    law_bindings = {item.law_id: item for item in registry.law_bindings}
    law = laws[hypothesis.law_id]
    tolerance_or_error = _tolerance(arithmetic, theory, law)
    functional_id = (
        law.violation_functional_id
        if isinstance(tolerance_or_error, str)
        else tolerance_or_error[0]
    )
    common = _evaluation_common(
        hypothesis=hypothesis,
        law=law,
        functional_id=functional_id,
        provenance=provenance,
    )
    if isinstance(tolerance_or_error, str):
        return _error_evaluation(common=common, error_code=tolerance_or_error)
    tolerance = tolerance_or_error[1]

    transforms = {item.transform_id: item for item in bundle.transform_catalog}
    for transform_id in hypothesis.transform_path_ids:
        transform = transforms.get(transform_id)
        if transform is None:
            return _error_evaluation(common=common, error_code="unknown_transform")
        if transform.operation not in policy.supported_transforms:
            return _error_evaluation(
                common=common,
                error_code="unsupported_transform_semantics:"
                + transform.operation.value,
            )

    receipt_by_observation = {
        item.observation_id: item for item in receipt.observations
    }
    quantity_by_observable = {
        item.observable_id: item.quantity_id for item in registry.observable_channels
    }
    kinds = _observable_registry()[law.kind]
    law_binding = law_bindings[law.law_id]
    observables: dict[str, object] = {}
    observations_used: list[TypedObservation] = []
    compilation_ids: list[str] = []
    for observable_name in law.required_observables:
        observation_or_error = _candidate_observation(
            bundle=bundle,
            hypothesis=hypothesis,
            law=law,
            law_binding=law_binding,
            observable_name=observable_name,
            quantity_id=quantity_by_observable[observable_name],
        )
        if isinstance(observation_or_error, str):
            return _error_evaluation(
                common=common,
                error_code=observation_or_error,
                used_compilation_ids=tuple(compilation_ids),
            )
        if observation_or_error.unit_dimension.si_exponents != (0,) * 7:
            return _error_evaluation(
                common=common,
                error_code="nondimensionless_unit_semantics_not_implemented",
                used_compilation_ids=tuple(compilation_ids),
            )
        compiled = receipt_by_observation[observation_or_error.observation_id]
        compilation_ids.append(compiled.compilation_id)
        value_or_error = _compiled_observable(
            arithmetic,
            compiled,
            kinds[observable_name],
        )
        if isinstance(value_or_error, str):
            return _error_evaluation(
                common=common,
                error_code=value_or_error,
                used_compilation_ids=tuple(compilation_ids),
            )
        observables[observable_name] = value_or_error
        observations_used.append(observation_or_error)

    first = observations_used[0]
    if any(
        observation.temporal_support != first.temporal_support
        for observation in observations_used[1:]
    ):
        return _error_evaluation(
            common=common,
            error_code="unaligned_temporal_support",
            used_compilation_ids=tuple(compilation_ids),
        )
    if any(
        observation.spatial_support != first.spatial_support
        for observation in observations_used[1:]
    ):
        return _error_evaluation(
            common=common,
            error_code="unaligned_spatial_support",
            used_compilation_ids=tuple(compilation_ids),
        )
    try:
        residual_or_error = _EXACT_VERIFIERS[law.kind](arithmetic, observables)
    except _ResourceLimit:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return _error_evaluation(
            common=common,
            error_code="exact_verifier_internal_error",
            used_compilation_ids=tuple(compilation_ids),
        )
    if isinstance(residual_or_error, str):
        return _error_evaluation(
            common=common,
            error_code=residual_or_error,
            used_compilation_ids=tuple(compilation_ids),
        )
    normalized = _divide(arithmetic, residual_or_error, tolerance)
    return ExactCandidateEvaluation(
        **common,  # type: ignore[arg-type]
        used_observation_compilation_ids=tuple(sorted(compilation_ids)),
        residual=residual_or_error,
        tolerance=tolerance,
        normalized=normalized,
        completed=True,
    )


def _observation_width(observation: TypedObservation) -> int:
    value = observation.value
    if isinstance(value, NumericValue):
        return len(value.values)
    if isinstance(value, NumericInterval):
        return len(value.lower)
    if isinstance(value, BooleanValue) or value is None:
        return 1
    raise TypeError("public observation has an unknown value type")


def _shallow_authority_size_preflight(
    bundle: PublicEvidenceBundle,
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
    policy: ExactBridgePolicy,
) -> str | None:
    bundle_sequences = (
        bundle.entity_candidates,
        bundle.role_ids,
        bundle.quantity_ids,
        bundle.observations,
        bundle.transform_catalog,
        bundle.missingness_mask,
    )
    theory_sequences = tuple(
        getattr(theory, name) for name in _THEORY_SEQUENCE_FIELD_NAMES
    )
    registry_sequences = (registry.law_bindings, registry.observable_channels)
    if any(
        type(value) is not tuple
        for value in (*bundle_sequences, *theory_sequences, *registry_sequences)
    ):
        raise TypeError("authority top-level sequences must be exact tuples")
    if len(bundle.observations) > policy.maximum_observation_count:
        return "RESOURCE_LIMIT:observation_count"
    if len(bundle.entity_candidates) > policy.maximum_entity_candidate_count:
        return "RESOURCE_LIMIT:entity_candidate_count"
    if len(bundle.role_ids) > policy.maximum_role_id_count:
        return "RESOURCE_LIMIT:role_id_count"
    if len(bundle.quantity_ids) > policy.maximum_quantity_id_count:
        return "RESOURCE_LIMIT:quantity_id_count"
    if len(bundle.transform_catalog) > policy.maximum_transform_catalog_count:
        return "RESOURCE_LIMIT:transform_catalog_count"
    if len(registry.law_bindings) > policy.maximum_registry_law_binding_count:
        return "RESOURCE_LIMIT:registry_law_binding_count"
    if (
        len(registry.observable_channels)
        > policy.maximum_registry_observable_channel_count
    ):
        return "RESOURCE_LIMIT:registry_observable_channel_count"
    if len(theory.relation_laws) > policy.maximum_theory_law_count:
        return "RESOURCE_LIMIT:theory_law_count"
    if (
        len(theory.violation_functionals)
        > policy.maximum_theory_functional_count
    ):
        return "RESOURCE_LIMIT:theory_functional_count"
    if (
        sum(len(value) for value in theory_sequences)
        > policy.maximum_theory_top_level_items
    ):
        return "RESOURCE_LIMIT:theory_top_level_items"
    return None


def _require_exact_authority_tree(
    bundle: PublicEvidenceBundle,
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
    policy: ExactBridgePolicy,
) -> None:
    budget = _AuthorityTreeBudget(policy)
    budget.visit(bundle, "bundle")
    budget.visit(theory, "theory")
    budget.visit(registry, "registry")


def _bounded_bundle_preflight(
    bundle: PublicEvidenceBundle,
    registry: Phase2BAdapterRegistry,
    policy: ExactBridgePolicy,
) -> str | None:
    graph = bundle.aggregation_graph
    if len(registry.law_bindings) > policy.maximum_registry_law_binding_count:
        return "RESOURCE_LIMIT:registry_law_binding_count"
    if (
        sum(len(binding.role_ids) for binding in registry.law_bindings)
        > policy.maximum_total_registry_role_bindings
    ):
        return "RESOURCE_LIMIT:total_registry_role_bindings"
    if (
        sum(
            len(binding.required_observable_ids)
            for binding in registry.law_bindings
        )
        > policy.maximum_total_registry_observable_requirements
    ):
        return "RESOURCE_LIMIT:total_registry_observable_requirements"
    if (
        len(bundle.transform_catalog)
        > policy.maximum_transform_catalog_count
    ):
        return "RESOURCE_LIMIT:transform_catalog_count"
    if len(bundle.observations) > policy.maximum_observation_count:
        return "RESOURCE_LIMIT:observation_count"
    if len(bundle.entity_candidates) > policy.maximum_entity_candidate_count:
        return "RESOURCE_LIMIT:entity_candidate_count"
    if len(bundle.role_ids) > policy.maximum_role_id_count:
        return "RESOURCE_LIMIT:role_id_count"
    if len(bundle.quantity_ids) > policy.maximum_quantity_id_count:
        return "RESOURCE_LIMIT:quantity_id_count"
    if (
        len(registry.observable_channels)
        > policy.maximum_registry_observable_channel_count
    ):
        return "RESOURCE_LIMIT:registry_observable_channel_count"
    if any(
        len(observation.entity_ids) + len(observation.role_candidate_ids)
        > policy.maximum_observation_reference_width
        for observation in bundle.observations
    ):
        return "RESOURCE_LIMIT:observation_reference_width"
    total_role_memberships = sum(
        len(entity.role_candidate_ids) for entity in bundle.entity_candidates
    ) + sum(
        len(observation.role_candidate_ids)
        for observation in bundle.observations
    )
    if total_role_memberships > policy.maximum_total_role_memberships:
        return "RESOURCE_LIMIT:total_role_memberships"
    if len(graph.scale_ids) > policy.maximum_scale_count:
        return "RESOURCE_LIMIT:scale_count"
    if len(graph.edges) > policy.maximum_edge_count:
        return "RESOURCE_LIMIT:edge_count"
    widths = tuple(_observation_width(item) for item in bundle.observations)
    if any(width > policy.maximum_vector_width for width in widths):
        return "RESOURCE_LIMIT:vector_width"
    if sum(widths) > policy.maximum_total_observation_components:
        return "RESOURCE_LIMIT:total_observation_components"

    catalog_ids = {item.transform_id for item in bundle.transform_catalog}
    edge_transform_ids = {item.transform_id for item in graph.edges}
    if catalog_ids != edge_transform_ids:
        return "unused_or_missing_transform_catalog_entry"

    outgoing: dict[str, list[str]] = {scale_id: [] for scale_id in graph.scale_ids}
    indegree = {scale_id: 0 for scale_id in graph.scale_ids}
    for edge in graph.edges:
        outgoing[edge.source_scale_id].append(edge.target_scale_id)
        indegree[edge.target_scale_id] += 1
    frontier = sorted(scale for scale, degree in indegree.items() if degree == 0)
    path_counts = {
        scale_id: int(scale_id in graph.root_scale_ids)
        for scale_id in graph.scale_ids
    }
    visited = 0
    while frontier:
        source = frontier.pop(0)
        visited += 1
        for target in outgoing[source]:
            path_counts[target] = min(
                2,
                path_counts[target] + path_counts[source],
            )
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
                frontier.sort()
    if visited != len(graph.scale_ids):
        return "aggregation_graph_cycle"
    if any(count != 1 for count in path_counts.values()):
        return "nonunique_transform_path"

    required_wire_roles = {
        wire_role
        for law in registry.law_bindings
        for _, wire_role in law.role_ids
    }
    entity_counts = {role_id: 0 for role_id in required_wire_roles}
    for entity in bundle.entity_candidates:
        for role_id in entity.role_candidate_ids:
            if role_id not in entity_counts:
                continue
            entity_counts[role_id] += 1
            if entity_counts[role_id] > policy.maximum_candidate_count:
                return "RESOURCE_LIMIT:raw_role_binding_product"
    projected_upper_bound = 0
    adapter_scan_work = len(bundle.role_ids) * len(bundle.entity_candidates)
    if adapter_scan_work > policy.maximum_adapter_scan_work:
        return "RESOURCE_LIMIT:adapter_scan_work"
    scale_count = len(graph.scale_ids)
    for law in registry.law_bindings:
        raw_product = 1
        for _, wire_role in law.role_ids:
            count = entity_counts.get(wire_role, 0)
            if count == 0:
                raw_product = 0
                break
            if raw_product > policy.maximum_candidate_count // count:
                return "RESOURCE_LIMIT:raw_role_binding_product"
            raw_product *= count
        if raw_product > policy.maximum_candidate_count // scale_count:
            return "RESOURCE_LIMIT:raw_role_binding_scale_product"
        law_cells = raw_product * scale_count
        if (
            projected_upper_bound
            > policy.maximum_candidate_count - law_cells
        ):
            return "RESOURCE_LIMIT:projected_candidate_count"
        projected_upper_bound += law_cells
        per_cell_work = (
            len(bundle.observations)
            * (len(law.required_observable_ids) + 1)
            + len(bundle.transform_catalog)
            + len(registry.observable_channels)
            + len(registry.law_bindings)
        )
        if per_cell_work and law_cells > (
            policy.maximum_adapter_scan_work - adapter_scan_work
        ) // per_cell_work:
            return "RESOURCE_LIMIT:adapter_scan_work"
        adapter_scan_work += law_cells * per_cell_work
    return None


def _validate_complete_evaluations(
    hypotheses: tuple[CandidateHypothesis, ...],
    evaluations: tuple[ExactCandidateEvaluation, ...],
    receipt: BundleUncertaintyCompilation,
) -> str | None:
    if len(hypotheses) != len(evaluations):
        return "internal_candidate_grid_length_drift"
    compilation_observation_by_id = {
        item.compilation_id: item.observation_id for item in receipt.observations
    }
    for hypothesis, evaluation in zip(hypotheses, evaluations, strict=True):
        if (
            evaluation.candidate_id != hypothesis.candidate_id
            or evaluation.law_id != hypothesis.law_id
            or evaluation.law_kind is not hypothesis.law_kind
            or evaluation.role_binding != hypothesis.role_binding
            or evaluation.scale_hypothesis_id != hypothesis.scale_hypothesis_id
            or evaluation.footprint_id != hypothesis.footprint_id
            or evaluation.source_observation_ids
            != hypothesis.source_observation_ids
        ):
            return "internal_candidate_metadata_drift"
        used_ids = evaluation.used_observation_compilation_ids
        if len(used_ids) != len(set(used_ids)):
            return "internal_used_compilation_id_duplicate"
        if any(
            item not in compilation_observation_by_id for item in used_ids
        ):
            return "internal_used_compilation_id_unknown"
        if any(
            compilation_observation_by_id[item]
            not in hypothesis.source_observation_ids
            for item in used_ids
        ):
            return "internal_used_compilation_outside_footprint"
        if evaluation.completed and not used_ids:
            return "internal_completed_candidate_without_observation_receipt"
    return None


def _abstaining_compilation(
    *,
    reason: str,
    bundle: PublicEvidenceBundle,
    receipt: BundleUncertaintyCompilation,
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
    policy: ExactBridgePolicy,
) -> ExactBridgeCompilation:
    return ExactBridgeCompilation(
        disposition=ExactBridgeDisposition.ABSTAIN,
        reason=reason,
        bundle_content_id=bundle.content_id,
        uncertainty_result_id=receipt.result_id,
        uncertainty_policy_id=receipt.compiler_policy_id,
        phase2b_exact_freeze_id=receipt.phase2b_exact_freeze_id,
        rational_grid_id=receipt.rational_grid_id,
        theory_version_id=theory.version_id,
        registry_id=registry.registry_id,
        adapter_result_id=None,
        candidate_grid_commitment_id=None,
        bridge_policy_id=policy.policy_id,
        verifier_semantics_id=policy.verifier_semantics_id,
        evaluations=(),
    )


def _compile_exact_candidate_grid(
    *,
    bundle: PublicEvidenceBundle,
    uncertainty_receipt: BundleUncertaintyCompilation,
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
    policy: ExactBridgePolicy = DEFAULT_EXACT_BRIDGE_POLICY,
) -> ExactBridgeCompilation:
    """Compile the complete adapter grid from one canonical exact receipt."""

    if type(bundle) is not PublicEvidenceBundle:
        raise TypeError("exact bridge requires exact PublicEvidenceBundle type")
    if type(uncertainty_receipt) is not BundleUncertaintyCompilation:
        raise TypeError("exact bridge requires exact uncertainty receipt type")
    if type(theory) is not TheoryState:
        raise TypeError("exact bridge requires exact TheoryState type")
    if type(registry) is not Phase2BAdapterRegistry:
        raise TypeError("exact bridge requires exact adapter registry type")
    if type(policy) is not ExactBridgePolicy:
        raise TypeError("exact bridge policy has the wrong type")
    if registry.theory_version_id != theory.version_id:
        raise ValueError("exact bridge theory and adapter registry disagree")
    preflight_error = _bounded_bundle_preflight(bundle, registry, policy)
    if preflight_error is not None:
        return _abstaining_compilation(
            reason=preflight_error,
            bundle=bundle,
            receipt=uncertainty_receipt,
            theory=theory,
            registry=registry,
            policy=policy,
        )
    _validate_registry(theory, registry)

    canonical_receipt = compile_bundle_uncertainty(bundle)
    if uncertainty_receipt != canonical_receipt:
        return _abstaining_compilation(
            reason="noncanonical_uncertainty_receipt",
            bundle=bundle,
            receipt=uncertainty_receipt,
            theory=theory,
            registry=registry,
            policy=policy,
        )
    if uncertainty_receipt.disposition is not BundleUncertaintyDisposition.COMPLETE:
        return _abstaining_compilation(
            reason="uncertainty_" + uncertainty_receipt.reason,
            bundle=bundle,
            receipt=uncertainty_receipt,
            theory=theory,
            registry=registry,
            policy=policy,
        )
    if (
        uncertainty_receipt.compiler_policy_id != policy.uncertainty_policy_id
        or uncertainty_receipt.phase2b_exact_freeze_id
        != policy.phase2b_exact_freeze_id
        or uncertainty_receipt.rational_grid_id != policy.rational_grid_id
    ):
        return _abstaining_compilation(
            reason="uncertainty_policy_or_grid_drift",
            bundle=bundle,
            receipt=uncertainty_receipt,
            theory=theory,
            registry=registry,
            policy=policy,
        )
    if {item.observation_id for item in uncertainty_receipt.observations} != {
        item.observation_id for item in bundle.observations
    }:
        return _abstaining_compilation(
            reason="uncertainty_receipt_observation_coverage_drift",
            bundle=bundle,
            receipt=uncertainty_receipt,
            theory=theory,
            registry=registry,
            policy=policy,
        )

    adapter = enumerate_candidate_hypotheses(bundle, registry)
    if adapter.disposition is AdapterDisposition.ABSTAIN:
        return _abstaining_compilation(
            reason="adapter_" + adapter.reason,
            bundle=bundle,
            receipt=uncertainty_receipt,
            theory=theory,
            registry=registry,
            policy=policy,
        )
    grid_commitment = _exact_grid_commitment(adapter)
    grid_commitment_id = grid_commitment.commitment_id
    theory_version_id = theory.version_id
    registry_id = registry.registry_id
    provenance = {
        "bundle_content_id": bundle.content_id,
        "uncertainty_result_id": uncertainty_receipt.result_id,
        "uncertainty_policy_id": uncertainty_receipt.compiler_policy_id,
        "phase2b_exact_freeze_id": uncertainty_receipt.phase2b_exact_freeze_id,
        "rational_grid_id": uncertainty_receipt.rational_grid_id,
        "theory_version_id": theory_version_id,
        "registry_id": registry_id,
        "adapter_result_id": adapter.result_id,
        "candidate_grid_commitment_id": grid_commitment_id,
        "bridge_policy_id": policy.policy_id,
        "verifier_semantics_id": policy.verifier_semantics_id,
    }
    arithmetic = _ArithmeticBudget(policy)
    evaluations_list: list[ExactCandidateEvaluation] = []
    try:
        for hypothesis in adapter.hypotheses:
            arithmetic.start_candidate()
            evaluations_list.append(
                _compile_hypothesis(
                    bundle=bundle,
                    receipt=uncertainty_receipt,
                    hypothesis=hypothesis,
                    theory=theory,
                    registry=registry,
                    provenance=provenance,
                    policy=policy,
                    arithmetic=arithmetic,
                )
            )
    except _ResourceLimit as exc:
        return _abstaining_compilation(
            reason=str(exc),
            bundle=bundle,
            receipt=uncertainty_receipt,
            theory=theory,
            registry=registry,
            policy=policy,
        )
    evaluations = tuple(evaluations_list)
    grid_error = _validate_complete_evaluations(
        adapter.hypotheses,
        evaluations,
        uncertainty_receipt,
    )
    if grid_error is not None:
        return _abstaining_compilation(
            reason=grid_error,
            bundle=bundle,
            receipt=uncertainty_receipt,
            theory=theory,
            registry=registry,
            policy=policy,
        )
    return ExactBridgeCompilation(
        disposition=ExactBridgeDisposition.COMPLETE,
        reason="complete_exact_root_identity_candidate_grid",
        bundle_content_id=provenance["bundle_content_id"],
        uncertainty_result_id=provenance["uncertainty_result_id"],
        uncertainty_policy_id=provenance["uncertainty_policy_id"],
        phase2b_exact_freeze_id=provenance["phase2b_exact_freeze_id"],
        rational_grid_id=provenance["rational_grid_id"],
        theory_version_id=theory_version_id,
        registry_id=registry_id,
        adapter_result_id=provenance["adapter_result_id"],
        candidate_grid_commitment_id=grid_commitment_id,
        bridge_policy_id=provenance["bridge_policy_id"],
        verifier_semantics_id=provenance["verifier_semantics_id"],
        evaluations=evaluations,
    )


def _abstaining_decision(
    reason: str,
    compilation: ExactBridgeCompilation,
    policy: ExactSelectionPolicy,
) -> ExactSelectorDecision:
    return ExactSelectorDecision(
        disposition=ExactSelectionDisposition.ABSTAIN,
        reason=reason,
        selection_policy_id=policy.policy_id,
        bridge_result_id=compilation.result_id,
        bundle_content_id=compilation.bundle_content_id,
        uncertainty_result_id=compilation.uncertainty_result_id,
        candidate_grid_commitment_id=compilation.candidate_grid_commitment_id,
        evaluated_candidate_ids=tuple(
            sorted(item.candidate_id for item in compilation.evaluations)
        ),
    )


def _select_exact_candidate_grid(
    compilation: ExactBridgeCompilation,
    *,
    bundle: PublicEvidenceBundle,
    registry: Phase2BAdapterRegistry,
    policy: ExactSelectionPolicy = DEFAULT_EXACT_SELECTION_POLICY,
) -> ExactSelectorDecision:
    """Re-enumerate and select using only exact rational comparisons."""

    if type(compilation) is not ExactBridgeCompilation:
        raise TypeError("exact selector requires exact bridge result type")
    if type(bundle) is not PublicEvidenceBundle:
        raise TypeError("exact selector requires exact evidence bundle type")
    if type(registry) is not Phase2BAdapterRegistry:
        raise TypeError("exact selector requires exact adapter registry type")
    if type(policy) is not ExactSelectionPolicy:
        raise TypeError("exact selector policy has the wrong type")
    if compilation.disposition is ExactBridgeDisposition.ABSTAIN:
        return _abstaining_decision("bridge_" + compilation.reason, compilation, policy)
    adapter = enumerate_candidate_hypotheses(bundle, registry)
    if adapter.disposition is AdapterDisposition.ABSTAIN:
        return _abstaining_decision("adapter_" + adapter.reason, compilation, policy)
    grid_commitment = _exact_grid_commitment(adapter)
    if (
        compilation.bundle_content_id != bundle.content_id
        or compilation.registry_id != registry.registry_id
        or compilation.adapter_result_id != adapter.result_id
        or compilation.candidate_grid_commitment_id
        != grid_commitment.commitment_id
    ):
        return _abstaining_decision("bridge_provenance_drift", compilation, policy)
    evaluations = compilation.evaluations
    candidate_ids = tuple(sorted(item.candidate_id for item in evaluations))
    if candidate_ids != grid_commitment.expected_candidate_ids:
        return _abstaining_decision("incomplete_candidate_grid", compilation, policy)
    if len(evaluations) > policy.maximum_candidate_count:
        return _abstaining_decision("candidate_budget_exceeded", compilation, policy)
    if any(item.status is ExactCandidateStatus.ERROR for item in evaluations):
        return _abstaining_decision("candidate_evaluation_error", compilation, policy)
    if policy.require_complete_family_coverage and {
        item.law_kind for item in evaluations
    } != set(LawKind):
        return _abstaining_decision("incomplete_family_coverage", compilation, policy)

    groups: dict[
        tuple[LawKind, tuple[tuple[str, str], ...]],
        list[ExactCandidateEvaluation],
    ] = {}
    for item in evaluations:
        groups.setdefault((item.law_kind, item.role_binding), []).append(item)
    passing_groups = tuple(
        key
        for key, items in groups.items()
        if any(item.status is ExactCandidateStatus.PASS for item in items)
    )
    if not passing_groups:
        reason = (
            "nonidentifiable_interval_overlap"
            if any(
                item.status is ExactCandidateStatus.INCONCLUSIVE
                for item in evaluations
            )
            else "no_passing_structure"
        )
        return _abstaining_decision(reason, compilation, policy)
    if len(passing_groups) != 1:
        return _abstaining_decision("multiple_passing_structures", compilation, policy)
    selected_key = passing_groups[0]
    selected_items = tuple(groups[selected_key])
    if any(
        item.status is ExactCandidateStatus.INCONCLUSIVE for item in selected_items
    ):
        return _abstaining_decision(
            "selected_structure_has_inconclusive_scale",
            compilation,
            policy,
        )
    passing_scales = tuple(
        sorted(
            {
                item.scale_hypothesis_id
                for item in selected_items
                if item.status is ExactCandidateStatus.PASS
            }
        )
    )
    if policy.require_scale_competitor and len(
        {item.scale_hypothesis_id for item in selected_items}
    ) < 2:
        return _abstaining_decision("missing_scale_competitor", compilation, policy)
    if policy.require_binding_competitor and not any(
        item.law_kind is selected_key[0] and item.role_binding != selected_key[1]
        for item in evaluations
    ):
        return _abstaining_decision("missing_binding_competitor", compilation, policy)
    selected_upper = min(
        item.normalized_interval.upper_fraction
        for item in selected_items
        if item.status is ExactCandidateStatus.PASS
        and item.normalized_interval is not None
    )
    competitors = tuple(
        item for key, items in groups.items() if key != selected_key for item in items
    )
    if not competitors:
        return _abstaining_decision(
            "missing_structural_competitor",
            compilation,
            policy,
        )
    if any(
        item.status is ExactCandidateStatus.INCONCLUSIVE for item in competitors
    ):
        return _abstaining_decision(
            "inconclusive_structural_competitor",
            compilation,
            policy,
        )
    competitor_lower = min(
        item.normalized_interval.lower_fraction
        for item in competitors
        if item.normalized_interval is not None
    )
    margin = competitor_lower - selected_upper
    if (
        margin.numerator.bit_length()
        > DEFAULT_EXACT_BRIDGE_POLICY.maximum_fraction_bit_length
        or margin.denominator.bit_length()
        > DEFAULT_EXACT_BRIDGE_POLICY.maximum_fraction_bit_length
    ):
        return _abstaining_decision(
            "RESOURCE_LIMIT:selection_margin_bit_length",
            compilation,
            policy,
        )
    if margin < policy.minimum_structural_margin.as_fraction():
        return _abstaining_decision(
            "insufficient_structural_margin",
            compilation,
            policy,
        )
    disposition = (
        ExactSelectionDisposition.UNIQUE_IDENTIFICATION
        if len(passing_scales) == 1
        else ExactSelectionDisposition.ADMISSIBLE_SCALE_SET
    )
    return ExactSelectorDecision(
        disposition=disposition,
        reason="unique_structure_with_exact_admissible_scales",
        selection_policy_id=policy.policy_id,
        bridge_result_id=compilation.result_id,
        bundle_content_id=compilation.bundle_content_id,
        uncertainty_result_id=compilation.uncertainty_result_id,
        candidate_grid_commitment_id=compilation.candidate_grid_commitment_id,
        evaluated_candidate_ids=candidate_ids,
        selected_law_kind=selected_key[0],
        selected_role_binding=selected_key[1],
        admissible_scale_hypothesis_ids=passing_scales,
        normalized_structural_margin=ExactFractionAtom.from_fraction(margin),
    )


def run_exact_rational_bridge(
    *,
    bundle: PublicEvidenceBundle,
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
) -> ExactBridgeRun | ExactBridgePreflightRejection:
    """Authoritative exact bridge entrypoint over raw public authorities.

    The caller cannot supply an uncertainty receipt, candidate grid,
    evaluation, commitment root, or selection.  The function performs bounded
    preflight first, then deterministically rebuilds every downstream receipt.
    """

    if type(bundle) is not PublicEvidenceBundle:
        raise TypeError("exact bridge run requires exact evidence bundle type")
    if type(theory) is not TheoryState:
        raise TypeError("exact bridge run requires exact theory type")
    if type(registry) is not Phase2BAdapterRegistry:
        raise TypeError("exact bridge run requires exact adapter registry type")
    bridge_policy = DEFAULT_EXACT_BRIDGE_POLICY
    selection_policy = DEFAULT_EXACT_SELECTION_POLICY
    shallow_error = _shallow_authority_size_preflight(
        bundle,
        theory,
        registry,
        bridge_policy,
    )
    if shallow_error is not None:
        return _preflight_rejection(
            reason=shallow_error,
            bundle=bundle,
            theory=theory,
            registry=registry,
            bridge_policy=bridge_policy,
            selection_policy=selection_policy,
        )
    try:
        _require_exact_authority_tree(bundle, theory, registry, bridge_policy)
    except _ResourceLimit as exc:
        return _preflight_rejection(
            reason=str(exc),
            bundle=bundle,
            theory=theory,
            registry=registry,
            bridge_policy=bridge_policy,
            selection_policy=selection_policy,
        )
    preflight_error = _bounded_bundle_preflight(bundle, registry, bridge_policy)
    if preflight_error is not None:
        return _preflight_rejection(
            reason=preflight_error,
            bundle=bundle,
            theory=theory,
            registry=registry,
            bridge_policy=bridge_policy,
            selection_policy=selection_policy,
        )
    theory_version_id = theory.version_id
    if registry.theory_version_id != theory_version_id:
        raise ValueError("exact bridge theory and registry disagree")
    bundle_content_id = bundle.content_id
    registry_id = registry.registry_id
    common = {
        "bundle_content_id": bundle_content_id,
        "theory_version_id": theory_version_id,
        "registry_id": registry_id,
        "bridge_policy_id": bridge_policy.policy_id,
        "selection_policy_id": selection_policy.policy_id,
    }
    _validate_registry(theory, registry)
    receipt = compile_bundle_uncertainty(bundle)
    compilation = _compile_exact_candidate_grid(
        bundle=bundle,
        uncertainty_receipt=receipt,
        theory=theory,
        registry=registry,
        policy=bridge_policy,
    )
    decision = _select_exact_candidate_grid(
        compilation,
        bundle=bundle,
        registry=registry,
        policy=selection_policy,
    )
    return ExactBridgeRun(
        **common,
        disposition=compilation.disposition,
        reason=compilation.reason,
        uncertainty_receipt=receipt,
        compilation=compilation,
        decision=decision,
    )


__all__ = ("run_exact_rational_bridge",)
