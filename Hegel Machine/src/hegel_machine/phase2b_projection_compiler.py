"""Conservative Phase-2B root/identity projection compilation mechanics.

This module closes the deterministic adapter-to-selector edge only for the
semantics that the current public wire actually specifies completely:

* observed Boolean values with ``not_applicable`` uncertainty;
* numeric values or intervals with component-wise ``absolute_bound`` radii;
* dimensionless, support-aligned point observations at root-scale hypotheses
  and explicit identity transform paths.

It deliberately does *not* assign semantics to temporal/spatial aggregation,
sampling, unit conversion, affine coordinates, split/merge, or coarse
graining.  Those operations need separately frozen executable contracts.
Unsupported observation uncertainty aborts the bundle atomically.  Unsupported
transform semantics become explicit error cells in the complete candidate
grid, so the public selector abstains instead of using a partial grid or a
midpoint approximation.  Non-degenerate numeric envelopes are also error
cells: evaluating only their corners would not, in general, prove a
conservative residual interval for a nonlinear verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from itertools import product
from math import fsum, inf, isfinite, nextafter, sqrt
from sys import float_info

from .hashing import stable_hash
from .laws import evaluate_law
from .phase2b_adapter import (
    AdapterDisposition,
    CandidateHypothesis,
    LawWireBinding,
    Phase2BAdapterRegistry,
    enumerate_candidate_hypotheses,
)
from .phase2b_selector import CandidateEvaluation, ClosedInterval
from .phase2b_wire import (
    BooleanValue,
    Missingness,
    NumericInterval,
    NumericValue,
    PublicEvidenceBundle,
    TransformOperation,
    TypedObservation,
    UncertaintyModel,
)
from .schema import LawKind, RelationLaw, TheoryState, require_tuple


class ObservationCompilationDisposition(str, Enum):
    COMPILED = "compiled"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"


class ProjectionCompilationDisposition(str, Enum):
    COMPLETE = "complete"
    ABSTAIN = "abstain"


class VerifierObservableKind(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    BOOLEAN = "boolean"


# This is verifier input shape, not a hidden answer table.  The registry is
# intentionally exact: adding a law observable requires an explicit compiler
# decision rather than an inferred scalar/vector coercion.
VERIFIER_OBSERVABLE_KINDS: tuple[
    tuple[LawKind, tuple[tuple[str, VerifierObservableKind], ...]], ...
] = (
    (
        LawKind.SYMMETRY,
        (
            ("common_codomains", VerifierObservableKind.BOOLEAN),
            ("forward", VerifierObservableKind.VECTOR),
            ("transformed", VerifierObservableKind.VECTOR),
        ),
    ),
    (
        LawKind.MONOTONICITY,
        tuple(
            (name, VerifierObservableKind.SCALAR)
            for name in ("direction", "x_high", "x_low", "y_high", "y_low")
        ),
    ),
    (
        LawKind.CONSERVATION,
        (
            ("boundary_observed", VerifierObservableKind.BOOLEAN),
            ("inflows", VerifierObservableKind.VECTOR),
            ("outflows", VerifierObservableKind.VECTOR),
            ("sinks", VerifierObservableKind.VECTOR),
            ("sources", VerifierObservableKind.VECTOR),
            ("storage_delta", VerifierObservableKind.SCALAR),
        ),
    ),
    (
        LawKind.COMPLEMENTARITY,
        tuple(
            (name, VerifierObservableKind.SCALAR)
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
            ("controlled_quantity_observed", VerifierObservableKind.BOOLEAN),
            ("deviation_after_response", VerifierObservableKind.SCALAR),
            ("deviation_before_response", VerifierObservableKind.SCALAR),
            ("disturbance_delta", VerifierObservableKind.SCALAR),
            ("disturbance_precedes_response", VerifierObservableKind.BOOLEAN),
            ("local_stability_window_observed", VerifierObservableKind.BOOLEAN),
            ("mitigation_margin", VerifierObservableKind.SCALAR),
            ("response_delta", VerifierObservableKind.SCALAR),
            ("response_margin", VerifierObservableKind.SCALAR),
            ("same_controlled_quantity", VerifierObservableKind.BOOLEAN),
            ("system_induced_response", VerifierObservableKind.BOOLEAN),
        ),
    ),
    (
        LawKind.LOCALITY,
        (
            ("blanket_observed", VerifierObservableKind.BOOLEAN),
            ("conditional_a", VerifierObservableKind.VECTOR),
            ("conditional_b", VerifierObservableKind.VECTOR),
            ("same_blanket_state", VerifierObservableKind.BOOLEAN),
        ),
    ),
)

_SAFE_PRODUCT_MAGNITUDE_MIN = sqrt(float_info.min)
_SAFE_PRODUCT_MAGNITUDE_MAX = sqrt(float_info.max)


@dataclass(frozen=True, slots=True)
class CompiledObservationValue:
    observation_id: str
    observation_content_id: str
    disposition: ObservationCompilationDisposition
    reason: str
    numeric_bounds: tuple[ClosedInterval, ...] = ()
    boolean_value: bool | None = None

    def __post_init__(self) -> None:
        require_tuple(self.numeric_bounds, "compiled observation numeric bounds")
        if not all(
            (self.observation_id, self.observation_content_id, self.reason)
        ):
            raise ValueError("compiled observation identity is incomplete")
        if self.disposition is ObservationCompilationDisposition.COMPILED:
            numeric = bool(self.numeric_bounds)
            boolean = self.boolean_value is not None
            if numeric == boolean:
                raise ValueError(
                    "compiled observation must carry exactly one value kind"
                )
        elif self.numeric_bounds or self.boolean_value is not None:
            raise ValueError("noncompiled observation cannot carry a value")

    @property
    def compilation_id(self) -> str:
        return stable_hash(self, prefix="phase2b_observation_compilation_")


@dataclass(frozen=True, slots=True)
class ProjectionCompilerPolicy:
    supported_transform_operations: tuple[TransformOperation, ...] = (
        TransformOperation.IDENTITY,
    )
    minimum_nonzero_numeric_magnitude: float = _SAFE_PRODUCT_MAGNITUDE_MIN
    maximum_numeric_magnitude: float = _SAFE_PRODUCT_MAGNITUDE_MAX

    def __post_init__(self) -> None:
        require_tuple(
            self.supported_transform_operations,
            "projection compiler supported transforms",
        )
        if self.supported_transform_operations != (TransformOperation.IDENTITY,):
            raise ValueError("v1 projection compiler is root/identity only")
        if (
            self.minimum_nonzero_numeric_magnitude
            != _SAFE_PRODUCT_MAGNITUDE_MIN
            or self.maximum_numeric_magnitude != _SAFE_PRODUCT_MAGNITUDE_MAX
        ):
            raise ValueError("v1 projection compiler numeric domain is frozen")

    @property
    def policy_id(self) -> str:
        return stable_hash(self, prefix="phase2b_projection_compiler_policy_")


DEFAULT_PROJECTION_COMPILER_POLICY = ProjectionCompilerPolicy()


@dataclass(frozen=True, slots=True)
class ProjectionCompilationResult:
    disposition: ProjectionCompilationDisposition
    reason: str
    bundle_content_id: str
    registry_id: str
    adapter_result_id: str
    candidate_grid_commitment_id: str | None
    compiler_policy_id: str
    evaluations: tuple[CandidateEvaluation, ...]

    def __post_init__(self) -> None:
        require_tuple(self.evaluations, "projection compiler evaluations")
        if not all(
            (
                self.reason,
                self.bundle_content_id,
                self.registry_id,
                self.adapter_result_id,
                self.compiler_policy_id,
            )
        ):
            raise ValueError("projection compiler result identity is incomplete")
        if self.disposition is ProjectionCompilationDisposition.COMPLETE:
            if self.candidate_grid_commitment_id is None or not self.evaluations:
                raise ValueError("complete projection compilation needs a full grid")
        elif self.candidate_grid_commitment_id is not None or self.evaluations:
            raise ValueError("abstaining projection compilation cannot return a grid")

    @property
    def result_id(self) -> str:
        return stable_hash(self, prefix="phase2b_projection_compilation_")


def compile_observation_absolute_bound(
    observation: TypedObservation,
) -> CompiledObservationValue:
    """Compile one observation without midpoint substitution.

    This is a diagnostic building block.  The authoritative candidate-grid
    entrypoint below performs bundle-atomic uncertainty preflight before using
    any compiled observation.
    """

    observation_content_id = stable_hash(
        observation,
        prefix="phase2b_typed_observation_",
    )
    common = {
        "observation_id": observation.observation_id,
        "observation_content_id": observation_content_id,
    }
    if observation.missingness is Missingness.MISSING:
        return CompiledObservationValue(
            **common,
            disposition=ObservationCompilationDisposition.MISSING,
            reason="missing_observation",
        )
    if observation.uncertainty.model is UncertaintyModel.STANDARD_ERROR:
        return CompiledObservationValue(
            **common,
            disposition=ObservationCompilationDisposition.UNSUPPORTED,
            reason="STANDARD_ERROR_UNSUPPORTED",
        )
    if isinstance(observation.value, BooleanValue):
        return CompiledObservationValue(
            **common,
            disposition=ObservationCompilationDisposition.COMPILED,
            reason="exact_boolean_compiled",
            boolean_value=observation.value.value,
        )
    if observation.uncertainty.model is not UncertaintyModel.ABSOLUTE_BOUND:
        return CompiledObservationValue(
            **common,
            disposition=ObservationCompilationDisposition.UNSUPPORTED,
            reason="numeric_uncertainty_model_unsupported",
        )
    if isinstance(observation.value, NumericValue):
        exact_bounds = tuple(
            (
                Fraction.from_float(value) - Fraction.from_float(radius),
                Fraction.from_float(value) + Fraction.from_float(radius),
            )
            for value, radius in zip(
                observation.value.values,
                observation.uncertainty.radius,
                strict=True,
            )
        )
    elif isinstance(observation.value, NumericInterval):
        exact_bounds = tuple(
            (
                Fraction.from_float(lower) - Fraction.from_float(radius),
                Fraction.from_float(upper) + Fraction.from_float(radius),
            )
            for lower, upper, radius in zip(
                observation.value.lower,
                observation.value.upper,
                observation.uncertainty.radius,
                strict=True,
            )
        )
    else:
        return CompiledObservationValue(
            **common,
            disposition=ObservationCompilationDisposition.UNSUPPORTED,
            reason="observed_value_kind_unsupported",
        )
    raw_bounds = tuple(
        (
            _directed_float(lower, lower=True),
            _directed_float(upper, lower=False),
        )
        for lower, upper in exact_bounds
    )
    if any(not isfinite(endpoint) for pair in raw_bounds for endpoint in pair):
        return CompiledObservationValue(
            **common,
            disposition=ObservationCompilationDisposition.UNSUPPORTED,
            reason="absolute_bound_overflow",
        )
    return CompiledObservationValue(
        **common,
        disposition=ObservationCompilationDisposition.COMPILED,
        reason="absolute_bound_compiled",
        numeric_bounds=tuple(
            ClosedInterval(lower, upper) for lower, upper in raw_bounds
        ),
    )


def _directed_float(value: Fraction, *, lower: bool) -> float:
    try:
        rounded = float(value)
    except OverflowError:
        return -inf if value < 0 else inf
    if not isfinite(rounded):
        return rounded
    represented = Fraction.from_float(rounded)
    if lower and represented > value:
        return nextafter(rounded, -inf)
    if not lower and represented < value:
        return nextafter(rounded, inf)
    return rounded


def _observable_kind_registry() -> dict[LawKind, dict[str, VerifierObservableKind]]:
    return {kind: dict(rows) for kind, rows in VERIFIER_OBSERVABLE_KINDS}


def _validate_compiler_registry(
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
) -> None:
    registered = _observable_kind_registry()
    if set(registered) != set(LawKind):
        raise AssertionError("projection compiler law registry is incomplete")
    theory_by_id = {law.law_id: law for law in theory.relation_laws}
    binding_by_id = {binding.law_id: binding for binding in registry.law_bindings}
    if set(binding_by_id) != set(theory_by_id):
        raise ValueError("projection compiler adapter law registry differs")
    for law in theory.relation_laws:
        if set(registered[law.kind]) != set(law.required_observables):
            raise ValueError(
                f"projection compiler observable registry drift for {law.kind.value}"
            )
        binding = binding_by_id[law.law_id]
        if (
            binding.law_kind is not law.kind
            or tuple(sorted(role for role, _ in binding.role_ids))
            != tuple(sorted(law.roles))
            or binding.required_observable_ids
            != tuple(sorted(law.required_observables))
        ):
            raise ValueError("projection compiler adapter law binding differs")
    expected_observables = {
        observable
        for law in theory.relation_laws
        for observable in law.required_observables
    }
    if {
        channel.observable_id for channel in registry.observable_channels
    } != expected_observables:
        raise ValueError("projection compiler adapter channels differ")


def _error_evaluation(
    hypothesis: CandidateHypothesis,
    error_code: str,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id=hypothesis.candidate_id,
        law_kind=hypothesis.law_kind,
        role_binding=hypothesis.role_binding,
        scale_hypothesis_id=hypothesis.scale_hypothesis_id,
        residual=None,
        tolerance=None,
        completed=False,
        error_code=error_code,
        footprint_id=hypothesis.footprint_id,
    )


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
    # Some verifier controls (for example a preregistered interaction margin)
    # are law-context values rather than a single-role measurement.  The
    # public wire forbids an empty entity set, so they are bound to the whole
    # candidate role tuple instead of a caller-chosen sentinel entity.
    if not witness_roles:
        witness_roles = law.roles
    entities_by_role = dict(hypothesis.role_binding)
    wire_roles_by_role = dict(law_binding.role_ids)
    expected_entities = tuple(sorted(entities_by_role[role] for role in witness_roles))
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


def _value_options(
    compiled: CompiledObservationValue,
    expected_kind: VerifierObservableKind,
) -> tuple[object, ...] | str:
    if compiled.disposition is not ObservationCompilationDisposition.COMPILED:
        return compiled.reason
    if expected_kind is VerifierObservableKind.BOOLEAN:
        if compiled.boolean_value is None:
            return "observable_shape_mismatch"
        return (compiled.boolean_value,)
    if compiled.boolean_value is not None or not compiled.numeric_bounds:
        return "observable_shape_mismatch"
    if any(bound.lower != bound.upper for bound in compiled.numeric_bounds):
        return "nondegenerate_interval_residual_semantics_not_implemented"
    if expected_kind is VerifierObservableKind.SCALAR:
        if len(compiled.numeric_bounds) != 1:
            return "observable_shape_mismatch"
        bound = compiled.numeric_bounds[0]
        return tuple(dict.fromkeys((bound.lower, bound.upper)))
    component_options = tuple(
        tuple(dict.fromkeys((bound.lower, bound.upper)))
        for bound in compiled.numeric_bounds
    )
    return tuple(tuple(values) for values in product(*component_options))


def _numeric_domain_is_safe(
    observables: dict[str, object],
    policy: ProjectionCompilerPolicy,
) -> bool:
    numeric_values: list[float] = []
    for value in observables.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, tuple):
            numeric_values.extend(float(item) for item in value)
        else:
            numeric_values.append(float(value))
    try:
        absolute_sum = fsum(abs(value) for value in numeric_values)
    except (OverflowError, ValueError):
        return False
    if not isfinite(absolute_sum):
        return False
    return all(
        value == 0.0
        or policy.minimum_nonzero_numeric_magnitude
        <= abs(value)
        <= policy.maximum_numeric_magnitude
        for value in numeric_values
    )


def _compile_hypothesis(
    *,
    bundle: PublicEvidenceBundle,
    hypothesis: CandidateHypothesis,
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
    policy: ProjectionCompilerPolicy,
) -> CandidateEvaluation:
    transform_by_id = {item.transform_id: item for item in bundle.transform_catalog}
    for transform_id in hypothesis.transform_path_ids:
        transform = transform_by_id.get(transform_id)
        if transform is None:
            return _error_evaluation(hypothesis, "unknown_transform")
        if transform.operation not in policy.supported_transform_operations:
            return _error_evaluation(
                hypothesis,
                "unsupported_transform_semantics:" + transform.operation.value,
            )

    laws = {law.law_id: law for law in theory.relation_laws}
    law_bindings = {item.law_id: item for item in registry.law_bindings}
    law = laws[hypothesis.law_id]
    law_binding = law_bindings[hypothesis.law_id]
    quantity_by_observable = {
        item.observable_id: item.quantity_id for item in registry.observable_channels
    }
    kind_by_observable = _observable_kind_registry()[law.kind]
    options_by_name: list[tuple[str, tuple[object, ...]]] = []
    observations_used: list[TypedObservation] = []
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
            return _error_evaluation(hypothesis, observation_or_error)
        if observation_or_error.unit_dimension.si_exponents != (0,) * 7:
            return _error_evaluation(
                hypothesis,
                "nondimensionless_unit_semantics_not_implemented",
            )
        observations_used.append(observation_or_error)
        compiled = compile_observation_absolute_bound(observation_or_error)
        options_or_error = _value_options(
            compiled,
            kind_by_observable[observable_name],
        )
        if isinstance(options_or_error, str):
            return _error_evaluation(hypothesis, options_or_error)
        options_by_name.append((observable_name, options_or_error))

    first_observation = observations_used[0]
    if any(
        observation.temporal_support != first_observation.temporal_support
        for observation in observations_used[1:]
    ):
        return _error_evaluation(hypothesis, "unaligned_temporal_support")
    if any(
        observation.spatial_support != first_observation.spatial_support
        for observation in observations_used[1:]
    ):
        return _error_evaluation(hypothesis, "unaligned_spatial_support")

    tolerance_by_functional = {
        item.functional_id: item.tolerance for item in theory.violation_functionals
    }
    tolerance = tolerance_by_functional[law.violation_functional_id]
    residuals: list[float] = []
    option_product = product(*(options for _, options in options_by_name))
    for values in option_product:
        observables = {
            name: value
            for (name, _), value in zip(options_by_name, values, strict=True)
        }
        if not _numeric_domain_is_safe(observables, policy):
            return _error_evaluation(
                hypothesis,
                "verifier_numeric_domain_unsupported",
            )
        try:
            evaluation = evaluate_law(law.kind, observables, tolerance=tolerance)
        except (ArithmeticError, ValueError):
            return _error_evaluation(hypothesis, "verifier_numeric_error")
        if evaluation.abstained or evaluation.residual is None:
            return _error_evaluation(
                hypothesis,
                "verifier_abstained",
            )
        residuals.append(evaluation.residual)
    if not residuals:
        return _error_evaluation(hypothesis, "empty_observable_assignment")
    return CandidateEvaluation(
        candidate_id=hypothesis.candidate_id,
        law_kind=hypothesis.law_kind,
        role_binding=hypothesis.role_binding,
        scale_hypothesis_id=hypothesis.scale_hypothesis_id,
        residual=ClosedInterval(min(residuals), max(residuals)),
        tolerance=ClosedInterval(tolerance, tolerance),
        completed=True,
        footprint_id=hypothesis.footprint_id,
    )


def compile_candidate_evaluations(
    *,
    bundle: PublicEvidenceBundle,
    theory: TheoryState,
    registry: Phase2BAdapterRegistry,
    policy: ProjectionCompilerPolicy = DEFAULT_PROJECTION_COMPILER_POLICY,
) -> ProjectionCompilationResult:
    """Compile a complete adapter grid, with errors represented in-grid."""

    if registry.theory_version_id != theory.version_id:
        raise ValueError("projection compiler theory and adapter registry disagree")
    _validate_compiler_registry(theory, registry)
    enumeration = enumerate_candidate_hypotheses(bundle, registry)
    if enumeration.disposition is AdapterDisposition.ABSTAIN:
        return ProjectionCompilationResult(
            disposition=ProjectionCompilationDisposition.ABSTAIN,
            reason="adapter_" + enumeration.reason,
            bundle_content_id=bundle.content_id,
            registry_id=registry.registry_id,
            adapter_result_id=enumeration.result_id,
            candidate_grid_commitment_id=None,
            compiler_policy_id=policy.policy_id,
            evaluations=(),
        )
    # The formal v1 selector admits absolute-bound numeric observations only.
    # Enforce that restriction across the whole public bundle before compiling
    # candidate footprints, so a caller cannot hide an unsupported channel in
    # a projection-disfavoured subset.
    for observation in bundle.observations:
        compiled_observation = compile_observation_absolute_bound(observation)
        if (
            observation.missingness is Missingness.OBSERVED
            and compiled_observation.disposition
            is not ObservationCompilationDisposition.COMPILED
        ):
            return ProjectionCompilationResult(
                disposition=ProjectionCompilationDisposition.ABSTAIN,
                reason="bundle_uncertainty_preflight:"
                + compiled_observation.reason,
                bundle_content_id=bundle.content_id,
                registry_id=registry.registry_id,
                adapter_result_id=enumeration.result_id,
                candidate_grid_commitment_id=None,
                compiler_policy_id=policy.policy_id,
                evaluations=(),
            )
    grid_commitment = enumeration.candidate_grid_commitment
    evaluations = tuple(
        sorted(
            (
                _compile_hypothesis(
                    bundle=bundle,
                    hypothesis=hypothesis,
                    theory=theory,
                    registry=registry,
                    policy=policy,
                )
                for hypothesis in enumeration.hypotheses
            ),
            key=lambda item: item.candidate_id,
        )
    )
    if tuple(item.candidate_id for item in evaluations) != (
        grid_commitment.expected_candidate_ids
    ):
        raise AssertionError("projection compiler lost or reordered a candidate cell")
    return ProjectionCompilationResult(
        disposition=ProjectionCompilationDisposition.COMPLETE,
        reason="complete_candidate_evaluation_grid",
        bundle_content_id=bundle.content_id,
        registry_id=registry.registry_id,
        adapter_result_id=enumeration.result_id,
        candidate_grid_commitment_id=grid_commitment.commitment_id,
        compiler_policy_id=policy.policy_id,
        evaluations=evaluations,
    )


__all__ = (
    "CompiledObservationValue",
    "DEFAULT_PROJECTION_COMPILER_POLICY",
    "ObservationCompilationDisposition",
    "ProjectionCompilationDisposition",
    "ProjectionCompilationResult",
    "ProjectionCompilerPolicy",
    "VERIFIER_OBSERVABLE_KINDS",
    "VerifierObservableKind",
    "compile_candidate_evaluations",
    "compile_observation_absolute_bound",
)
